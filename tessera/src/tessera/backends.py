"""LLM backend abstraction.

Tessera has three sites that call an LLM — per-session narrative extraction,
cross-session synthesis, and the weekly experiment evaluator. All three
historically used the Anthropic Claude SDK (routing through the user's
authenticated `claude` CLI). This module adds a thin abstraction so the
same call sites can use the Codex CLI or Gemini CLI instead, letting users
A/B which agent's model produces better tessera output for their workflow.

Design choices:
- The backend interface is one async method: `complete(prompt, model, *,
  system_prompt) -> str`. Raw text in, raw text out. Retry, JSON extraction,
  and validation stay at the call site (each has its own recovery rules).
- Backends do NOT manage their own auth. We assume the user already ran
  `claude login` / `codex login` / `gemini auth login` — same as `tessera
  doctor` checks for.
- Each backend has its own DEFAULT_MODEL so `--model` defaults match the
  selected backend without the user having to know which Claude/OpenAI/
  Google model name to pass.
- CLI backends (Codex, Gemini) shell out via asyncio subprocess; the
  Claude backend uses `claude_agent_sdk.query` directly. The interface
  hides the difference.

NOTE: Backends are NOT interchangeable for output structure. Sonnet's
JSON reliability differs from gpt-5's differs from gemini-2.5's. The
JSON recovery in `narratives/synthesis._extract_json` handles Sonnet
quirks; expect to add backend-specific quirks as users report them.
"""
from __future__ import annotations

import asyncio
import os
import shutil
from abc import ABC, abstractmethod


DEFAULT_SYSTEM_PROMPT = "Return only valid JSON. No markdown fence, no preamble."

# Hard timeout for CLI subprocesses (codex/gemini/agy). Synthesis prompts
# can legitimately take 5-10 min; we set 15 to leave headroom but cap the
# blast radius of a hung CLI (observed: codex got stuck on one session
# overnight and froze the entire bake-off until manually killed).
_CLI_TIMEOUT_SEC = 900


class LLMBackend(ABC):
    """A pluggable LLM backend. Stateless — instances are cheap to create."""

    name: str = ""
    default_model: str = ""
    cli_binary: str = ""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Send a prompt to the model and return the raw text response.

        Args:
            prompt: The user/turn prompt.
            model: Model identifier. Empty string means "use backend default."
            system_prompt: System-level instructions. If None, uses the
                shared DEFAULT_SYSTEM_PROMPT ('Return only valid JSON...').

        Returns:
            The raw text response from the model. JSON parsing, recovery,
            and validation are caller responsibilities.

        Raises:
            RuntimeError: when the backend CLI is missing, the call fails,
                or auth is broken. The caller decides whether to retry.
        """

    def available(self) -> bool:
        """Best-effort check that this backend can actually run.

        Default: check the CLI binary is on PATH. Subclasses override if
        availability depends on more (auth state, env vars, etc.).
        """
        if not self.cli_binary:
            return True  # SDK-only backends bypass this
        return shutil.which(self.cli_binary) is not None


class ClaudeSDKBackend(LLMBackend):
    """The original tessera backend — Anthropic Claude SDK routing through
    the user's authenticated `claude` CLI. Zero API key required."""

    name = "claude"
    default_model = "claude-sonnet-4-6"
    cli_binary = "claude"

    async def complete(self, prompt, model, *, system_prompt=None):
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )
        collected = ""
        opts = ClaudeAgentOptions(
            model=model or self.default_model,
            allowed_tools=[],
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        )
        agen = query(prompt=prompt, options=opts)
        try:
            async for message in agen:
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            collected += block.text
                elif isinstance(message, ResultMessage):
                    break
        finally:
            # Explicit aclose avoids GC racing a later query() with
            # "generator already running" — same fix as the existing
            # extractor/synthesis call sites.
            await agen.aclose()
        return collected


class CodexCLIBackend(LLMBackend):
    """Shell out to OpenAI's `codex exec` for headless completion.

    Uses `--output-last-message <FILE>` to capture ONLY the agent's final
    response — stdout includes session metadata, prompt echo, token counts,
    and the response, which is too noisy to parse reliably.

    Model selection: when default_model is empty we omit `--model` and let
    codex pick its session default. ChatGPT-subscription auth has a
    different model whitelist than API-key auth; the safest default is
    'whatever codex picks for this session.'
    """

    name = "codex"
    default_model = ""  # empty → omit --model; codex uses session default
    cli_binary = "codex"

    async def complete(self, prompt, model, *, system_prompt=None):
        import tempfile
        full_prompt = (
            f"{system_prompt or DEFAULT_SYSTEM_PROMPT}\n\n{prompt}"
        )
        # --output-last-message writes only the final answer; the file is
        # consumed once and unlinked. Tempfile lives in /tmp by default.
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False, prefix="tessera-codex-"
        ) as f:
            out_file = f.name
        try:
            cmd = [
                self.cli_binary, "exec",
                "--skip-git-repo-check",
                "--sandbox", "read-only",
                "--output-last-message", out_file,
                "-",  # read prompt from stdin
            ]
            chosen = model or self.default_model
            if chosen:
                cmd[2:2] = ["--model", chosen]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=full_prompt.encode()),
                    timeout=_CLI_TIMEOUT_SEC,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise RuntimeError(
                    f"codex exec hung past {_CLI_TIMEOUT_SEC}s — killed"
                )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"codex exec failed (exit {proc.returncode}): "
                    f"{stderr.decode(errors='replace')[:500]}"
                )
            try:
                with open(out_file, encoding="utf-8") as f:
                    return f.read()
            except OSError:
                # Codex didn't write the file (rare — usually means auth or
                # rate-limit error didn't surface as a non-zero exit). Fall
                # back to raw stdout so the retry layer can see the signal.
                return stdout.decode(errors="replace")
        finally:
            import os
            try:
                os.unlink(out_file)
            except OSError:
                pass


class GeminiCLIBackend(LLMBackend):
    """Shell out to Google's `gemini -p` for headless completion.

    Uses `-o json` for clean parsing — the JSON envelope's `.response`
    field is the model's exact output. `-y` enables YOLO mode (auto-approve
    tool calls, required for headless), `--skip-trust` bypasses the
    workspace-trust prompt.
    """

    name = "gemini"
    default_model = ""  # empty → omit -m; gemini auto-selects (routes to flash/pro)
    cli_binary = "gemini"

    async def complete(self, prompt, model, *, system_prompt=None):
        import json as _json
        full_prompt = (
            f"{system_prompt or DEFAULT_SYSTEM_PROMPT}\n\n{prompt}"
        )
        cmd = [
            self.cli_binary,
            "-p", full_prompt,
            "-y",
            "-o", "json",
            "--skip-trust",
        ]
        chosen = model or self.default_model
        if chosen:
            cmd.extend(["-m", chosen])
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_CLI_TIMEOUT_SEC
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"gemini -p hung past {_CLI_TIMEOUT_SEC}s — killed")
        if proc.returncode != 0:
            raise RuntimeError(
                f"gemini -p failed (exit {proc.returncode}): "
                f"{stderr.decode(errors='replace')[:500]}"
            )
        raw = stdout.decode(errors="replace")
        try:
            envelope = _json.loads(raw)
            return envelope.get("response", raw)
        except _json.JSONDecodeError:
            # Should never happen with -o json, but fall back gracefully.
            return raw


class AntigravityBackend(LLMBackend):
    """Shell out to Google's Antigravity CLI (`agy`).

    Antigravity is a multi-provider agentic runtime — its `models` list
    includes Gemini 3.5 Flash, Gemini 3.1 Pro, Claude Sonnet/Opus 4.6, and
    GPT-OSS 120B. Tessera exposes it as one backend; pick the actual model
    via --model "Gemini 3.5 Flash (Medium)" (note the spaces and parens —
    agy's model names are display strings).

    Two agentic quirks we work around:
    - It narrates ("I will check the files...") before producing output.
      We run from a clean tempdir so it has nothing to explore, AND the
      existing JSON extractor finds the balanced {...} block at the end.
    - It defaults to interactive tool prompts. --dangerously-skip-permissions
      auto-approves so it can answer in headless mode.
    """

    name = "antigravity"
    # Bake-off (29 sessions, locked input, calibration-audited):
    # - Flash High matched ground truth on the headline (22/22), produced 21
    #   items including a unique temporal meta-insight, and ran 4× faster
    #   than Claude Sonnet 4.6.
    # - Flash Medium hung past 15 min on the synthesis prompt — avoid.
    # - Flash Low matched speed but undercounted by 6 (16 vs 22) and
    #   surfaced only 5 items.
    # → High is the calibrated default. The user can still override via --model.
    default_model = "Gemini 3.5 Flash (High)"
    cli_binary = "agy"

    async def complete(self, prompt, model, *, system_prompt=None):
        import tempfile
        full_prompt = (
            f"{system_prompt or DEFAULT_SYSTEM_PROMPT}\n\n{prompt}"
        )
        # Run in a fresh tempdir so agy's "let me explore the workspace"
        # default doesn't dump file listings into stdout before answering.
        with tempfile.TemporaryDirectory(prefix="tessera-agy-") as work_dir:
            cmd = [
                self.cli_binary,
                "--model", model or self.default_model,
                "--dangerously-skip-permissions",
                "--print-timeout", "10m",  # match tessera's slowest synth budget
                "-p", full_prompt,
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=work_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=_CLI_TIMEOUT_SEC
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise RuntimeError(f"agy -p hung past {_CLI_TIMEOUT_SEC}s — killed")
            if proc.returncode != 0:
                raise RuntimeError(
                    f"agy -p failed (exit {proc.returncode}): "
                    f"{stderr.decode(errors='replace')[:500]}"
                )
            return stdout.decode(errors="replace")


_REGISTRY: dict[str, type[LLMBackend]] = {
    "claude": ClaudeSDKBackend,
    "codex": CodexCLIBackend,
    "gemini": GeminiCLIBackend,
    "antigravity": AntigravityBackend,
}


def list_backends() -> list[str]:
    """Names of registered backends in display order."""
    return list(_REGISTRY)


def get_backend(name: str | None = None) -> LLMBackend:
    """Resolve a backend by name. Falls back to env var, then to whichever
    backend is installed.

    Resolution order:
      1. explicit name (--backend X on the CLI)
      2. $TESSERA_BACKEND env var
      3. antigravity (Flash 3.5 High) IF `agy` is installed
      4. claude (always-on safety net)

    Why antigravity (after evidence): we ran the same 29 sessions
    through THREE rounds of 3× consistency checks with progressively
    better recovery. With all the recovery in place (raw_decode for
    trailing prose, escape-cleanup for invalid \\<char>, hung-past
    retry, rogue-tool-use retry), Flash 3.5 High hit:
      - 0/3 hard failures (down from 2/3 pre-fix)
      - spread [21, 22, 21] — 1 exact hit, 2 off by -1
      - cost ~$0.03-0.09 per success
    Compared to Claude's 3× check on the same input:
      - 1/3 hard failure (Sonnet escape collapse — partial fix only)
      - spread [17, 17, fail] — 0 exact hits, off by -5 when wrong
      - cost ~$0.24 per call

    Flash isn't perfectly calibrated, but neither is Claude. Both wrong
    counts get caught automatically by the calibration audit
    (narratives.calibration) which grades quantified claims against the
    narratives and surfaces ✗ inline. Given equal "audit catches it"
    safety, Flash's reliability + cost + speed make it the better
    default. Override either via --backend or $TESSERA_BACKEND.

    Raises:
        ValueError: if `name` is not a registered backend.
    """
    requested = (name or os.environ.get("TESSERA_BACKEND") or "").lower()
    if requested:
        if requested not in _REGISTRY:
            raise ValueError(
                f"Unknown backend '{requested}'. Available: {list(_REGISTRY)}"
            )
        return _REGISTRY[requested]()
    # No explicit request — pick antigravity if installed, else claude.
    agy = _REGISTRY["antigravity"]()
    if agy.available():
        return agy
    return _REGISTRY["claude"]()


def default_model_for(backend_name: str | None) -> str:
    """Return the default model id for a backend without instantiating it."""
    name = (backend_name or "claude").lower()
    cls = _REGISTRY.get(name)
    return cls.default_model if cls else ClaudeSDKBackend.default_model
