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
            stdout, stderr = await proc.communicate(input=full_prompt.encode())
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
        stdout, stderr = await proc.communicate()
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


_REGISTRY: dict[str, type[LLMBackend]] = {
    "claude": ClaudeSDKBackend,
    "codex": CodexCLIBackend,
    "gemini": GeminiCLIBackend,
}


def list_backends() -> list[str]:
    """Names of registered backends in display order."""
    return list(_REGISTRY)


def get_backend(name: str | None = None) -> LLMBackend:
    """Resolve a backend by name. Falls back to env var then 'claude'.

    Resolution order: explicit name > $TESSERA_BACKEND > 'claude'.

    Raises:
        ValueError: if `name` is not a registered backend.
    """
    resolved = (name or os.environ.get("TESSERA_BACKEND") or "claude").lower()
    if resolved not in _REGISTRY:
        raise ValueError(
            f"Unknown backend '{resolved}'. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[resolved]()


def default_model_for(backend_name: str | None) -> str:
    """Return the default model id for a backend without instantiating it."""
    name = (backend_name or "claude").lower()
    cls = _REGISTRY.get(name)
    return cls.default_model if cls else ClaudeSDKBackend.default_model
