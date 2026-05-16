"""Tests for Codex apply_patch file-path extraction.

Bug uncovered by persona-review #3: 58 of 152 dogfood sessions tagged as
'exploration' (no files touched) were actually Codex sessions doing real
work via apply_patch. The deterministic extractor only knew Claude's tool
input shape (file_path / path keys). This regression covers the fix.
"""

from __future__ import annotations

from tessera.narratives.deterministic import (
    _file_paths_from_input,
    _paths_from_apply_patch,
)


def test_apply_patch_update_file_extracted():
    patch = """*** Update File: src/foo.py
@@ -1,3 +1,4 @@
 line1
+new_line
"""
    assert _paths_from_apply_patch(patch) == ["src/foo.py"]


def test_apply_patch_multiple_files_extracted():
    patch = """*** Update File: src/a.py
@@ -1 +1,2 @@
 a
*** Add File: src/b.py
+brand_new
*** Delete File: old/legacy.py
"""
    assert _paths_from_apply_patch(patch) == [
        "src/a.py",
        "src/b.py",
        "old/legacy.py",
    ]


def test_apply_patch_via_main_extractor():
    """Confirm the apply_patch shape flows through _file_paths_from_input."""
    tool_input = {
        "patch": "*** Update File: src/foo.py\n@@ -1 +1,2 @@\n line\n+new\n"
    }
    assert _file_paths_from_input(tool_input, "apply_patch") == ["src/foo.py"]


def test_apply_patch_with_input_key_alias():
    """Codex sometimes wraps the patch in an `input` key — handle both."""
    tool_input = {"input": "*** Update File: scripts/build.sh\n+#!/bin/sh\n"}
    assert _file_paths_from_input(tool_input, "apply_patch") == ["scripts/build.sh"]


def test_apply_patch_with_diff_key_alias():
    tool_input = {"diff": "*** Add File: lib/util.ts\n+export const x = 1;\n"}
    assert _file_paths_from_input(tool_input, "apply_patch") == ["lib/util.ts"]


def test_claude_file_path_still_works():
    """Make sure the Codex path didn't break the Claude path."""
    assert _file_paths_from_input({"file_path": "/abs/path.py"}, "Edit") == [
        "/abs/path.py"
    ]


def test_multi_edit_still_works():
    edits = [{"file_path": "a.py"}, {"file_path": "b.py"}]
    assert _file_paths_from_input({"edits": edits}, "MultiEdit") == ["a.py", "b.py"]


def test_apply_patch_paths_with_spaces():
    """Header allows spaces — paths with spaces should round-trip."""
    patch = "*** Update File: src/has spaces in it.py\n@@\n"
    assert _paths_from_apply_patch(patch) == ["src/has spaces in it.py"]


def test_no_apply_patch_no_panic():
    """Empty / non-patch inputs return empty list, not error."""
    assert _file_paths_from_input(None) == []
    assert _file_paths_from_input({"unrelated": "field"}, "apply_patch") == []


def test_codex_raw_string_input():
    """Codex sends apply_patch as a raw unified-diff string (no JSON wrap),
    keyed under tool_name='apply_patch'. Helper must handle that shape."""
    raw = "*** Begin Patch\n*** Add File: src/foo.py\n+x = 1\n*** End Patch"
    assert _file_paths_from_input(raw, "apply_patch") == ["src/foo.py"]


def test_codex_raw_string_input_detected_by_prefix_even_without_tool_name():
    """If we don't have tool_name but the string starts with the patch
    sentinel, treat it as a patch — defends against normalizer rename drift."""
    raw = "*** Begin Patch\n*** Update File: a.ts\n@@\n"
    assert _file_paths_from_input(raw) == ["a.ts"]


def test_random_string_without_patch_marker_returns_empty():
    """Don't be over-eager — random strings shouldn't be parsed as patches."""
    assert _file_paths_from_input("just some text", "apply_patch") == []
    assert _file_paths_from_input("not json at all") == []
