"""Run the example in each README and check the output it documents."""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
READMES = {"README.md": "## Example", "README.ja.md": "## 例"}
_PYTHON = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _example(name: str) -> str:
    text = (ROOT / name).read_text(encoding="utf-8")
    section = text.split(READMES[name], 1)[1]
    match = _PYTHON.search(section)
    assert match is not None, f"no python block after {READMES[name]!r} in {name}"
    return match.group(1)


def _documented_output(code: str) -> list[str]:
    """Comment lines that directly follow a ``print(...)`` line."""
    lines = code.splitlines()
    return [
        line.removeprefix("# ")
        for prev, line in zip(lines, lines[1:], strict=False)
        if prev.startswith("print(") and line.startswith("# ")
    ]


@pytest.mark.integration
@pytest.mark.parametrize("name", list(READMES))
def test_readme_example(name: str) -> None:
    code = _example(name)
    expected = _documented_output(code)
    assert len(expected) == 2

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, name, "exec"), {"__name__": "__readme__"})

    assert out.getvalue().splitlines() == expected


def test_readme_examples_are_identical_code() -> None:
    def strip_comments(code: str) -> list[str]:
        return [re.sub(r"\s*#.*$", "", line) for line in code.splitlines()]

    assert strip_comments(_example("README.md")) == strip_comments(_example("README.ja.md"))
