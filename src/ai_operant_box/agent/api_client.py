"""LLM client wrappers.

:class:`ClaudeCLIClient` drives the ``claude`` command-line tool (Claude Code)
in print mode and needs no API key: calls run on the user's Claude
subscription. It is the default route for real LLM subjects.

:class:`AnthropicClient` wraps the Anthropic SDK and needs an API key. The
SDK is an *optional* dependency. Importing this module is safe
without it; only constructing :class:`AnthropicClient` actually triggers the
import, which raises ``RuntimeError`` if the package is missing.

API keys are read from the ``ANTHROPIC_API_KEY`` environment variable. They
are never accepted as positional constructor arguments to discourage
hardcoding.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Iterable, Sequence
from typing import Any, Protocol


class LLMClientProtocol(Protocol):
    """Minimal generate() interface used by :class:`LLMSubject`."""

    def generate(self, model: str, system: str, messages: Iterable[dict]) -> str:
        ...


class AnthropicClient:
    """Thin wrapper around the Anthropic Python SDK (lazy-imported).

    Parameters
    ----------
    default_model : str
        Default model identifier; can be overridden per :meth:`generate` call.
    max_tokens : int
        Maximum tokens to request per completion.
    api_key_env : str
        Name of the environment variable holding the API key. Default
        ``ANTHROPIC_API_KEY``.

    Notes
    -----
    The actual ``anthropic`` import happens inside ``__init__`` so that simply
    importing this module (and the package) does not require the SDK.
    """

    def __init__(
        self,
        default_model: str = "claude-3-5-sonnet-latest",
        max_tokens: int = 64,
        api_key_env: str = "ANTHROPIC_API_KEY",
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise RuntimeError(
                "The 'anthropic' package is required to use AnthropicClient. "
                "Install it via `pip install ai-operant-box[llm]` or "
                "`pip install anthropic>=0.39`."
            ) from exc

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable '{api_key_env}' is not set. "
                "Anthropic API keys must never be hardcoded."
            )

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self.default_model = default_model
        self.max_tokens = max_tokens

    def generate(
        self,
        model: str | None,
        system: str,
        messages: Iterable[dict],
    ) -> str:
        """Call Claude and return the first text block of the response."""
        msgs = list(messages)
        response: Any = self._client.messages.create(
            model=model or self.default_model,
            max_tokens=self.max_tokens,
            system=system,
            messages=msgs,
        )
        for block in getattr(response, "content", []):
            text = getattr(block, "text", None)
            if text:
                return str(text).strip()
        return ""


class ClaudeCLIError(RuntimeError):
    """The ``claude`` process failed, exited, timed out, or returned an error."""


class ClaudeCLIClient:
    """One resident ``claude -p`` process; each :meth:`send` is one user turn.

    The process is started with ``--input-format stream-json
    --output-format stream-json`` and kept alive, so the conversation held
    by the process *is* the subject's in-context history. Only the new turn
    is sent each step, and the reply is awaited before the next turn is
    written (closing stdin early can drop the last reply).

    ``--system-prompt`` replaces the default Claude Code system prompt.
    Appending to it instead (``--append-system-prompt``) leaves the coding
    assistant persona in place, and small models then decline the role.
    Tools, MCP servers, settings files (and their hooks), and session
    persistence are all switched off, and the process runs in an empty
    temporary directory so no project ``CLAUDE.md`` is picked up.

    Parameters
    ----------
    model : str
        Model alias or full name passed to ``--model`` (``"haiku"``,
        ``"sonnet"``, ``"opus"``, ...).
    system_prompt : str
        Replaces the default system prompt.
    claude_command : str or sequence of str
        Executable (and leading arguments) to run. Default ``"claude"``.
    timeout : float
        Seconds to wait for one reply before raising :class:`ClaudeCLIError`.
    extra_args : sequence of str
        Appended to the command line.
    cwd : str, optional
        Working directory. Default: a fresh temporary directory.

    Attributes
    ----------
    latencies : list of float
        Seconds from writing each turn to receiving its reply.
    last_result : dict or None
        The latest ``result`` event. Its ``total_cost_usd`` is the
        conversation's running API-equivalent price; on a subscription it is
        not billed but tracks how much of the usage limit the run consumed.
    rate_limit_info : dict or None
        The latest ``rate_limit_event`` payload (usage-limit utilisation),
        when the CLI emits one.
    """

    def __init__(
        self,
        model: str = "haiku",
        system_prompt: str = "",
        claude_command: str | Sequence[str] = "claude",
        timeout: float = 120.0,
        extra_args: Sequence[str] = (),
        cwd: str | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        if isinstance(claude_command, str):
            self._command = [claude_command]
        else:
            self._command = list(claude_command)
        self.timeout = timeout
        self.extra_args = list(extra_args)
        self.cwd = cwd
        self.latencies: list[float] = []
        self.last_result: dict[str, Any] | None = None
        self.rate_limit_info: dict[str, Any] | None = None
        self._proc: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=50)
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None

    # -- process lifecycle -------------------------------------------------

    def command_line(self) -> list[str]:
        """The full argument vector used to start the process."""
        return [
            *self._command,
            "-p",
            "--model", self.model,
            "--system-prompt", self.system_prompt,
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--tools", "",
            "--strict-mcp-config",
            "--setting-sources", "",
            "--no-session-persistence",
            *self.extra_args,
        ]

    def start(self) -> None:
        if self._proc is not None:
            return
        if len(self._command) == 1 and shutil.which(self._command[0]) is None:
            raise ClaudeCLIError(
                f"'{self._command[0]}' was not found on PATH. Install Claude Code "
                "and log in once with `claude` before using ClaudeCLIClient."
            )
        cwd = self.cwd
        if cwd is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="aob-claude-")
            cwd = self._tmpdir.name
        self._lines = queue.Queue()
        self._proc = subprocess.Popen(
            self.command_line(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=cwd,
        )
        threading.Thread(target=self._pump_stdout, args=(self._proc,), daemon=True).start()
        threading.Thread(target=self._pump_stderr, args=(self._proc,), daemon=True).start()

    def _pump_stdout(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _pump_stderr(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            self._stderr.append(line.rstrip())

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
                proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                proc.kill()
                proc.wait()
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def __enter__(self) -> ClaudeCLIClient:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- turns ---------------------------------------------------------------

    def send(self, text: str) -> str:
        """Send one user turn and return the reply text."""
        self.start()
        proc = self._proc
        assert proc is not None and proc.stdin is not None
        started = time.monotonic()
        message = {"type": "user", "message": {"role": "user", "content": text}}
        try:
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()
        except OSError as exc:
            raise ClaudeCLIError(f"claude process is not accepting input: {exc}") from exc

        deadline = started + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ClaudeCLIError(f"no reply from claude within {self.timeout} s")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                proc.wait()
                tail = " | ".join(list(self._stderr)[-5:])
                raise ClaudeCLIError(
                    f"claude exited with code {proc.returncode} before replying: {tail}"
                )
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "rate_limit_event":
                self.rate_limit_info = event.get("rate_limit_info")
                continue
            if event.get("type") != "result":
                continue
            if event.get("is_error") or event.get("subtype") != "success":
                raise ClaudeCLIError(
                    f"claude returned an error ({event.get('subtype')}): {event.get('result')}"
                )
            self.latencies.append(time.monotonic() - started)
            self.last_result = event
            return str(event.get("result") or "").strip()

    def generate(
        self,
        model: str | None,
        system: str,
        messages: Iterable[dict],
    ) -> str:
        """Stateless :class:`LLMClientProtocol` call (slow; not the default).

        Starts a new process, sends the whole transcript as one turn, and
        closes it. Each step therefore re-sends and re-reads the full
        history; prefer :meth:`send` with
        :class:`~ai_operant_box.agent.llm_agent.StreamingLLMSubject`.
        """
        transcript = "\n".join(
            f"[{m.get('role', 'user')}] {m.get('content', '')}" for m in messages
        )
        one_shot = ClaudeCLIClient(
            model=model or self.model,
            system_prompt=system,
            claude_command=self._command,
            timeout=self.timeout,
            extra_args=self.extra_args,
            cwd=self.cwd,
        )
        try:
            return one_shot.send(transcript)
        finally:
            one_shot.close()


__all__ = ["AnthropicClient", "ClaudeCLIClient", "ClaudeCLIError", "LLMClientProtocol"]
