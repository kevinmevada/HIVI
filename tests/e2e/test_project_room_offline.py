"""ProjectRoom generate → test → retry loop, offline (fake LLM + in-memory Hindsight)."""
from __future__ import annotations

from pathlib import Path

import pytest

from hivi.memory.manager import MemoryManager
from hivi.memory.policies import ActorRole
from hivi.memory.providers.memory_store import InMemoryProvider
from hivi.orchestrator.project_room import (
    ProjectRoom,
    ProjectRoomPaths,
    extract_python,
    verify_calculator_file,
)

GOOD = '''```python
def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def div(a, b):
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a / b
```'''

# div returns None for zero instead of raising: the exact failure HIVI remembers
BAD = GOOD.replace('raise ZeroDivisionError("division by zero")', "return None")


def _paths(tmp_path: Path) -> ProjectRoomPaths:
    root = tmp_path / "root"
    project_dir = root / "data" / "projects" / "calculator"
    data = root / "data" / "hivi"
    project_dir.mkdir(parents=True)
    data.mkdir(parents=True)
    return ProjectRoomPaths(
        root=root,
        project_dir=project_dir,
        artifact_path=project_dir / "calculator.py",
        ledger_db=data / "hivi.sqlite",
        checkpoint_db=data / "langgraph_checkpoints.sqlite",
    )


class ScriptedLLM:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def _room(tmp_path: Path, llm, *, mm: MemoryManager | None = None, thread: str = "t1", **kw):
    paths = _paths(tmp_path) if not (tmp_path / "root").exists() else ProjectRoomPaths(
        root=tmp_path / "root",
        project_dir=tmp_path / "root/data/projects/calculator",
        artifact_path=tmp_path / "root/data/projects/calculator/calculator.py",
        ledger_db=tmp_path / "root/data/hivi/hivi.sqlite",
        checkpoint_db=tmp_path / "root/data/hivi/langgraph_checkpoints.sqlite",
    )
    mm = mm or MemoryManager(provider=InMemoryProvider(), db_path=paths.ledger_db)
    return ProjectRoom(
        paths=paths, project_id="001", thread_id=thread, preflight=False,
        generate_fn=llm, mm=mm, llm_review=False, **kw,
    )


def test_first_try_success_writes_model_output(tmp_path: Path):
    llm = ScriptedLLM(GOOD)
    room = _room(tmp_path, llm)
    try:
        out = room.run()
        assert out["deliverable_ok"] is True
        assert out["attempt"] == 1
        src = Path(out["artifact_path"]).read_text(encoding="utf-8")
        assert "def div" in src and "ZeroDivisionError" in src
        assert out["event_id"]
        assert len(llm.prompts) == 1
    finally:
        room.close()


def test_failure_is_fed_back_and_retried(tmp_path: Path):
    llm = ScriptedLLM(BAD, GOOD)
    room = _room(tmp_path, llm)
    try:
        out = room.run()
        assert out["deliverable_ok"] is True
        assert out["attempt"] == 2
        # attempt 2 prompt carries the concrete test failure from attempt 1
        assert "did not raise ZeroDivisionError" in llm.prompts[1]
        events = room.mm.ledger.list_for_project("001")
        kinds = [e.event_type for e in events]
        assert "test_failure" in kinds and "final_deliverable" in kinds
    finally:
        room.close()


def test_gives_up_after_max_attempts(tmp_path: Path):
    llm = ScriptedLLM(BAD)
    room = _room(tmp_path, llm, max_attempts=2)
    try:
        out = room.run()
        assert out["deliverable_ok"] is False
        assert out["attempt"] == 2
        kinds = [e.event_type for e in room.mm.ledger.list_for_project("001")]
        assert kinds.count("test_failure") == 2
        assert "user_approval" not in kinds
    finally:
        room.close()


def test_generation_error_does_not_crash_graph(tmp_path: Path):
    def boom(prompt: str) -> str:
        raise TimeoutError("backend busy")

    room = _room(tmp_path, boom, max_attempts=2)
    try:
        out = room.run()
        assert out["deliverable_ok"] is False
        assert "generation failed" in (out.get("verify_detail") or "")
    finally:
        room.close()


def test_recalled_memory_reaches_the_prompt(tmp_path: Path):
    paths = _paths(tmp_path)
    mm = MemoryManager(provider=InMemoryProvider(), db_path=paths.ledger_db)
    mm.record_event(
        project_id="001",
        event_type="test_failure",
        content="Calculator failed: division by zero on empty input",
        role=ActorRole.WORKER,
        sync_retain=True,
        enqueue=False,
    )
    llm = ScriptedLLM(GOOD)
    room = _room(tmp_path, llm, mm=mm)
    try:
        room.run(query="division by zero")
        assert "division by zero on empty input" in llm.prompts[0]
    finally:
        room.close()


def test_checkpoint_survives_restart(tmp_path: Path):
    room = _room(tmp_path, ScriptedLLM(GOOD), thread="resume")
    try:
        room.run()
    finally:
        room.close()
    room2 = _room(tmp_path, ScriptedLLM(GOOD), thread="resume")
    try:
        values = room2.get_state().values
        assert values["deliverable_ok"] is True
        assert Path(values["artifact_path"]).is_file()
    finally:
        room2.close()


@pytest.mark.parametrize(
    ("source", "expect"),
    [
        ("import sys\nsys.exit(3)\n", "verifier_crashed"),
        ("raise RuntimeError('boom')\n", "import_error"),
        ("def add(a, b):\n    return a + b\n", "missing_fns"),
    ],
)
def test_verifier_contains_bad_modules(tmp_path: Path, source: str, expect: str):
    f = tmp_path / "calculator.py"
    f.write_text(source, encoding="utf-8")
    ok, detail = verify_calculator_file(f)
    assert ok is False
    assert expect in detail


def test_verifier_times_out_on_hang(tmp_path: Path):
    f = tmp_path / "calculator.py"
    f.write_text("while True:\n    pass\n", encoding="utf-8")
    ok, detail = verify_calculator_file(f, timeout_s=2)
    assert ok is False and "timeout" in detail


def test_extract_python_variants():
    assert "def div" in extract_python(GOOD)
    assert extract_python("<think>plan</think>\ndef f():\n    return 1\n").startswith("def f")
    assert extract_python("Sorry, I can't help with that.") is None
