"""Calculator project-room runner: live inference + Hindsight + SQLite checkpoints.

Flow (LangGraph):

    load_project_context ─► generate_code ─► test_code ─┬─ pass ─► store_execution_memory ─► approve_deliverable ─► END
                               ▲                         ├─ fail, attempts left ─► record_failure ─┘ (back to generate_code)
                               └─────────────────────────┘
                                                         └─ fail, no attempts left ─► record_failure ─► END

The model writes calculator.py from the spec plus recalled project memory. A
deterministic test, run in a separate Python process with a timeout, is the
only acceptance gate. Each failure is written to the ledger and fed back into
the next attempt, so memory changes what the model does.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from hivi.memory.manager import MemoryManager
from hivi.memory.policies import ActorRole
from hivi.memory.providers.hindsight import HindsightProvider
from .memory_nodes import MemoryGraphState, make_memory_nodes
from .preflight import (
    PreflightResult,
    fetch_model_id,
    resolve_hindsight_url,
    resolve_inference_url,
    run_preflight,
)

CALCULATOR_SPEC = """\
Write a Python module named calculator.py with exactly these top-level functions:
- add(a, b) -> a + b
- sub(a, b) -> a - b
- mul(a, b) -> a * b
- div(a, b) -> a / b, and raise ZeroDivisionError when b == 0
Use only the standard library. No input(), no printing, no code at module level
other than the function definitions and an optional module docstring.
Reply with the complete file in one ```python code block and nothing else."""

GenerateFn = Callable[[str], str]
"""Takes a full prompt, returns the model's raw reply text."""

DEFAULT_MAX_ATTEMPTS = int(os.environ.get("HIVI_ROOM_MAX_ATTEMPTS", "3"))
_VERIFY_TIMEOUT_S = float(os.environ.get("HIVI_VERIFY_TIMEOUT_S", "20"))
_GENERATE_TIMEOUT_S = float(os.environ.get("HIVI_GENERATE_TIMEOUT_S", "300"))


@dataclass
class ProjectRoomPaths:
    root: Path
    project_dir: Path
    artifact_path: Path
    ledger_db: Path
    checkpoint_db: Path

    @classmethod
    def default(cls, root: Path | None = None) -> ProjectRoomPaths:
        base = Path(root) if root else Path(os.environ.get("HIVI_ROOT", Path.cwd()))
        project_dir = base / "data" / "projects" / "calculator"
        data_hivi = Path(os.environ.get("HIVI_DATA_DIR", base / "data" / "hivi"))
        data_hivi.mkdir(parents=True, exist_ok=True)
        project_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            root=base,
            project_dir=project_dir,
            artifact_path=project_dir / "calculator.py",
            ledger_db=data_hivi / "hivi.sqlite",
            checkpoint_db=data_hivi / "langgraph_checkpoints.sqlite",
        )


class ProjectRoom:
    """Owns MemoryManager, SQLite checkpointer, and the project-room graph."""

    def __init__(
        self,
        *,
        paths: ProjectRoomPaths | None = None,
        project_id: str = "001",
        thread_id: str = "calculator-room-001",
        preflight: bool = True,
        generate_fn: GenerateFn | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        mm: MemoryManager | None = None,
        llm_review: bool = True,
    ) -> None:
        self.paths = paths or ProjectRoomPaths.default()
        self.project_id = project_id
        self.thread_id = thread_id
        self.max_attempts = max(1, int(max_attempts))
        self.llm_review = llm_review
        self.inference_url = resolve_inference_url()
        self.hindsight_url = resolve_hindsight_url()
        self.model_id: str | None = None
        self._preflight: PreflightResult | None = None

        if preflight:
            self._preflight = run_preflight(project_id=project_id)
            self._preflight.raise_if_failed()
            self.model_id = self._preflight.model_id
        else:
            try:
                self.model_id = fetch_model_id(self.inference_url)
            except Exception:
                self.model_id = None

        if mm is None:
            provider = HindsightProvider(base_url=self.hindsight_url)
            mm = MemoryManager(provider=provider, db_path=self.paths.ledger_db)
        self.mm = mm
        self.generate_fn: GenerateFn = generate_fn or self._llm_generate

        self._ckpt_conn = sqlite3.connect(
            str(self.paths.checkpoint_db), check_same_thread=False
        )
        self.checkpointer = SqliteSaver(self._ckpt_conn)
        self.checkpointer.setup()
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ #
    # LLM calls
    # ------------------------------------------------------------------ #

    def _chat(self, prompt: str, *, max_tokens: int, temperature: float, timeout: float) -> str:
        model = self.model_id or fetch_model_id(self.inference_url)
        r = httpx.post(
            f"{self.inference_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return str(
            r.json().get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        )

    def _llm_generate(self, prompt: str) -> str:
        return self._chat(prompt, max_tokens=1024, temperature=0.2, timeout=_GENERATE_TIMEOUT_S)

    # ------------------------------------------------------------------ #
    # Graph
    # ------------------------------------------------------------------ #

    def _build_graph(self):
        nodes = make_memory_nodes(self.mm)
        room = self

        def generate_code(state: MemoryGraphState) -> dict[str, Any]:
            attempt = int(state.get("attempt") or 0) + 1
            prompt = build_generation_prompt(
                context_text=state.get("context_text") or "",
                feedback=state.get("generation_feedback"),
            )
            try:
                reply = room.generate_fn(prompt)
            except Exception as e:  # inference down / timeout
                return {
                    "attempt": attempt,
                    "generated_source": None,
                    "generation_feedback": f"generation failed: {type(e).__name__}: {e}",
                    "error": "generation failed",
                }
            source = extract_python(reply)
            return {
                "attempt": attempt,
                "generated_source": source,
                "generated_by": room.model_id or "injected",
                "generation_feedback": None if source else "reply contained no Python code block",
                "error": None if source else "no code in reply",
            }

        def test_code(state: MemoryGraphState) -> dict[str, Any]:
            source = state.get("generated_source")
            if not source:
                return {
                    "deliverable_ok": False,
                    "verify_detail": state.get("generation_feedback") or "nothing generated",
                }
            path = room.paths.artifact_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
            ok, detail = verify_calculator_file(path)
            return {
                "artifact_path": str(path),
                "evidence_path": str(path),
                "deliverable_ok": ok,
                "verify_detail": detail,
                "generation_feedback": None if ok else f"Your previous file failed the test: {detail}",
                "error": None if ok else "local verify failed",
            }

        def record_failure(state: MemoryGraphState) -> dict[str, Any]:
            """Failures are project knowledge: ledger now, Hindsight via the queue."""
            detail = state.get("verify_detail") or state.get("error") or "unknown failure"
            evt = room.mm.record_event(
                project_id=state["project_id"],
                event_type="test_failure",
                content=f"Calculator attempt {state.get('attempt')} failed: {detail}"[:400],
                role=ActorRole.WORKER,
                agent_id=state.get("agent_id"),
                task_id=state.get("task_id"),
                evidence_path=state.get("artifact_path"),
                enqueue=True,
                sync_retain=False,
            )
            return {"event_id": evt.id}

        def store_execution_memory(state: MemoryGraphState) -> dict[str, Any]:
            """Ledger + queue only — do not drain here (avoids LLM contention)."""
            path = Path(state.get("artifact_path") or room.paths.artifact_path)
            rel = (
                str(path.relative_to(room.paths.root))
                if path.is_relative_to(room.paths.root)
                else path.name
            )
            content = (
                f"Final deliverable written: {rel} (generated by {state.get('generated_by')}, "
                f"attempt {state.get('attempt')}). Calculator exposes add/sub/mul/div; "
                "div rejects zero divisor; passed local tests."
            )
            evt = room.mm.record_event(
                project_id=state["project_id"],
                event_type="final_deliverable",
                content=content,
                role=state.get("role") or ActorRole.WORKER.value,
                agent_id=state.get("agent_id"),
                task_id=state.get("task_id"),
                evidence_path=str(path),
                enqueue=True,
                sync_retain=False,
            )
            return {"event_id": evt.id, "last_event_type": "final_deliverable",
                    "last_event_content": content}

        def approve_deliverable(state: MemoryGraphState) -> dict[str, Any]:
            """Tests already passed. Optional LLM review is a note, never the gate."""
            path = Path(state.get("artifact_path") or room.paths.artifact_path)
            review = "llm_review_skipped"
            if room.llm_review:
                try:
                    answer = room._chat(
                        "Reply with exactly VERIFIED or REJECTED. Is this a calculator with "
                        "add/sub/mul/div and a zero-divisor guard?\n"
                        f"```python\n{path.read_text(encoding='utf-8')}\n```",
                        max_tokens=16, temperature=0, timeout=20.0,
                    )
                    review = answer.strip()[:120] or "llm_review_empty"
                except Exception as e:
                    review = f"llm_review_unavailable:{type(e).__name__}"
            # Keep retain text short — long paths trigger Colibrì extraction runaways.
            room.mm.record_event(
                project_id=room.project_id,
                event_type="user_approval",
                content=(
                    "Verified calculator deliverable. "
                    "Functions add, sub, mul, and div passed tests; div rejects zero."
                ),
                role=ActorRole.PROJECT_OWNER,
                evidence_path=str(path),
                sync_retain=True,
                enqueue=False,
                wait=True,
            )
            return {
                "deliverable_ok": True,
                "review_note": review,
                "verify_detail": f"{state.get('verify_detail')}; {review}",
                "error": None,
            }

        def after_test(state: MemoryGraphState) -> str:
            if state.get("deliverable_ok"):
                return "store_execution_memory"
            return "record_failure"

        def after_failure(state: MemoryGraphState) -> str:
            limit = int(state.get("max_attempts") or room.max_attempts)
            if int(state.get("attempt") or 0) < limit:
                return "generate_code"
            return END

        g = StateGraph(MemoryGraphState)
        g.add_node("load_project_context", nodes["load_project_context"])
        g.add_node("generate_code", generate_code)
        g.add_node("test_code", test_code)
        g.add_node("record_failure", record_failure)
        g.add_node("store_execution_memory", store_execution_memory)
        g.add_node("approve_deliverable", approve_deliverable)
        g.add_edge(START, "load_project_context")
        g.add_edge("load_project_context", "generate_code")
        g.add_edge("generate_code", "test_code")
        g.add_conditional_edges("test_code", after_test,
                                ["store_execution_memory", "record_failure"])
        g.add_conditional_edges("record_failure", after_failure, ["generate_code", END])
        g.add_edge("store_execution_memory", "approve_deliverable")
        g.add_edge("approve_deliverable", END)
        return g.compile(checkpointer=self.checkpointer)

    @property
    def config(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": self.thread_id}}

    def run(
        self,
        *,
        query: str = "calculator UI decisions, failures, and deliverable status",
        role: str = ActorRole.SYSTEM.value,
        agent_id: str = "room_runner",
        task_id: str = "calc_deliverable",
    ) -> dict[str, Any]:
        payload: MemoryGraphState = {
            "project_id": self.project_id,
            "query": query,
            "role": role,
            "agent_id": agent_id,
            "task_id": task_id,
            "attempt": 0,
            "max_attempts": self.max_attempts,
            "generation_feedback": None,
            "deliverable_ok": False,
        }
        # Each attempt walks ~3 nodes; size LangGraph's step limit to the retry budget.
        config = {**self.config, "recursion_limit": 10 + 4 * self.max_attempts}
        out = self.graph.invoke(payload, config)
        # Async drain only — never block the room runner on another sync extract.
        try:
            self.mm.drain_retention_queue(limit=5)
        except Exception:
            pass
        return out

    def get_state(self) -> Any:
        return self.graph.get_state(self.config)

    def close(self) -> None:
        try:
            self.mm.close()
        except Exception:
            pass
        try:
            self._ckpt_conn.close()
        except Exception:
            pass


def build_generation_prompt(*, context_text: str, feedback: str | None) -> str:
    parts = [CALCULATOR_SPEC]
    if context_text.strip():
        parts.append(
            "Relevant project memory (decisions and past failures — respect them):\n"
            + context_text.strip()[:3000]
        )
    if feedback:
        parts.append(feedback)
    return "\n\n".join(parts)


_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S | re.I)
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def extract_python(reply: str) -> str | None:
    """First fenced Python block; else the whole reply if it parses as Python."""
    import ast

    text = _THINK.sub("", reply or "").strip()
    m = _FENCE.search(text)
    candidate = (m.group(1) if m else text).strip()
    if not candidate:
        return None
    try:
        ast.parse(candidate)
    except SyntaxError:
        return None
    return candidate + "\n"


# Runs in a child interpreter: the generated module is untrusted model output.
_VERIFY_SCRIPT = r"""
import importlib.util, json, sys
path = sys.argv[1]
def done(ok, detail):
    print(json.dumps({"ok": ok, "detail": detail})); sys.exit(0)
try:
    spec = importlib.util.spec_from_file_location("calculator_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
except Exception as e:
    done(False, f"import_error:{type(e).__name__}:{e}")
missing = [n for n in ("add", "sub", "mul", "div") if not callable(getattr(mod, n, None))]
if missing:
    done(False, f"missing_fns:{missing}")
cases = [("add", (2, 3), 5), ("add", (-1.5, 1.5), 0), ("sub", (5, 3), 2), ("sub", (0, 4), -4),
         ("mul", (4, 2.5), 10), ("mul", (-3, 3), -9), ("div", (8, 2), 4), ("div", (1, 4), 0.25)]
for name, args, want in cases:
    try:
        got = getattr(mod, name)(*args)
    except Exception as e:
        done(False, f"{name}{args} raised {type(e).__name__}")
    if abs(got - want) > 1e-9:
        done(False, f"{name}{args} returned {got!r}, expected {want!r}")
try:
    mod.div(1, 0)
    done(False, "div(1, 0) did not raise ZeroDivisionError")
except ZeroDivisionError:
    pass
except Exception as e:
    done(False, f"div(1, 0) raised {type(e).__name__}, expected ZeroDivisionError")
done(True, "local_verified")
"""


def verify_calculator_file(path: Path, *, timeout_s: float = _VERIFY_TIMEOUT_S) -> tuple[bool, str]:
    """Deterministic acceptance test in a separate, isolated (-I) interpreter.

    The parent never imports or exec()s generated code. This contains crashes,
    hangs and module-level side effects to a throwaway process with a timeout.
    It is not a security sandbox: the child can still touch the filesystem, so
    run untrusted generations in a container before giving them real access.
    """
    with tempfile.TemporaryDirectory(prefix="hivi_verify_") as tmp:
        script = Path(tmp) / "verify_calculator.py"
        script.write_text(_VERIFY_SCRIPT, encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k.upper() in {"PATH", "SYSTEMROOT", "TEMP", "TMP"}}
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script), str(Path(path).resolve())],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout_s:.0f}s"
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
    if not lines:
        err = (proc.stderr or "").strip().splitlines()
        return False, f"verifier_crashed:{err[-1] if err else proc.returncode}"
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError:
        return False, "verifier_bad_output"
    return bool(result.get("ok")), str(result.get("detail"))


def seed_calculator_memories(mm: MemoryManager, project_id: str = "001") -> None:
    """Ensure room has a decision + failure to recall (waits for Hindsight extraction)."""
    mm.record_event(
        project_id=project_id,
        event_type="architecture_decision",
        content="Calculator UI is web-based",
        role=ActorRole.PROJECT_OWNER,
        evidence_path="data/projects/calculator/ADR-ui.md",
        sync_retain=True,
        enqueue=False,
        wait=True,
    )
    mm.record_event(
        project_id=project_id,
        event_type="test_failure",
        content="Calculator failed: division by zero on empty input",
        role=ActorRole.WORKER,
        agent_id="tester_01",
        sync_retain=True,
        enqueue=False,
        wait=True,
    )
