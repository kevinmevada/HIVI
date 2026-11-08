"""C2 — Live ProjectRoom validation (requires :8080 + :8888).

Acceptance requires execution + persistence + recovery evidence, not only a green graph invoke.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import pytest

from hivi.memory.policies import ActorRole
from hivi.orchestrator.preflight import run_preflight
from hivi.orchestrator.project_room import (
    ProjectRoom,
    ProjectRoomPaths,
    seed_calculator_memories,
)

# Clean bank — avoids stuck consolidation on hivi_project_001 (see C1 notes).
C2_PROJECT_ID = "098"
EVIDENCE_DIR = Path("data/hivi/c2_evidence")


@pytest.fixture(scope="module")
def preflight():
    result = run_preflight(project_id=C2_PROJECT_ID)
    if not result.ok:
        pytest.skip("live stack unavailable: " + "; ".join(result.errors))
    return result


@pytest.fixture()
def room_paths(tmp_path: Path, preflight) -> ProjectRoomPaths:
    root = tmp_path / "hivi_root"
    project_dir = root / "data" / "projects" / "calculator"
    data_hivi = root / "data" / "hivi"
    project_dir.mkdir(parents=True)
    data_hivi.mkdir(parents=True)
    return ProjectRoomPaths(
        root=root,
        project_dir=project_dir,
        artifact_path=project_dir / "calculator.py",
        ledger_db=data_hivi / "hivi.sqlite",
        checkpoint_db=data_hivi / "langgraph_checkpoints.sqlite",
    )


def test_preflight_ok(preflight):
    assert preflight.hindsight_ok
    assert preflight.inference_models_ok
    assert preflight.inference_chat_ok
    assert preflight.bank_ok
    assert preflight.model_id


def test_project_room_e2e_lifecycle(room_paths: ProjectRoomPaths, preflight):
    thread_id = f"c2-room-{uuid.uuid4().hex[:8]}"
    timings: dict[str, float] = {}
    evidence: dict = {
        "project_id": C2_PROJECT_ID,
        "thread_id": thread_id,
        "model": preflight.model_id,
        "criteria": {},
        "timings_s": timings,
        "notes": [
            "C1 retain latency ~109s on Colibrì 8B/4k is a known limitation.",
            "Bank 098 used to avoid stuck consolidation on project 001.",
        ],
    }

    t0 = time.perf_counter()
    room = ProjectRoom(
        paths=room_paths,
        project_id=C2_PROJECT_ID,
        thread_id=thread_id,
        preflight=True,
    )
    timings["open_s"] = round(time.perf_counter() - t0, 2)
    try:
        # Reuse bank memories from prior C2 attempt when present (avoids 2× ~110s seeds).
        t_seed = time.perf_counter()
        pre_ctx = room.mm.build_context(
            project_id=C2_PROJECT_ID,
            query="UI requirement and division failures",
        )
        if not pre_ctx.items:
            seed_calculator_memories(room.mm, C2_PROJECT_ID)
            pre_ctx = room.mm.build_context(
                project_id=C2_PROJECT_ID,
                query="UI requirement and division failures",
            )
        timings["seed_retain_s"] = round(time.perf_counter() - t_seed, 2)
        evidence["criteria"]["seed_memories"] = bool(pre_ctx.items)
        pre_blob = pre_ctx.text.lower()
        evidence["criteria"]["pre_run_recall"] = bool(pre_ctx.items)
        evidence["pre_run_context_sample"] = pre_ctx.text[:500]
        assert pre_ctx.items, "seeded memories not recallable before ProjectRoom run"
        assert "web" in pre_blob or "calculator" in pre_blob or "division" in pre_blob

        t1 = time.perf_counter()
        out = room.run(
            query="current UI requirement, known failures, deliverable",
            role=ActorRole.SYSTEM.value,
        )
        timings["run_s"] = round(time.perf_counter() - t1, 2)

        # Execution: context loaded
        ctx = (out.get("context_text") or "")
        evidence["criteria"]["context_loaded"] = bool(ctx.strip())
        evidence["context_sample"] = ctx[:500]
        assert ctx.strip(), "empty project context"

        # Deliverable + evidence file on disk
        artifact = Path(out.get("artifact_path") or room_paths.artifact_path)
        evidence["artifact_path"] = str(artifact)
        evidence["criteria"]["artifact_exists"] = artifact.is_file()
        assert artifact.is_file(), "deliverable missing"
        src = artifact.read_text(encoding="utf-8")
        evidence["criteria"]["artifact_has_div"] = "def div" in src
        assert "def div" in src

        # Ledger event from store node
        evidence["event_id"] = out.get("event_id")
        evidence["criteria"]["ledger_event"] = bool(out.get("event_id"))
        assert out.get("event_id"), "deliverable not recorded in ledger"

        # Verified deliverable (LLM verify + wait retain)
        evidence["verify_detail"] = out.get("verify_detail")
        evidence["criteria"]["deliverable_verified"] = out.get("deliverable_ok") is True
        assert out.get("deliverable_ok") is True, out.get("verify_detail")

        verified = room.mm.recall(
            project_id=C2_PROJECT_ID, query="Verified calculator deliverable"
        )
        evidence["criteria"]["verified_memory_recall"] = bool(verified.results)
        evidence["verified_sample"] = [r.text[:160] for r in verified.results[:3]]
        assert verified.results, "verified deliverable not recallable"

        fail = room.mm.recall(project_id=C2_PROJECT_ID, query="division by zero")
        evidence["criteria"]["failure_recall"] = bool(fail.results)
        assert fail.results, "failure memory not recallable"

        state1 = room.get_state()
        evidence["criteria"]["checkpoint_after_run"] = (
            state1.values.get("deliverable_ok") is True
        )
        assert state1.values.get("deliverable_ok") is True

        # Ledger persistence before close
        events = room.mm.ledger.list_for_project(C2_PROJECT_ID)
        evidence["ledger_event_count"] = len(events)
        evidence["criteria"]["ledger_persisted"] = len(events) > 0
        assert events
    finally:
        room.close()

    # Recovery: new process handle, same SQLite paths + thread_id
    t2 = time.perf_counter()
    room2 = ProjectRoom(
        paths=room_paths,
        project_id=C2_PROJECT_ID,
        thread_id=thread_id,
        preflight=True,
    )
    timings["reopen_s"] = round(time.perf_counter() - t2, 2)
    try:
        state2 = room2.get_state()
        evidence["criteria"]["checkpoint_resume"] = (
            state2.values.get("deliverable_ok") is True
            and bool(state2.values.get("artifact_path"))
        )
        assert state2.values.get("deliverable_ok") is True
        assert state2.values.get("artifact_path")
        resumed_artifact = Path(state2.values["artifact_path"])
        evidence["criteria"]["artifact_survives_restart"] = resumed_artifact.is_file()
        assert resumed_artifact.is_file()

        ctx2 = room2.mm.build_context(
            project_id=C2_PROJECT_ID,
            query="UI requirement and verified deliverable",
        )
        blob = ctx2.text.lower()
        evidence["post_restart_context_sample"] = ctx2.text[:500]
        evidence["criteria"]["memory_after_restart"] = bool(ctx2.items) and (
            "web" in blob or "verified" in blob or "calculator" in blob
        )
        assert ctx2.items
        assert "web" in blob or "verified" in blob or "calculator" in blob

        events2 = room2.mm.ledger.list_for_project(C2_PROJECT_ID)
        evidence["criteria"]["ledger_after_restart"] = len(events2) > 0
        assert events2
    finally:
        room2.close()

    evidence["pass"] = all(bool(v) for v in evidence["criteria"].values())
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVIDENCE_DIR / f"{thread_id}.json"
    out_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    latest = EVIDENCE_DIR / "latest.json"
    latest.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print("C2_EVIDENCE", out_path)
    print("E2E_TIMINGS", timings)
    assert evidence["pass"], evidence["criteria"]
