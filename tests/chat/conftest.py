import sys
from pathlib import Path

import pytest

CHAT_DIR = Path(__file__).parents[2] / "interfaces" / "hivi-chat"
sys.path.insert(0, str(CHAT_DIR))


@pytest.fixture(autouse=True)
def _fresh_backend_gate(monkeypatch):
    """The gate's asyncio events bind to one event loop; each test gets its own."""
    import server

    monkeypatch.setattr(server, "GATE", server.BackendGate())
