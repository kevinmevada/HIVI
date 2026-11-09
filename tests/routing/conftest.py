from pathlib import Path
import sys

CHAT_DIR = Path(__file__).parents[2] / "interfaces" / "hivi-chat"
sys.path.insert(0, str(CHAT_DIR))
