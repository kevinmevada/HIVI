"""Map HIVI project identities to Hindsight bank IDs."""
from __future__ import annotations

import re


_PROJECT_RE = re.compile(r"^(?:project[_-]?)?(\d+)$", re.I)


def normalize_project_id(project_id: str) -> str:
    """Normalize to zero-padded digits, e.g. '1' / 'project_001' → '001'."""
    raw = (project_id or "").strip()
    m = _PROJECT_RE.match(raw)
    if m:
        return f"{int(m.group(1)):03d}"
    # Allow named projects (future): sanitize to slug
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()
    if not slug:
        raise ValueError("project_id is required")
    return slug


def bank_id_for_project(project_id: str) -> str:
    """One Hindsight bank per project: hivi_project_001."""
    pid = normalize_project_id(project_id)
    if pid.isdigit() or re.fullmatch(r"\d{3,}", pid):
        return f"hivi_project_{pid}"
    return f"hivi_project_{pid}"
