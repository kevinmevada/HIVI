"""Read/write memory access rules (HIVI governance — not Hindsight ACL)."""
from __future__ import annotations

from enum import Enum

from .models import MemoryStatus, MemoryType


class ActorRole(str, Enum):
    PROJECT_OWNER = "project_owner"
    PROJECT_DIRECTOR = "project_director"
    FACT_CHECKER = "fact_checker"
    INDEPENDENT_REVIEWER = "independent_reviewer"
    WORKER = "worker"
    SYSTEM = "system"


# Roles that may apply authoritative corrections (supersede / invalidate)
_CORRECT_ROLES = {
    ActorRole.PROJECT_OWNER,
    ActorRole.PROJECT_DIRECTOR,
    ActorRole.FACT_CHECKER,
    ActorRole.SYSTEM,
}

_AUTHORITATIVE_WRITE = {
    ActorRole.PROJECT_OWNER,
    ActorRole.PROJECT_DIRECTOR,
    ActorRole.SYSTEM,
}


class PolicyDenied(PermissionError):
    pass


def can_retain(
    role: ActorRole | str,
    *,
    memory_type: MemoryType | None = None,
    status: MemoryStatus = MemoryStatus.UNVERIFIED,
) -> bool:
    r = ActorRole(role)
    if r == ActorRole.WORKER:
        # Workers may only submit unverified / supported assertions
        return status in {MemoryStatus.UNVERIFIED, MemoryStatus.SUPPORTED}
    if r in _AUTHORITATIVE_WRITE | {ActorRole.FACT_CHECKER, ActorRole.INDEPENDENT_REVIEWER}:
        return True
    return False


def can_correct(role: ActorRole | str) -> bool:
    return ActorRole(role) in _CORRECT_ROLES


def can_read_project(role: ActorRole | str, *, project_id: str, requested_project: str) -> bool:
    """Bank isolation: agents may only read their assigned project."""
    _ = role  # all roles scoped by project_id match
    return normalize_match(project_id) == normalize_match(requested_project)


def normalize_match(project_id: str) -> str:
    from .scopes import normalize_project_id

    return normalize_project_id(project_id)


def assert_can_retain(role: ActorRole | str, **kwargs) -> None:
    if not can_retain(role, **kwargs):
        raise PolicyDenied(f"role {role} cannot retain with {kwargs}")


def assert_can_correct(role: ActorRole | str) -> None:
    if not can_correct(role):
        raise PolicyDenied(f"role {role} cannot correct memories")


def assert_project_access(role: ActorRole | str, project_id: str, requested: str) -> None:
    if not can_read_project(role, project_id=project_id, requested_project=requested):
        raise PolicyDenied(
            f"role {role} denied cross-project access {project_id} -> {requested}"
        )
