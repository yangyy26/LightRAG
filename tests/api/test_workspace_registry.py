from pathlib import Path

import pytest

from lightrag.api.workspace import (
    WORKSPACE_HEADER,
    WorkspaceAlreadyExistsError,
    WorkspaceNotFoundError,
    WorkspaceRegistry,
    validate_workspace_id,
)


def test_validate_workspace_id_accepts_alnum_and_underscore():
    assert WORKSPACE_HEADER == "LIGHTRAG-WORKSPACE"
    assert validate_workspace_id("project_01") == "project_01"


@pytest.mark.parametrize("workspace_id", ["", " ", "project-a", "../x", "a/b", "中文"])
def test_validate_workspace_id_rejects_invalid_values(workspace_id):
    with pytest.raises(ValueError):
        validate_workspace_id(workspace_id)


def test_registry_create_list_get_and_reload(tmp_path: Path):
    registry = WorkspaceRegistry(tmp_path)

    created = registry.create("project_a", name="Project A")

    assert created.id == "project_a"
    assert created.name == "Project A"
    assert created.created_at
    assert created.updated_at
    assert (tmp_path / "workspaces.json").exists()

    reloaded = WorkspaceRegistry(tmp_path)
    assert [item.id for item in reloaded.list()] == ["project_a"]
    assert reloaded.get("project_a").name == "Project A"


def test_registry_create_duplicate_raises_conflict(tmp_path: Path):
    registry = WorkspaceRegistry(tmp_path)
    registry.create("project_a")

    with pytest.raises(WorkspaceAlreadyExistsError):
        registry.create("project_a")


def test_registry_get_and_delete_missing_raise_not_found(tmp_path: Path):
    registry = WorkspaceRegistry(tmp_path)

    with pytest.raises(WorkspaceNotFoundError):
        registry.get("missing")

    with pytest.raises(WorkspaceNotFoundError):
        registry.delete("missing")


def test_registry_delete_removes_record(tmp_path: Path):
    registry = WorkspaceRegistry(tmp_path)
    registry.create("project_a")

    deleted = registry.delete("project_a")

    assert deleted.id == "project_a"
    assert registry.list() == []
