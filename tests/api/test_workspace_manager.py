import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException

from lightrag.api.workspace import (
    WorkspaceBusyError,
    WorkspaceDefaultDeletionError,
    WorkspaceManager,
    WorkspaceRegistry,
)


class FakeStorage:
    def __init__(self):
        self.dropped = False

    async def drop(self):
        self.dropped = True
        return {"status": "success"}


class FakeRAG:
    def __init__(self, workspace: str):
        self.workspace = workspace
        self.initialized = False
        self.migrated = False
        self.finalized = False
        self.pipeline_status = {
            "busy": False,
            "scanning": False,
            "destructive_busy": False,
            "pending_enqueues": 0,
        }
        for attr in (
            "full_docs",
            "text_chunks",
            "llm_response_cache",
            "full_entities",
            "full_relations",
            "entity_chunks",
            "relation_chunks",
            "entities_vdb",
            "relationships_vdb",
            "chunks_vdb",
            "chunk_entity_relation_graph",
            "doc_status",
        ):
            setattr(self, attr, FakeStorage())

    async def initialize_storages(self):
        self.initialized = True

    async def check_and_migrate_data(self):
        self.migrated = True

    async def finalize_storages(self):
        self.finalized = True


class SlowFakeRAG(FakeRAG):
    async def initialize_storages(self):
        await asyncio.sleep(0.01)
        self.initialized = True


class FakeDocManager:
    def __init__(self, input_dir: str, workspace: str):
        self.input_dir = Path(input_dir) / workspace
        self.workspace = workspace
        self.input_dir.mkdir(parents=True, exist_ok=True)


def test_manager_lazily_creates_and_reuses_context(tmp_path: Path):
    async def run():
        registry = WorkspaceRegistry(tmp_path / "rag")
        registry.create("project_a")
        created_rags: list[FakeRAG] = []

        def rag_factory(workspace: str):
            rag = FakeRAG(workspace)
            created_rags.append(rag)
            return rag

        manager = WorkspaceManager(
            registry=registry,
            working_dir=tmp_path / "rag",
            input_dir=tmp_path / "inputs",
            rag_factory=rag_factory,
            document_manager_cls=FakeDocManager,
        )

        first = await manager.get_context("project_a")
        second = await manager.get_context("project_a")

        assert first is second
        assert first.workspace_id == "project_a"
        assert first.rag.initialized is True
        assert first.rag.migrated is True
        assert first.doc_manager.input_dir == tmp_path / "inputs" / "project_a"
        assert len(created_rags) == 1

    asyncio.run(run())


def test_manager_registers_default_workspace(tmp_path: Path):
    registry = WorkspaceRegistry(tmp_path / "rag")

    manager = WorkspaceManager(
        registry=registry,
        working_dir=tmp_path / "rag",
        input_dir=tmp_path / "inputs",
        rag_factory=FakeRAG,
        document_manager_cls=FakeDocManager,
        default_workspace="default",
    )

    assert manager.default_workspace == "default"
    assert [workspace.id for workspace in registry.list()] == ["default"]


def test_manager_uses_default_context_when_workspace_is_omitted(tmp_path: Path):
    async def run():
        manager = WorkspaceManager(
            registry=WorkspaceRegistry(tmp_path / "rag"),
            working_dir=tmp_path / "rag",
            input_dir=tmp_path / "inputs",
            rag_factory=FakeRAG,
            document_manager_cls=FakeDocManager,
            default_workspace="default",
        )

        context = await manager.require_context(None)

        assert context.workspace_id == "default"
        assert context.rag.workspace == "default"

    asyncio.run(run())


def test_manager_serializes_first_context_initialization(tmp_path: Path):
    async def run():
        registry = WorkspaceRegistry(tmp_path / "rag")
        registry.create("project_a")
        created_rags: list[SlowFakeRAG] = []

        def rag_factory(workspace: str):
            rag = SlowFakeRAG(workspace)
            created_rags.append(rag)
            return rag

        manager = WorkspaceManager(
            registry=registry,
            working_dir=tmp_path / "rag",
            input_dir=tmp_path / "inputs",
            rag_factory=rag_factory,
            document_manager_cls=FakeDocManager,
        )

        contexts = await asyncio.gather(
            manager.get_context("project_a"),
            manager.get_context("project_a"),
            manager.get_context("project_a"),
        )

        assert len({id(context) for context in contexts}) == 1
        assert len(created_rags) == 1

    asyncio.run(run())


def test_manager_rejects_unknown_workspace(tmp_path: Path):
    async def run():
        manager = WorkspaceManager(
            registry=WorkspaceRegistry(tmp_path / "rag"),
            working_dir=tmp_path / "rag",
            input_dir=tmp_path / "inputs",
            rag_factory=FakeRAG,
            document_manager_cls=FakeDocManager,
        )

        with pytest.raises(HTTPException) as exc:
            await manager.require_context("missing")

        assert exc.value.status_code == 404

    asyncio.run(run())


def test_manager_shutdown_finalizes_cached_contexts(tmp_path: Path):
    async def run():
        registry = WorkspaceRegistry(tmp_path / "rag")
        registry.create("project_a")
        manager = WorkspaceManager(
            registry=registry,
            working_dir=tmp_path / "rag",
            input_dir=tmp_path / "inputs",
            rag_factory=FakeRAG,
            document_manager_cls=FakeDocManager,
        )
        context = await manager.get_context("project_a")

        await manager.shutdown()

        assert context.rag.finalized is True

    asyncio.run(run())


def test_manager_delete_busy_workspace_raises(tmp_path: Path):
    async def run():
        registry = WorkspaceRegistry(tmp_path / "rag")
        registry.create("project_a")
        manager = WorkspaceManager(
            registry=registry,
            working_dir=tmp_path / "rag",
            input_dir=tmp_path / "inputs",
            rag_factory=FakeRAG,
            document_manager_cls=FakeDocManager,
        )
        context = await manager.get_context("project_a")
        context.rag.pipeline_status["busy"] = True

        with pytest.raises(WorkspaceBusyError):
            await manager.delete_workspace("project_a")

    asyncio.run(run())


def test_manager_delete_default_workspace_raises(tmp_path: Path):
    async def run():
        manager = WorkspaceManager(
            registry=WorkspaceRegistry(tmp_path / "rag"),
            working_dir=tmp_path / "rag",
            input_dir=tmp_path / "inputs",
            rag_factory=FakeRAG,
            document_manager_cls=FakeDocManager,
            default_workspace="default",
        )

        with pytest.raises(WorkspaceDefaultDeletionError):
            await manager.delete_workspace("default")

    asyncio.run(run())


def test_manager_delete_idle_workspace_removes_registry_and_dirs(tmp_path: Path):
    async def run():
        registry = WorkspaceRegistry(tmp_path / "rag")
        registry.create("project_a")
        manager = WorkspaceManager(
            registry=registry,
            working_dir=tmp_path / "rag",
            input_dir=tmp_path / "inputs",
            rag_factory=FakeRAG,
            document_manager_cls=FakeDocManager,
        )
        context = await manager.get_context("project_a")

        deleted = await manager.delete_workspace("project_a")

        assert deleted.id == "project_a"
        assert context.rag.finalized is True
        assert [workspace.id for workspace in registry.list()] == ["default"]
        assert not (tmp_path / "rag" / "project_a").exists()
        assert not (tmp_path / "inputs" / "project_a").exists()

    asyncio.run(run())
