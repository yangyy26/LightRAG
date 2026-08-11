import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import lightrag.pipeline as pipeline_module
from lightrag.kg.shared_storage import (
    finalize_share_data,
    get_namespace_data,
    initialize_pipeline_status,
    initialize_share_data,
)
from lightrag.knowledge_hierarchy import HierarchyNode, NormalizedHierarchy
from lightrag.pipeline import _PipelineMixin
from lightrag.base import DocStatus


async def wait_until(predicate, timeout=0.5):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


class FakeDocStatus:
    def __init__(self):
        self.docs = {
            "doc-a": {
                "status": DocStatus.PROCESSED,
                "content_summary": "summary",
                "content_length": 10,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "file_path": "trees.pdf",
                "track_id": "track-a",
                "chunks_count": 1,
                "chunks_list": ["chunk-a"],
                "metadata": {},
            }
        }
        self.upserts = []

    async def get_by_id(self, doc_id):
        return self.docs.get(doc_id)

    async def upsert(self, data):
        self.upserts.append(data)
        for doc_id, payload in data.items():
            self.docs[doc_id] = payload


class FakeFullDocs:
    async def get_by_id(self, doc_id):
        return {"content": "Title: Verified Resource Title: Norms, Theory, and Values"}


def make_minimal_hierarchy():
    root = HierarchyNode(
        entity_id="resource:doc-a",
        entity_name="trees.pdf",
        entity_type="Resource",
        description="trees.pdf",
        source_id="",
        file_path="trees.pdf",
        hierarchy_kind="root",
        root_id="resource:doc-a",
        parent_id=None,
        level=0,
    )
    child = HierarchyNode(
        entity_id="resource:doc-a:kp:tree",
        entity_name="Tree",
        entity_type="KnowledgePoint",
        description="A tree concept.",
        source_id="chunk-a",
        file_path="trees.pdf",
        hierarchy_kind="knowledge_point",
        root_id="resource:doc-a",
        parent_id="resource:doc-a",
        level=1,
    )
    return NormalizedHierarchy(
        root=root,
        nodes=[child],
        edges=[
            (
                "resource:doc-a",
                "resource:doc-a:kp:tree",
                {
                    "edge_type": "hierarchy",
                    "root_id": "resource:doc-a",
                    "parent_id": "resource:doc-a",
                },
            )
        ],
    )


@pytest.mark.asyncio
async def test_maybe_build_knowledge_hierarchy_calls_builder_and_persists(monkeypatch):
    calls = []

    async def fake_build_resource_knowledge_hierarchy(**kwargs):
        calls.append(("build", kwargs["doc_id"]))
        return make_minimal_hierarchy()

    async def fake_persist_resource_knowledge_hierarchy(hierarchy, graph, entity_vdb):
        calls.append(("persist", hierarchy.root.entity_id))
        return True

    async def fake_cleanup_legacy_hierarchy_duplicates(graph, entity_vdb):
        calls.append(("cleanup", "legacy"))

    monkeypatch.setattr(
        pipeline_module,
        "build_resource_knowledge_hierarchy",
        fake_build_resource_knowledge_hierarchy,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_module,
        "persist_resource_knowledge_hierarchy",
        fake_persist_resource_knowledge_hierarchy,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_module,
        "cleanup_legacy_hierarchy_duplicates",
        fake_cleanup_legacy_hierarchy_duplicates,
        raising=False,
    )

    class Harness(_PipelineMixin):
        enable_knowledge_hierarchy = True
        chunk_entity_relation_graph = AsyncMock()
        entities_vdb = AsyncMock()

        def __init__(self):
            self.doc_status = FakeDocStatus()
            self.full_docs = FakeFullDocs()

        def _build_global_config(self):
            return {"enable_knowledge_hierarchy": True}

        async def _insert_done(self):
            pass

    await Harness()._maybe_build_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="trees.pdf",
        chunk_results=[({}, {})],
    )

    assert calls == [
        ("build", "doc-a"),
        ("cleanup", "legacy"),
        ("persist", "resource:doc-a"),
    ]


@pytest.mark.asyncio
async def test_maybe_build_knowledge_hierarchy_passes_grounding_text(monkeypatch):
    captured = {}

    async def fake_build_resource_knowledge_hierarchy(**kwargs):
        captured["grounding_text"] = kwargs["grounding_text"]
        return make_minimal_hierarchy()

    async def fake_persist_resource_knowledge_hierarchy(hierarchy, graph, entity_vdb):
        return True

    monkeypatch.setattr(
        pipeline_module,
        "build_resource_knowledge_hierarchy",
        fake_build_resource_knowledge_hierarchy,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_module,
        "persist_resource_knowledge_hierarchy",
        fake_persist_resource_knowledge_hierarchy,
        raising=False,
    )

    class Harness(_PipelineMixin):
        enable_knowledge_hierarchy = True
        chunk_entity_relation_graph = AsyncMock()
        entities_vdb = AsyncMock()

        def __init__(self):
            self.doc_status = FakeDocStatus()
            self.full_docs = FakeFullDocs()

        def _build_global_config(self):
            return {"enable_knowledge_hierarchy": True}

        async def _insert_done(self):
            pass

    await Harness()._maybe_build_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="trees.pdf",
        chunk_results=[({}, {})],
    )

    assert "Verified Resource Title" in captured["grounding_text"]


@pytest.mark.asyncio
async def test_schedule_knowledge_hierarchy_returns_before_builder_finishes(monkeypatch):
    started = asyncio.Event()
    finish = asyncio.Event()

    async def slow_build_resource_knowledge_hierarchy(**kwargs):
        started.set()
        await finish.wait()
        return make_minimal_hierarchy()

    async def fake_persist_resource_knowledge_hierarchy(hierarchy, graph, entity_vdb):
        return True

    monkeypatch.setattr(
        pipeline_module,
        "build_resource_knowledge_hierarchy",
        slow_build_resource_knowledge_hierarchy,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_module,
        "persist_resource_knowledge_hierarchy",
        fake_persist_resource_knowledge_hierarchy,
        raising=False,
    )

    class Harness(_PipelineMixin):
        enable_knowledge_hierarchy = True
        chunk_entity_relation_graph = AsyncMock()
        entities_vdb = AsyncMock()

        def __init__(self):
            self.doc_status = FakeDocStatus()
            self.flush_count = 0

        def _build_global_config(self):
            return {"enable_knowledge_hierarchy": True}

        async def _insert_done(self):
            self.flush_count += 1

    harness = Harness()
    task = await harness._schedule_knowledge_hierarchy_build(
        doc_id="doc-a",
        file_path="trees.pdf",
        chunk_results=[({}, {})],
    )

    assert task is not None
    await asyncio.wait_for(started.wait(), timeout=0.2)
    assert harness.doc_status.docs["doc-a"]["metadata"]["hierarchy_status"] == "processing"
    assert harness.flush_count == 0

    finish.set()
    await asyncio.wait_for(task, timeout=0.2)
    assert harness.doc_status.docs["doc-a"]["metadata"]["hierarchy_status"] == "success"
    assert harness.flush_count == 1


@pytest.mark.asyncio
async def test_maybe_build_knowledge_hierarchy_flushes_after_persist(monkeypatch):
    async def fake_build_resource_knowledge_hierarchy(**kwargs):
        root = HierarchyNode(
            entity_id="resource:doc-a",
            entity_name="trees.pdf",
            entity_type="Resource",
            description="trees.pdf",
            source_id="",
            file_path="trees.pdf",
            hierarchy_kind="root",
            root_id="resource:doc-a",
            parent_id=None,
            level=0,
        )
        return NormalizedHierarchy(
            root=root,
            nodes=[
                HierarchyNode(
                    entity_id="Photosynthesis",
                    entity_name="Photosynthesis",
                    entity_type="KnowledgePoint",
                    description="Photosynthesis.",
                    source_id="chunk-a",
                    file_path="trees.pdf",
                    hierarchy_kind="knowledge_point",
                    root_id="resource:doc-a",
                    parent_id="resource:doc-a",
                    level=1,
                ),
                HierarchyNode(
                    entity_id="Chlorophyll",
                    entity_name="Chlorophyll",
                    entity_type="KnowledgePoint",
                    description="Chlorophyll.",
                    source_id="chunk-a",
                    file_path="trees.pdf",
                    hierarchy_kind="knowledge_point",
                    root_id="resource:doc-a",
                    parent_id="Photosynthesis",
                    level=2,
                ),
                HierarchyNode(
                    entity_id="Light Reaction",
                    entity_name="Light Reaction",
                    entity_type="KnowledgePoint",
                    description="Light reaction.",
                    source_id="chunk-a",
                    file_path="trees.pdf",
                    hierarchy_kind="knowledge_point",
                    root_id="resource:doc-a",
                    parent_id="Photosynthesis",
                    level=2,
                ),
            ],
            edges=[
                (
                    "resource:doc-a",
                    "Photosynthesis",
                    {
                        "edge_type": "hierarchy",
                        "root_id": "resource:doc-a",
                        "parent_id": "resource:doc-a",
                    },
                ),
                (
                    "Photosynthesis",
                    "Chlorophyll",
                    {
                        "edge_type": "hierarchy",
                        "root_id": "resource:doc-a",
                        "parent_id": "Photosynthesis",
                    },
                ),
                (
                    "Photosynthesis",
                    "Light Reaction",
                    {
                        "edge_type": "hierarchy",
                        "root_id": "resource:doc-a",
                        "parent_id": "Photosynthesis",
                    },
                ),
            ],
        )

    monkeypatch.setattr(
        pipeline_module,
        "build_resource_knowledge_hierarchy",
        fake_build_resource_knowledge_hierarchy,
        raising=False,
    )

    class Harness(_PipelineMixin):
        enable_knowledge_hierarchy = True
        chunk_entity_relation_graph = AsyncMock()
        entities_vdb = AsyncMock()

        def __init__(self):
            self.flush_count = 0
            self.doc_status = FakeDocStatus()

        def _build_global_config(self):
            return {"enable_knowledge_hierarchy": True}

        async def _insert_done(self):
            self.flush_count += 1

    harness = Harness()
    harness.chunk_entity_relation_graph.get_nodes_batch.return_value = {
        "Photosynthesis": {},
        "Chlorophyll": {},
        "Light Reaction": {},
    }
    harness.chunk_entity_relation_graph.get_all_edges.return_value = []
    await harness._maybe_build_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="trees.pdf",
        chunk_results=[({}, {})],
    )

    assert harness.flush_count == 1
    metadata = harness.doc_status.docs["doc-a"]["metadata"]
    assert metadata["hierarchy_edge_count"] == 3
    assert metadata["hierarchy_root_edge_count"] == 1
    assert metadata["hierarchy_non_root_edge_count"] == 2


@pytest.mark.asyncio
async def test_maybe_build_knowledge_hierarchy_failure_is_soft(monkeypatch):
    async def fake_build_resource_knowledge_hierarchy(**kwargs):
        raise RuntimeError("hierarchy provider down")

    monkeypatch.setattr(
        pipeline_module,
        "build_resource_knowledge_hierarchy",
        fake_build_resource_knowledge_hierarchy,
        raising=False,
    )

    class Harness(_PipelineMixin):
        enable_knowledge_hierarchy = True
        chunk_entity_relation_graph = AsyncMock()
        entities_vdb = AsyncMock()

        def __init__(self):
            self.doc_status = FakeDocStatus()

        def _build_global_config(self):
            return {"enable_knowledge_hierarchy": True}

        async def _insert_done(self):
            pass

    harness = Harness()
    await harness._maybe_build_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="trees.pdf",
        chunk_results=[({}, {})],
    )
    assert harness.doc_status.docs["doc-a"]["metadata"]["hierarchy_status"] == "failed"
    assert "hierarchy provider down" in harness.doc_status.docs["doc-a"]["metadata"]["hierarchy_error"]


@pytest.mark.asyncio
async def test_maybe_build_knowledge_hierarchy_timeout_is_soft(monkeypatch):
    async def slow_build_resource_knowledge_hierarchy(**kwargs):
        await asyncio.sleep(0.05)
        return make_minimal_hierarchy()

    persist = AsyncMock()
    monkeypatch.setattr(
        pipeline_module,
        "build_resource_knowledge_hierarchy",
        slow_build_resource_knowledge_hierarchy,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_module,
        "persist_resource_knowledge_hierarchy",
        persist,
        raising=False,
    )

    class Harness(_PipelineMixin):
        enable_knowledge_hierarchy = True
        chunk_entity_relation_graph = AsyncMock()
        entities_vdb = AsyncMock()

        def __init__(self):
            self.doc_status = FakeDocStatus()

        def _build_global_config(self):
            return {
                "enable_knowledge_hierarchy": True,
            }

        async def _insert_done(self):
            pass

    harness = Harness()
    await harness._maybe_build_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="trees.pdf",
        chunk_results=[({}, {})],
    )

    persist.assert_awaited_once()
    assert harness.doc_status.docs["doc-a"]["metadata"]["hierarchy_status"] == "success"


@pytest.mark.asyncio
async def test_hierarchy_build_updates_pipeline_status_without_main_busy(monkeypatch):
    finalize_share_data()
    initialize_share_data()
    await initialize_pipeline_status(workspace="course-a")

    async def fake_build_resource_knowledge_hierarchy(**kwargs):
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace="course-a"
        )
        assert pipeline_status["busy"] is False
        assert pipeline_status["hierarchy_busy"] is True
        assert pipeline_status["hierarchy_current_doc"] == "doc-a"
        return make_minimal_hierarchy()

    async def fake_persist_resource_knowledge_hierarchy(hierarchy, graph, entity_vdb):
        return True

    monkeypatch.setattr(
        pipeline_module,
        "build_resource_knowledge_hierarchy",
        fake_build_resource_knowledge_hierarchy,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_module,
        "persist_resource_knowledge_hierarchy",
        fake_persist_resource_knowledge_hierarchy,
        raising=False,
    )

    class Harness(_PipelineMixin):
        enable_knowledge_hierarchy = True
        workspace = "course-a"
        chunk_entity_relation_graph = AsyncMock()
        entities_vdb = AsyncMock()

        def __init__(self):
            self.doc_status = FakeDocStatus()

        def _build_global_config(self):
            return {"enable_knowledge_hierarchy": True}

        async def _insert_done(self):
            pass

    try:
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace="course-a"
        )
        pipeline_status["busy"] = False

        await Harness()._maybe_build_knowledge_hierarchy(
            doc_id="doc-a",
            file_path="trees.pdf",
            chunk_results=[({}, {})],
        )

        assert pipeline_status["busy"] is False
        assert pipeline_status["hierarchy_busy"] is False
        assert pipeline_status["hierarchy_done"] == 1
        assert pipeline_status["hierarchy_failed"] == 0
        assert "Knowledge hierarchy completed" in pipeline_status["hierarchy_latest_message"]
    finally:
        finalize_share_data()


@pytest.mark.asyncio
async def test_maybe_build_knowledge_hierarchy_augments_candidates_from_final_graph(monkeypatch):
    captured = {}

    async def fake_build_resource_knowledge_hierarchy(**kwargs):
        captured["chunk_results"] = kwargs["chunk_results"]
        return make_minimal_hierarchy()

    async def fake_persist_resource_knowledge_hierarchy(hierarchy, graph, entity_vdb):
        return True

    monkeypatch.setattr(
        pipeline_module,
        "build_resource_knowledge_hierarchy",
        fake_build_resource_knowledge_hierarchy,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_module,
        "persist_resource_knowledge_hierarchy",
        fake_persist_resource_knowledge_hierarchy,
        raising=False,
    )

    graph = AsyncMock()
    graph.get_all_nodes.return_value = [
        {
            "entity_id": "Extra Entity",
            "entity_name": "Extra Entity",
            "entity_type": "UNKNOWN",
            "description": "Created from a relation endpoint.",
            "source_id": "chunk-a",
            "file_path": "trees.pdf",
        }
    ]
    graph.get_all_edges.return_value = []

    class Harness(_PipelineMixin):
        enable_knowledge_hierarchy = True
        workspace = "course-a"
        entities_vdb = AsyncMock()

        def __init__(self):
            self.doc_status = FakeDocStatus()
            self.chunk_entity_relation_graph = graph

        def _build_global_config(self):
            return {"enable_knowledge_hierarchy": True}

        async def _insert_done(self):
            pass

    await Harness()._maybe_build_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="trees.pdf",
        chunk_results=[
            (
                {
                    "Original Entity": [
                        {
                            "entity_name": "Original Entity",
                            "entity_type": "concept",
                            "description": "Extracted from chunk.",
                            "source_id": "chunk-a",
                            "file_path": "trees.pdf",
                        }
                    ]
                },
                {},
            )
        ],
    )

    merged_nodes = {}
    for nodes, _edges in captured["chunk_results"]:
        merged_nodes.update(nodes)
    assert "Original Entity" in merged_nodes
    assert "Extra Entity" in merged_nodes


@pytest.mark.asyncio
async def test_maybe_build_knowledge_hierarchy_records_skip_reason(monkeypatch):
    async def fake_build_resource_knowledge_hierarchy(**kwargs):
        return None

    persist = AsyncMock()
    monkeypatch.setattr(
        pipeline_module,
        "build_resource_knowledge_hierarchy",
        fake_build_resource_knowledge_hierarchy,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_module,
        "persist_resource_knowledge_hierarchy",
        persist,
        raising=False,
    )

    class Harness(_PipelineMixin):
        enable_knowledge_hierarchy = True
        chunk_entity_relation_graph = AsyncMock()
        entities_vdb = AsyncMock()

        def __init__(self):
            self.doc_status = FakeDocStatus()

        def _build_global_config(self):
            return {"enable_knowledge_hierarchy": True}

        async def _insert_done(self):
            pass

    harness = Harness()
    await harness._maybe_build_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="trees.pdf",
        chunk_results=[({}, {})],
    )

    persist.assert_not_awaited()
    metadata = harness.doc_status.docs["doc-a"]["metadata"]
    assert metadata["hierarchy_status"] == "failed"
    assert "No hierarchy was generated" in metadata["hierarchy_error"]


@pytest.mark.asyncio
async def test_maybe_build_knowledge_hierarchy_records_persist_failure(monkeypatch):
    async def fake_build_resource_knowledge_hierarchy(**kwargs):
        return SimpleNamespace(
            root=SimpleNamespace(entity_id="resource:doc-a"),
            nodes=[SimpleNamespace(entity_id="resource:doc-a:kp:tree")],
            edges=[
                (
                    "resource:doc-a:kp:tree",
                    "resource:doc-a",
                    {},
                )
            ],
        )

    async def fake_persist_resource_knowledge_hierarchy(hierarchy, graph, entity_vdb):
        return False

    monkeypatch.setattr(
        pipeline_module,
        "build_resource_knowledge_hierarchy",
        fake_build_resource_knowledge_hierarchy,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_module,
        "persist_resource_knowledge_hierarchy",
        fake_persist_resource_knowledge_hierarchy,
        raising=False,
    )

    class Harness(_PipelineMixin):
        enable_knowledge_hierarchy = True
        chunk_entity_relation_graph = AsyncMock()
        entities_vdb = AsyncMock()

        def __init__(self):
            self.doc_status = FakeDocStatus()

        def _build_global_config(self):
            return {"enable_knowledge_hierarchy": True}

        async def _insert_done(self):
            pass

    harness = Harness()
    await harness._maybe_build_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="trees.pdf",
        chunk_results=[({}, {})],
    )

    metadata = harness.doc_status.docs["doc-a"]["metadata"]
    assert metadata["hierarchy_status"] == "failed"
    assert "generated edges" in metadata["hierarchy_error"]


@pytest.mark.asyncio
async def test_maybe_build_knowledge_hierarchy_reports_empty_classification(monkeypatch):
    async def fake_build_resource_knowledge_hierarchy(**kwargs):
        return SimpleNamespace(
            root=SimpleNamespace(entity_id="resource:doc-a"), nodes=[], edges=[]
        )

    persist = AsyncMock()
    monkeypatch.setattr(
        pipeline_module,
        "build_resource_knowledge_hierarchy",
        fake_build_resource_knowledge_hierarchy,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_module,
        "persist_resource_knowledge_hierarchy",
        persist,
        raising=False,
    )

    class Harness(_PipelineMixin):
        enable_knowledge_hierarchy = True
        chunk_entity_relation_graph = AsyncMock()
        entities_vdb = AsyncMock()

        def __init__(self):
            self.doc_status = FakeDocStatus()

        def _build_global_config(self):
            return {"enable_knowledge_hierarchy": True}

        async def _insert_done(self):
            pass

    harness = Harness()
    await harness._maybe_build_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="trees.pdf",
        chunk_results=[({}, {})],
    )

    persist.assert_not_awaited()
    metadata = harness.doc_status.docs["doc-a"]["metadata"]
    assert metadata["hierarchy_status"] == "failed"
    assert "No eligible knowledge points remained" in metadata["hierarchy_error"]


@pytest.mark.asyncio
async def test_maybe_build_knowledge_hierarchy_skips_when_disabled(monkeypatch):
    build = AsyncMock()
    monkeypatch.setattr(
        pipeline_module,
        "build_resource_knowledge_hierarchy",
        build,
        raising=False,
    )

    class Harness(_PipelineMixin):
        enable_knowledge_hierarchy = False
        chunk_entity_relation_graph = AsyncMock()
        entities_vdb = AsyncMock()

        def _build_global_config(self):
            return {"enable_knowledge_hierarchy": False}

    await Harness()._maybe_build_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="trees.pdf",
        chunk_results=[({}, {})],
    )

    build.assert_not_awaited()
