from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.api.routers import graph_routes
from lightrag.knowledge_hierarchy import (
    get_all_hierarchy_trees,
    get_hierarchy_tree,
    get_knowledge_point_subtree,
)
from lightrag.lightrag import LightRAG
from lightrag.types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode


@pytest.mark.asyncio
async def test_get_hierarchy_tree_returns_nested_children():
    graph = AsyncMock()
    graph.get_node.side_effect = lambda node_id: {
        "resource:doc-a": {
            "entity_id": "resource:doc-a",
            "entity_name": "trees.pdf",
            "root_id": "resource:doc-a",
            "hierarchy_kind": "root",
            "level": 0,
        },
        "Tree": {
            "entity_id": "Tree",
            "entity_name": "Tree",
            "source_id": "chunk-a",
            "file_path": "trees.pdf",
        },
    }.get(node_id)
    graph.get_node_edges.side_effect = lambda node_id: {
        "resource:doc-a": [("resource:doc-a", "Tree")],
        "Tree": [],
    }.get(node_id, [])
    graph.get_edge.side_effect = lambda src, tgt: {
        ("resource:doc-a", "Tree"): {
            "edge_type": "hierarchy",
            "root_id": "resource:doc-a",
            "parent_id": "resource:doc-a",
            "child_id": "Tree",
            "source_id": "chunk-a",
            "file_path": "trees.pdf",
        }
    }.get((src, tgt))

    tree = await get_hierarchy_tree(graph, "resource:doc-a")

    assert tree["entity_id"] == "resource:doc-a"
    assert tree["children"][0]["entity_id"] == "Tree"
    assert tree["children"][0]["entity_name"] == "Tree"
    assert tree["children"][0]["source_id"] == "chunk-a"


@pytest.mark.asyncio
async def test_get_knowledge_point_subtree_returns_only_the_requested_branch():
    graph = AsyncMock()
    graph.get_node.side_effect = lambda node_id: {
        "resource:doc-a:kp:principle": {
            "entity_id": "resource:doc-a:kp:principle",
            "entity_name": "行刑原则",
            "root_id": "resource:doc-a",
            "node_type": "knowledge_point",
        },
        "resource:doc-a:kp:efficiency": {
            "entity_id": "resource:doc-a:kp:efficiency",
            "entity_name": "行刑效率原则",
            "root_id": "resource:doc-a",
            "node_type": "knowledge_point",
        },
        "resource:doc-a:kp:safety": {
            "entity_id": "resource:doc-a:kp:safety",
            "entity_name": "行刑安全原则",
            "root_id": "resource:doc-a",
            "node_type": "knowledge_point",
        },
    }.get(node_id)
    graph.get_all_edges.return_value = [
        {
            "edge_type": "hierarchy",
            "relation_type": "part_of",
            "root_id": "resource:doc-a",
            "parent_id": "resource:doc-a:kp:principle",
            "child_id": "resource:doc-a:kp:efficiency",
        },
        {
            "edge_type": "hierarchy",
            "relation_type": "part_of",
            "root_id": "resource:doc-a",
            "parent_id": "resource:doc-a:kp:principle",
            "child_id": "resource:doc-a:kp:safety",
        },
    ]

    tree = await get_knowledge_point_subtree(
        graph,
        "resource:doc-a:kp:principle",
    )

    assert tree["entity_name"] == "行刑原则"
    assert [item["entity_name"] for item in tree["children"]] == [
        "行刑安全原则",
        "行刑效率原则",
    ]


@pytest.mark.asyncio
async def test_get_all_hierarchy_trees_returns_roots_with_children_only():
    graph = AsyncMock()
    graph.get_all_nodes.return_value = [
        {
            "id": "resource:doc-b",
            "entity_id": "resource:doc-b",
            "entity_name": "b.pdf",
            "root_id": "resource:doc-b",
            "hierarchy_kind": "root",
        },
        {
            "id": "resource:doc-a",
            "entity_id": "resource:doc-a",
            "entity_name": "a.pdf",
            "root_id": "resource:doc-a",
            "hierarchy_kind": "root",
        },
        {
            "id": "resource:doc-empty",
            "entity_id": "resource:doc-empty",
            "entity_name": "empty.pdf",
            "root_id": "resource:doc-empty",
            "hierarchy_kind": "root",
        },
    ]
    graph.get_node.side_effect = lambda node_id: {
        "resource:doc-a": {
            "entity_id": "resource:doc-a",
            "entity_name": "a.pdf",
            "root_id": "resource:doc-a",
            "hierarchy_kind": "root",
        },
        "resource:doc-b": {
            "entity_id": "resource:doc-b",
            "entity_name": "b.pdf",
            "root_id": "resource:doc-b",
            "hierarchy_kind": "root",
        },
        "resource:doc-empty": {
            "entity_id": "resource:doc-empty",
            "entity_name": "empty.pdf",
            "root_id": "resource:doc-empty",
            "hierarchy_kind": "root",
        },
        "A": {"entity_id": "A", "entity_name": "A"},
        "B": {"entity_id": "B", "entity_name": "B"},
    }.get(node_id)
    graph.get_node_edges.side_effect = lambda node_id: {
        "resource:doc-a": [("resource:doc-a", "A")],
        "resource:doc-b": [("resource:doc-b", "B")],
        "resource:doc-empty": [],
        "A": [],
        "B": [],
    }.get(node_id, [])
    graph.get_edge.side_effect = lambda src, tgt: {
        ("resource:doc-a", "A"): {
            "edge_type": "hierarchy",
            "root_id": "resource:doc-a",
            "parent_id": "resource:doc-a",
            "child_id": "A",
        },
        ("resource:doc-b", "B"): {
            "edge_type": "hierarchy",
            "root_id": "resource:doc-b",
            "parent_id": "resource:doc-b",
            "child_id": "B",
        },
    }.get((src, tgt))

    trees = await get_all_hierarchy_trees(graph)

    assert [tree["entity_id"] for tree in trees] == ["resource:doc-a", "resource:doc-b"]


@pytest.mark.asyncio
async def test_get_hierarchy_tree_returns_deep_children_from_hierarchy_edges():
    graph = AsyncMock()
    graph.get_node.side_effect = lambda node_id: {
        "resource:doc-a": {
            "entity_id": "resource:doc-a",
            "entity_name": "trees.pdf",
            "root_id": "resource:doc-a",
            "hierarchy_kind": "root",
            "level": 0,
        },
        "Parent": {
            "entity_id": "Parent",
            "entity_name": "Parent",
            "source_id": "chunk-a",
            "file_path": "trees.pdf",
        },
        "Child": {
            "entity_id": "Child",
            "entity_name": "Child",
            "source_id": "chunk-b",
            "file_path": "trees.pdf",
        },
    }.get(node_id)
    graph.get_node_edges.side_effect = lambda node_id: {
        "resource:doc-a": [("resource:doc-a", "Parent")],
        "Parent": [("resource:doc-a", "Parent"), ("Parent", "Child")],
        "Child": [("Parent", "Child")],
    }.get(node_id, [])
    graph.get_edge.side_effect = lambda src, tgt: {
        ("resource:doc-a", "Parent"): {
            "edge_type": "hierarchy",
            "root_id": "resource:doc-a",
            "parent_id": "resource:doc-a",
            "child_id": "Parent",
        },
        ("Parent", "Child"): {
            "edge_type": "hierarchy",
            "root_id": "resource:doc-a",
            "parent_id": "Parent",
            "child_id": "Child",
        },
    }.get((src, tgt))

    tree = await get_hierarchy_tree(graph, "resource:doc-a")

    assert tree["children"][0]["entity_id"] == "Parent"
    assert tree["children"][0]["children"][0]["entity_id"] == "Child"


@pytest.mark.asyncio
async def test_get_hierarchy_tree_uses_edge_metadata_when_storage_edge_order_flips():
    graph = AsyncMock()
    graph.get_node.side_effect = lambda node_id: {
        "resource:doc-a": {
            "entity_id": "resource:doc-a",
            "entity_name": "trees.pdf",
            "root_id": "resource:doc-a",
            "hierarchy_kind": "root",
            "level": 0,
        },
        "Parent": {
            "entity_id": "Parent",
            "entity_name": "Parent",
            "source_id": "chunk-a",
            "file_path": "trees.pdf",
        },
        "Child": {
            "entity_id": "Child",
            "entity_name": "Child",
            "source_id": "chunk-b",
            "file_path": "trees.pdf",
        },
    }.get(node_id)
    graph.get_node_edges.side_effect = lambda node_id: {
        "resource:doc-a": [("resource:doc-a", "Parent")],
        "Parent": [("Parent", "resource:doc-a"), ("Child", "Parent")],
        "Child": [("Child", "Parent")],
    }.get(node_id, [])
    graph.get_edge.side_effect = lambda src, tgt: {
        ("resource:doc-a", "Parent"): {
            "edge_type": "hierarchy",
            "root_id": "resource:doc-a",
            "parent_id": "resource:doc-a",
            "child_id": "Parent",
        },
        ("Parent", "resource:doc-a"): {
            "edge_type": "hierarchy",
            "root_id": "resource:doc-a",
            "parent_id": "resource:doc-a",
            "child_id": "Parent",
        },
        ("Child", "Parent"): {
            "edge_type": "hierarchy",
            "root_id": "resource:doc-a",
            "parent_id": "Parent",
            "child_id": "Child",
        },
    }.get((src, tgt))

    tree = await get_hierarchy_tree(graph, "resource:doc-a")

    assert tree["children"][0]["entity_id"] == "Parent"
    assert tree["children"][0]["children"][0]["entity_id"] == "Child"


@pytest.mark.asyncio
async def test_get_hierarchy_tree_builds_from_all_root_edges_when_adjacency_is_flat():
    graph = AsyncMock()
    graph.get_node.side_effect = lambda node_id: {
        "resource:doc-a": {
            "entity_id": "resource:doc-a",
            "entity_name": "trees.pdf",
            "root_id": "resource:doc-a",
            "hierarchy_kind": "root",
            "level": 0,
        },
        "Parent": {
            "entity_id": "Parent",
            "entity_name": "Parent",
            "source_id": "chunk-a",
            "file_path": "trees.pdf",
        },
        "Child": {
            "entity_id": "Child",
            "entity_name": "Child",
            "source_id": "chunk-b",
            "file_path": "trees.pdf",
        },
    }.get(node_id)
    graph.get_node_edges.side_effect = lambda node_id: {
        "resource:doc-a": [("resource:doc-a", "Parent")],
        "Parent": [],
        "Child": [],
    }.get(node_id, [])
    graph.get_edge.side_effect = lambda src, tgt: {
        ("resource:doc-a", "Parent"): {
            "edge_type": "hierarchy",
            "root_id": "resource:doc-a",
            "parent_id": "resource:doc-a",
            "child_id": "Parent",
        },
    }.get((src, tgt))
    graph.get_all_edges.return_value = [
        {
            "source": "resource:doc-a",
            "target": "Parent",
            "edge_type": "hierarchy",
            "root_id": "resource:doc-a",
            "parent_id": "resource:doc-a",
            "child_id": "Parent",
        },
        {
            "source": "Parent",
            "target": "Child",
            "edge_type": "hierarchy",
            "root_id": "resource:doc-a",
            "parent_id": "Parent",
            "child_id": "Child",
        },
    ]

    tree = await get_hierarchy_tree(graph, "resource:doc-a")

    assert tree["children"][0]["entity_id"] == "Parent"
    assert tree["children"][0]["children"][0]["entity_id"] == "Child"


@pytest.mark.asyncio
async def test_get_hierarchy_tree_returns_none_for_orphan_root():
    graph = AsyncMock()
    graph.get_node.return_value = {
        "entity_id": "resource:doc-a",
        "entity_name": "trees.pdf",
        "root_id": "resource:doc-a",
        "hierarchy_kind": "root",
        "level": 0,
    }
    graph.get_node_edges.return_value = []

    tree = await get_hierarchy_tree(graph, "resource:doc-a")

    assert tree is None


@pytest.mark.asyncio
async def test_get_knowledge_graph_resolves_hierarchy_root_by_file_name():
    graph = AsyncMock()
    graph.get_node.side_effect = lambda node_id: {
        "resource:doc-a": {
            "entity_id": "resource:doc-a",
            "entity_name": "trees.pdf",
            "root_id": "resource:doc-a",
            "doc_id": "doc-a",
            "hierarchy_kind": "root",
        },
        "Tree": {
            "entity_id": "Tree",
            "entity_name": "Tree",
            "source_id": "chunk-a",
            "file_path": "trees.pdf",
        },
    }.get(node_id)
    graph.get_all_nodes.return_value = [
        {
            "id": "resource:doc-a",
            "entity_id": "resource:doc-a",
            "entity_name": "trees.pdf",
            "root_id": "resource:doc-a",
            "doc_id": "doc-a",
            "hierarchy_kind": "root",
        }
    ]
    graph.get_knowledge_graph.return_value = "hierarchy-subgraph"

    rag = object.__new__(LightRAG)
    rag.chunk_entity_relation_graph = graph
    rag.max_graph_nodes = 1000

    result = await rag.get_knowledge_graph("trees.pdf")

    assert result == "hierarchy-subgraph"
    graph.get_knowledge_graph.assert_awaited_once_with("resource:doc-a", 3, 1000)


@pytest.mark.asyncio
async def test_get_knowledge_graph_excludes_hierarchy_edges_from_main_graph():
    graph = AsyncMock()
    graph.get_node.return_value = None
    graph.get_all_nodes.return_value = []
    graph.get_knowledge_graph.return_value = KnowledgeGraph(
        nodes=[
            KnowledgeGraphNode(id="A", labels=["A"], properties={}),
            KnowledgeGraphNode(id="B", labels=["B"], properties={}),
            KnowledgeGraphNode(id="resource:doc-a", labels=["doc"], properties={}),
        ],
        edges=[
            KnowledgeGraphEdge(
                id="A-B",
                type="DIRECTED",
                source="A",
                target="B",
                properties={"keywords": "related"},
            ),
            KnowledgeGraphEdge(
                id="resource:doc-a-A",
                type="DIRECTED",
                source="resource:doc-a",
                target="A",
                properties={"edge_type": "hierarchy"},
            ),
        ],
    )

    rag = object.__new__(LightRAG)
    rag.chunk_entity_relation_graph = graph
    rag.max_graph_nodes = 1000

    result = await rag.get_knowledge_graph("*")

    assert [edge.id for edge in result.edges] == ["A-B"]


@pytest.mark.asyncio
async def test_get_knowledge_hierarchy_resolves_root_by_file_name():
    graph = AsyncMock()
    graph.get_node.side_effect = lambda node_id: {
        "resource:doc-a": {
            "entity_id": "resource:doc-a",
            "entity_name": "trees.pdf",
            "root_id": "resource:doc-a",
            "doc_id": "doc-a",
            "hierarchy_kind": "root",
        },
        "Tree": {
            "entity_id": "Tree",
            "entity_name": "Tree",
            "source_id": "chunk-a",
            "file_path": "trees.pdf",
        },
    }.get(node_id)
    graph.get_all_nodes.return_value = [
        {
            "id": "resource:doc-a",
            "entity_id": "resource:doc-a",
            "entity_name": "trees.pdf",
            "root_id": "resource:doc-a",
            "doc_id": "doc-a",
            "hierarchy_kind": "root",
        }
    ]
    graph.get_node_edges.side_effect = lambda node_id: {
        "resource:doc-a": [("resource:doc-a", "Tree")],
        "Tree": [],
    }.get(node_id, [])
    graph.get_edge.side_effect = lambda src, tgt: {
        ("resource:doc-a", "Tree"): {
            "edge_type": "hierarchy",
            "root_id": "resource:doc-a",
            "parent_id": "resource:doc-a",
            "child_id": "Tree",
        }
    }.get((src, tgt))

    rag = object.__new__(LightRAG)
    rag.chunk_entity_relation_graph = graph

    result = await rag.get_knowledge_hierarchy("trees.pdf")

    assert result["entity_id"] == "resource:doc-a"
    assert result["children"][0]["entity_id"] == "Tree"


@pytest.mark.asyncio
async def test_cleanup_legacy_knowledge_hierarchy_flushes_when_nodes_removed(
    monkeypatch,
):
    rag = object.__new__(LightRAG)
    rag.chunk_entity_relation_graph = AsyncMock()
    rag.entities_vdb = AsyncMock()
    rag._insert_done = AsyncMock()

    cleanup = AsyncMock(return_value=["resource:doc-a:kp:deadbeef"])
    monkeypatch.setattr(
        "lightrag.knowledge_hierarchy.cleanup_legacy_hierarchy_duplicates",
        cleanup,
    )

    result = await rag.cleanup_legacy_knowledge_hierarchy()

    assert result == {"removed": ["resource:doc-a:kp:deadbeef"], "count": 1}
    cleanup.assert_awaited_once_with(rag.chunk_entity_relation_graph, rag.entities_vdb)
    rag._insert_done.assert_awaited_once()


def test_cleanup_legacy_hierarchy_route_returns_removed_nodes(monkeypatch):
    rag = type(
        "FakeRAG",
        (),
        {
            "workspace": "",
            "cleanup_legacy_knowledge_hierarchy": AsyncMock(
                return_value={"removed": ["resource:doc-a:kp:deadbeef"], "count": 1}
            ),
        },
    )()

    async def allow_mutation(_rag):
        return None

    async def allow_auth():
        return None

    monkeypatch.setattr(
        graph_routes, "_get_router_auth_dependency", lambda _api_key: allow_auth
    )
    monkeypatch.setattr(graph_routes, "check_pipeline_busy_or_raise", allow_mutation)

    app = FastAPI()
    app.include_router(graph_routes.create_graph_routes(rag, api_key=None))
    client = TestClient(app)

    response = client.post("/graph/hierarchy/cleanup-legacy")

    assert response.status_code == 200, response.text
    assert response.json() == {
        "removed": ["resource:doc-a:kp:deadbeef"],
        "count": 1,
    }
    rag.cleanup_legacy_knowledge_hierarchy.assert_awaited_once()


def test_hierarchy_route_returns_404_for_orphan_root(monkeypatch):
    rag = type(
        "FakeRAG",
        (),
        {
            "workspace": "",
            "get_knowledge_hierarchy": AsyncMock(return_value=None),
        },
    )()

    async def allow_auth():
        return None

    monkeypatch.setattr(
        graph_routes, "_get_router_auth_dependency", lambda _api_key: allow_auth
    )

    app = FastAPI()
    app.include_router(graph_routes.create_graph_routes(rag, api_key=None))
    client = TestClient(app)

    response = client.get("/graph/hierarchy?root_id=resource:doc-a")

    assert response.status_code == 404


def test_hierarchies_route_returns_all_workspace_resource_trees(monkeypatch):
    trees = [
        {"entity_id": "resource:doc-a", "children": [{"entity_id": "A"}]},
        {"entity_id": "resource:doc-b", "children": [{"entity_id": "B"}]},
    ]
    rag = type(
        "FakeRAG",
        (),
        {
            "workspace": "course_a",
            "get_knowledge_hierarchies": AsyncMock(return_value=trees),
        },
    )()

    async def allow_auth():
        return None

    monkeypatch.setattr(
        graph_routes, "_get_router_auth_dependency", lambda _api_key: allow_auth
    )

    app = FastAPI()
    app.include_router(graph_routes.create_graph_routes(rag, api_key=None))
    client = TestClient(app)

    response = client.get("/graph/hierarchies?max_depth=8")

    assert response.status_code == 200, response.text
    assert response.json() == {"items": trees, "count": 2}
    rag.get_knowledge_hierarchies.assert_awaited_once_with(max_depth=8)


def test_knowledge_point_routes_return_subtree_and_associated_entities(monkeypatch):
    subtree = {
        "entity_id": "resource:doc-a:kp:principle",
        "entity_name": "行刑原则",
        "children": [],
    }
    associated_entities = [
        {"name": "贝卡利亚", "type": "person", "relation": "proposed_by"}
    ]
    rag = type(
        "FakeRAG",
        (),
        {
            "workspace": "course_a",
            "get_knowledge_point_subtree": AsyncMock(return_value=subtree),
            "get_knowledge_point_associated_entities": AsyncMock(
                return_value=associated_entities
            ),
        },
    )()

    async def allow_auth():
        return None

    monkeypatch.setattr(
        graph_routes, "_get_router_auth_dependency", lambda _api_key: allow_auth
    )

    app = FastAPI()
    app.include_router(graph_routes.create_graph_routes(rag, api_key=None))
    client = TestClient(app)
    knowledge_point_id = "resource:doc-a:kp:principle"

    subtree_response = client.get(
        f"/graph/knowledge-points/{knowledge_point_id}/subtree?max_depth=4"
    )
    entities_response = client.get(
        f"/graph/knowledge-points/{knowledge_point_id}/associated-entities"
    )

    assert subtree_response.status_code == 200, subtree_response.text
    assert subtree_response.json() == subtree
    assert entities_response.status_code == 200, entities_response.text
    assert entities_response.json() == {"items": associated_entities, "count": 1}
    rag.get_knowledge_point_subtree.assert_awaited_once_with(
        parent_id=knowledge_point_id,
        max_depth=4,
    )
    rag.get_knowledge_point_associated_entities.assert_awaited_once_with(
        knowledge_point_id
    )


def test_retry_hierarchy_route_schedules_document_hierarchy(monkeypatch):
    rag = type(
        "FakeRAG",
        (),
        {
            "workspace": "course_a",
            "schedule_knowledge_hierarchy_retry": AsyncMock(
                return_value={
                    "doc_id": "doc-a",
                    "root_id": "resource:doc-a",
                    "status": "processing",
                    "error": None,
                }
            ),
        },
    )()

    async def allow_auth():
        return None

    monkeypatch.setattr(
        graph_routes, "_get_router_auth_dependency", lambda _api_key: allow_auth
    )

    app = FastAPI()
    app.include_router(graph_routes.create_graph_routes(rag, api_key=None))
    client = TestClient(app)

    response = client.post("/graph/hierarchy/retry", json={"doc_id": "doc-a"})

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "processing"
    rag.schedule_knowledge_hierarchy_retry.assert_awaited_once_with("doc-a")
