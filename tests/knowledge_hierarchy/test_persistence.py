import json
from unittest.mock import AsyncMock

import networkx as nx
import pytest

from lightrag.knowledge_hierarchy import (
    HierarchyNode,
    NormalizedHierarchy,
    cleanup_legacy_hierarchy_duplicates,
    get_associated_entities,
    persist_resource_knowledge_hierarchy,
)


def test_knowledge_point_graph_data_is_graphml_compatible(tmp_path):
    node = HierarchyNode(
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
        aliases=["Tree", "Arborescence"],
        associated_entities=[
            {"name": "Ada", "type": "person", "relation": "proposed_by"}
        ],
        prerequisite_ids=["resource:doc-a:kp:forest"],
    )

    graph_data = node.to_graph_data()

    assert json.loads(graph_data["aliases"]) == ["Tree", "Arborescence"]
    assert json.loads(graph_data["prerequisite_ids"]) == [
        "resource:doc-a:kp:forest"
    ]
    assert json.loads(graph_data["associated_entities"]) == [
        {"name": "Ada", "type": "person", "relation": "proposed_by"}
    ]

    graph = nx.Graph()
    graph.add_node(node.entity_id, **graph_data)
    nx.write_graphml(graph, tmp_path / "hierarchy.graphml")


@pytest.mark.asyncio
async def test_persist_resource_knowledge_hierarchy_writes_graph_and_entity_vdb():
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
        entity_id="Binary Tree",
        entity_name="Binary Tree",
        entity_type="concept",
        description="A tree with at most two children.",
        source_id="chunk-a",
        file_path="trees.pdf",
        hierarchy_kind="knowledge_point",
        root_id="resource:doc-a",
        parent_id="resource:doc-a",
        level=1,
    )
    hierarchy = NormalizedHierarchy(
        root=root,
        nodes=[child],
        edges=[
            (
                "Binary Tree",
                "resource:doc-a",
                {
                    "description": "Knowledge point belongs to its parent in the resource knowledge hierarchy.",
                    "keywords": "part_of,hierarchy",
                    "weight": 1.0,
                    "source_id": "chunk-a",
                    "file_path": "trees.pdf",
                    "edge_type": "hierarchy",
                    "relation_type": "part_of",
                    "root_id": "resource:doc-a",
                    "parent_id": "resource:doc-a",
                    "child_id": "Binary Tree",
                },
            )
        ],
    )
    graph = AsyncMock()
    graph.get_nodes_batch.return_value = {"Binary Tree": {}}
    graph.get_all_edges.return_value = [
        {
            "source": "resource:doc-a",
            "target": "Old Child",
            "edge_type": "hierarchy",
            "root_id": "resource:doc-a",
        },
        {
            "source": "Other",
            "target": "Edge",
            "edge_type": "hierarchy",
            "root_id": "resource:doc-b",
        },
    ]
    graph.get_all_nodes.return_value = []
    entity_vdb = AsyncMock()

    await persist_resource_knowledge_hierarchy(hierarchy, graph, entity_vdb)

    graph.remove_edges.assert_awaited_once_with([("resource:doc-a", "Old Child")])
    graph.upsert_nodes_batch.assert_awaited_once()
    graph.upsert_edges_batch.assert_awaited_once()
    entity_vdb.upsert.assert_awaited_once()
    nodes = graph.upsert_nodes_batch.await_args.args[0]
    assert nodes[0][0] == "resource:doc-a"
    assert nodes[0][1]["doc_id"] == "doc-a"
    assert len(nodes) == 2
    edges = graph.upsert_edges_batch.await_args.args[0]
    assert edges[0][0] == "Binary Tree"
    assert edges[0][1] == "resource:doc-a"


@pytest.mark.asyncio
async def test_persist_resource_knowledge_hierarchy_skips_edges_with_missing_endpoints():
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
    hierarchy = NormalizedHierarchy(
        root=root,
        edges=[
            (
                "resource:doc-a",
                "Missing Entity",
                {
                    "edge_type": "hierarchy",
                    "relation_type": "contains",
                    "root_id": "resource:doc-a",
                    "parent_id": "resource:doc-a",
                    "child_id": "Missing Entity",
                    "source_id": "chunk-a",
                    "file_path": "trees.pdf",
                },
            )
        ],
    )
    graph = AsyncMock()
    graph.get_all_nodes.return_value = []
    graph.get_all_edges.return_value = []
    entity_vdb = AsyncMock()

    persisted = await persist_resource_knowledge_hierarchy(hierarchy, graph, entity_vdb)

    assert persisted is False
    graph.upsert_nodes_batch.assert_not_awaited()
    graph.upsert_edges_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_persist_resource_knowledge_hierarchy_clears_stale_edges_when_empty():
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
    hierarchy = NormalizedHierarchy(root=root, edges=[])
    graph = AsyncMock()
    graph.get_all_nodes.return_value = []
    graph.get_all_edges.return_value = [
        {
            "source": "resource:doc-a",
            "target": "Old Child",
            "edge_type": "hierarchy",
            "root_id": "resource:doc-a",
        },
        {
            "source": "resource:doc-b",
            "target": "Other Child",
            "edge_type": "hierarchy",
            "root_id": "resource:doc-b",
        },
    ]
    entity_vdb = AsyncMock()

    persisted = await persist_resource_knowledge_hierarchy(hierarchy, graph, entity_vdb)

    assert persisted is False
    graph.remove_edges.assert_awaited_once_with([("resource:doc-a", "Old Child")])
    graph.upsert_nodes_batch.assert_not_awaited()
    graph.upsert_edges_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_legacy_hierarchy_duplicates_removes_old_kp_nodes_and_vdb():
    graph = AsyncMock()
    graph.get_all_nodes.return_value = [
        {"id": "resource:doc-a", "hierarchy_kind": "root"},
        {
            "id": "resource:doc-a:kp:abc",
            "entity_id": "resource:doc-a:kp:abc",
            "hierarchy_kind": "knowledge_point",
        },
        {"id": "Binary Tree", "entity_id": "Binary Tree"},
    ]
    entity_vdb = AsyncMock()

    removed = await cleanup_legacy_hierarchy_duplicates(graph, entity_vdb)

    assert removed == ["resource:doc-a", "resource:doc-a:kp:abc"]
    assert [call.args[0] for call in graph.delete_node.await_args_list] == [
        "resource:doc-a",
        "resource:doc-a:kp:abc",
    ]
    entity_vdb.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_legacy_hierarchy_duplicates_removes_orphan_resource_roots():
    graph = AsyncMock()
    graph.get_all_nodes.return_value = [
        {
            "id": "resource:doc-a",
            "entity_id": "resource:doc-a",
            "hierarchy_kind": "root",
        },
        {
            "id": "resource:doc-b",
            "entity_id": "resource:doc-b",
            "hierarchy_kind": "root",
        },
    ]
    graph.get_all_edges.return_value = [
        {
            "source": "resource:doc-b",
            "target": "Tree",
            "root_id": "resource:doc-b",
            "edge_type": "hierarchy",
        }
    ]

    removed = await cleanup_legacy_hierarchy_duplicates(graph)

    assert removed == ["resource:doc-a"]
    graph.delete_node.assert_awaited_once_with("resource:doc-a")


@pytest.mark.asyncio
async def test_cleanup_legacy_hierarchy_duplicates_keeps_v2_knowledge_points():
    graph = AsyncMock()
    graph.get_all_nodes.return_value = [
        {
            "entity_id": "resource:doc-a:kp:abc",
            "hierarchy_kind": "knowledge_point",
            "node_type": "knowledge_point",
            "root_id": "resource:doc-a",
        }
    ]
    graph.get_all_edges.return_value = []

    removed = await cleanup_legacy_hierarchy_duplicates(graph)

    assert removed == []
    graph.delete_node.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_associated_entities_decodes_portable_node_property():
    graph = AsyncMock()
    graph.get_node.return_value = {
        "entity_id": "resource:doc-a:kp:law",
        "node_type": "knowledge_point",
        "associated_entities": '[{"name": "贝卡利亚", "type": "person", "relation": "proposed_by"}]',
    }

    items = await get_associated_entities(graph, "resource:doc-a:kp:law")

    assert items == [
        {"name": "贝卡利亚", "type": "person", "relation": "proposed_by"}
    ]
