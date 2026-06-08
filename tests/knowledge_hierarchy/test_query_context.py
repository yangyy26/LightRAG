from unittest.mock import AsyncMock

import pytest

from lightrag.base import QueryParam
from lightrag.knowledge_hierarchy import build_hierarchy_context
from lightrag.operate import _get_node_data


@pytest.mark.asyncio
async def test_build_hierarchy_context_stays_inside_root_id():
    graph = AsyncMock()
    graph.get_node.side_effect = lambda node_id: {
        "resource:doc-a:kp:tree": {
            "entity_id": "resource:doc-a:kp:tree",
            "entity_name": "Tree",
            "root_id": "resource:doc-a",
            "parent_id": "resource:doc-a",
            "source_id": "chunk-a",
            "file_path": "trees.pdf",
            "hierarchy_kind": "knowledge_point",
            "level": 1,
        },
        "resource:doc-a": {
            "entity_id": "resource:doc-a",
            "entity_name": "trees.pdf",
            "root_id": "resource:doc-a",
            "file_path": "trees.pdf",
            "hierarchy_kind": "root",
            "level": 0,
        },
        "resource:doc-b:kp:tree": {
            "entity_id": "resource:doc-b:kp:tree",
            "entity_name": "Tree",
            "root_id": "resource:doc-b",
            "parent_id": "resource:doc-b",
            "source_id": "chunk-b",
            "file_path": "other.pdf",
            "hierarchy_kind": "knowledge_point",
            "level": 1,
        },
    }.get(node_id)
    graph.get_node_edges.return_value = [
        ("resource:doc-a", "resource:doc-a:kp:tree"),
        ("resource:doc-b", "resource:doc-b:kp:tree"),
    ]

    context = await build_hierarchy_context(
        matched_entities=[
            {
                "entity_name": "resource:doc-a:kp:tree",
                "root_id": "resource:doc-a",
                "parent_id": "resource:doc-a",
                "hierarchy_kind": "knowledge_point",
            }
        ],
        knowledge_graph_inst=graph,
        query_param=QueryParam(enable_hierarchy_context=True),
        tokenizer=None,
    )

    assert "trees.pdf" in context
    assert "other.pdf" not in context
    assert "Tree" in context


@pytest.mark.asyncio
async def test_build_hierarchy_context_returns_empty_when_disabled():
    context = await build_hierarchy_context(
        matched_entities=[
            {
                "entity_name": "resource:doc-a:kp:tree",
                "root_id": "resource:doc-a",
                "hierarchy_kind": "knowledge_point",
            }
        ],
        knowledge_graph_inst=AsyncMock(),
        query_param=QueryParam(enable_hierarchy_context=False),
        tokenizer=None,
    )

    assert context == ""


@pytest.mark.asyncio
async def test_build_hierarchy_context_respects_child_depth():
    graph = AsyncMock()
    graph.get_node.side_effect = lambda node_id: {
        "resource:doc-a": {
            "entity_id": "resource:doc-a",
            "entity_name": "trees.pdf",
            "root_id": "resource:doc-a",
            "hierarchy_kind": "root",
            "level": 0,
        },
        "resource:doc-a:kp:root": {
            "entity_id": "resource:doc-a:kp:root",
            "entity_name": "Plant Biology",
            "root_id": "resource:doc-a",
            "parent_id": "resource:doc-a",
            "hierarchy_kind": "knowledge_point",
            "level": 1,
        },
        "resource:doc-a:kp:child": {
            "entity_id": "resource:doc-a:kp:child",
            "entity_name": "Photosynthesis",
            "root_id": "resource:doc-a",
            "parent_id": "resource:doc-a:kp:root",
            "hierarchy_kind": "knowledge_point",
            "level": 2,
        },
        "resource:doc-a:kp:grandchild": {
            "entity_id": "resource:doc-a:kp:grandchild",
            "entity_name": "Light Reaction",
            "root_id": "resource:doc-a",
            "parent_id": "resource:doc-a:kp:child",
            "hierarchy_kind": "knowledge_point",
            "level": 3,
        },
    }.get(node_id)

    graph.get_node_edges.side_effect = lambda node_id: {
        "resource:doc-a:kp:root": [
            ("resource:doc-a:kp:root", "resource:doc-a:kp:child")
        ],
        "resource:doc-a:kp:child": [
            ("resource:doc-a:kp:child", "resource:doc-a:kp:grandchild")
        ],
    }.get(node_id, [])

    shallow_context = await build_hierarchy_context(
        matched_entities=[
            {
                "entity_name": "resource:doc-a:kp:root",
                "root_id": "resource:doc-a",
                "hierarchy_kind": "knowledge_point",
            }
        ],
        knowledge_graph_inst=graph,
        query_param=QueryParam(
            enable_hierarchy_context=True,
            hierarchy_child_depth=1,
        ),
        tokenizer=None,
    )

    deep_context = await build_hierarchy_context(
        matched_entities=[
            {
                "entity_name": "resource:doc-a:kp:root",
                "root_id": "resource:doc-a",
                "hierarchy_kind": "knowledge_point",
            }
        ],
        knowledge_graph_inst=graph,
        query_param=QueryParam(
            enable_hierarchy_context=True,
            hierarchy_child_depth=2,
        ),
        tokenizer=None,
    )

    assert "Photosynthesis" in shallow_context
    assert "Light Reaction" not in shallow_context
    assert "Light Reaction" in deep_context


@pytest.mark.asyncio
async def test_build_hierarchy_context_resolves_original_kg_node_from_hierarchy_edges():
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
            "entity_type": "concept",
            "source_id": "chunk-a",
            "file_path": "trees.pdf",
        },
        "Binary Tree": {
            "entity_id": "Binary Tree",
            "entity_name": "Binary Tree",
            "entity_type": "concept",
            "source_id": "chunk-b",
            "file_path": "trees.pdf",
        },
    }.get(node_id)
    graph.get_node_edges.side_effect = lambda node_id: {
        "Tree": [
            ("resource:doc-a", "Tree"),
            ("Tree", "Binary Tree"),
        ],
        "resource:doc-a": [("resource:doc-a", "Tree")],
        "Binary Tree": [("Tree", "Binary Tree")],
    }.get(node_id, [])
    graph.get_edge.side_effect = lambda src, tgt: {
        ("resource:doc-a", "Tree"): {
            "edge_type": "hierarchy",
            "root_id": "resource:doc-a",
            "parent_id": "resource:doc-a",
            "child_id": "Tree",
            "file_path": "trees.pdf",
        },
        ("Tree", "Binary Tree"): {
            "edge_type": "hierarchy",
            "root_id": "resource:doc-a",
            "parent_id": "Tree",
            "child_id": "Binary Tree",
            "file_path": "trees.pdf",
        },
    }.get((src, tgt))

    context = await build_hierarchy_context(
        matched_entities=[{"entity_name": "Tree", "entity_type": "concept"}],
        knowledge_graph_inst=graph,
        query_param=QueryParam(enable_hierarchy_context=True),
        tokenizer=None,
    )

    assert "trees.pdf" in context
    assert "Tree" in context
    assert "Binary Tree" in context


@pytest.mark.asyncio
async def test_get_node_data_uses_original_entity_name_for_graph_lookup():
    entities_vdb = AsyncMock()
    entities_vdb.cosine_better_than_threshold = 0.2
    entities_vdb.query.return_value = [
        {
            "id": "ent-tree",
            "entity_name": "Tree",
            "entity_type": "concept",
        }
    ]

    graph = AsyncMock()
    graph.get_nodes_batch.return_value = {
        "Tree": {
            "entity_id": "Tree",
            "entity_name": "Tree",
            "source_id": "chunk-a",
            "file_path": "trees.pdf",
        }
    }
    graph.node_degrees_batch.return_value = {"Tree": 1}
    graph.get_nodes_edges_batch.return_value = {"Tree": []}

    nodes, relations = await _get_node_data(
        query="tree",
        knowledge_graph_inst=graph,
        entities_vdb=entities_vdb,
        query_param=QueryParam(top_k=1),
    )

    graph.get_nodes_batch.assert_awaited_once_with(["Tree"])
    assert relations == []
    assert nodes[0]["entity_id"] == "Tree"
    assert nodes[0]["entity_name"] == "Tree"
