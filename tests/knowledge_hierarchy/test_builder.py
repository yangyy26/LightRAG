import asyncio
import json

import pytest

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.knowledge_hierarchy import (
    ParentAssignment,
    _build_parent_assignment_prompt,
    assign_parents_from_relations,
    build_chunk_results_from_resource_graph,
    build_resource_knowledge_hierarchy,
    collect_hierarchy_candidates,
    filter_hierarchy_candidates_by_hard_rules,
    filter_hierarchy_candidates_by_grounding,
    normalize_assignments_to_hierarchy,
    normalize_hierarchy_tree,
    parse_parent_assignments,
)


def test_candidate_collection_dedupes_inside_one_file_and_merges_sources():
    chunk_results = [
        (
            {
                "Binary Tree": [
                    {
                        "entity_name": "Binary Tree",
                        "entity_type": "concept",
                        "description": "A tree where each node has at most two children.",
                        "source_id": "chunk-a",
                        "file_path": "trees.pdf",
                    }
                ],
                "binary tree": [
                    {
                        "entity_name": "binary tree",
                        "entity_type": "concept",
                        "description": "Binary trees are recursive data structures.",
                        "source_id": "chunk-b",
                        "file_path": "trees.pdf",
                    }
                ],
                "Author Name": [
                    {
                        "entity_name": "Author Name",
                        "entity_type": "person",
                        "description": "Document author.",
                        "source_id": "chunk-c",
                        "file_path": "trees.pdf",
                    }
                ],
            },
            {},
        )
    ]

    candidates = collect_hierarchy_candidates(
        chunk_results,
        candidate_entity_types=["concept"],
        file_path="trees.pdf",
    )

    assert list(candidates) == ["binary tree"]
    candidate = candidates["binary tree"]
    assert candidate.name == "Binary Tree"
    assert candidate.file_path == "trees.pdf"
    assert candidate.source_ids == ["chunk-a", "chunk-b"]
    assert "recursive data structures" in candidate.description


def test_candidate_collection_includes_all_types_when_filter_is_empty():
    chunk_results = [
        (
            {
                "Binary Tree": [
                    {
                        "entity_name": "Binary Tree",
                        "entity_type": "Other",
                        "description": "A tree where each node has at most two children.",
                        "source_id": "chunk-a",
                        "file_path": "trees.pdf",
                    }
                ]
            },
            {},
        )
    ]

    candidates = collect_hierarchy_candidates(
        chunk_results,
        candidate_entity_types=[],
        file_path="trees.pdf",
    )

    assert list(candidates) == ["binary tree"]


def test_candidate_grounding_filters_names_absent_from_current_document():
    candidates = collect_hierarchy_candidates(
        [
            (
                {
                    "Verified Resource Title: Norms, Theory, and Values": [
                        {
                            "entity_name": "Verified Resource Title: Norms, Theory, and Values",
                            "entity_type": "content",
                            "description": "Grounded title.",
                            "source_id": "doc-a-chunk-001",
                            "file_path": "doc.pdf",
                        }
                    ],
                    "Ungrounded Resource Title: Norms, Theory, and Values": [
                        {
                            "entity_name": "Ungrounded Resource Title: Norms, Theory, and Values",
                            "entity_type": "content",
                            "description": "Hallucinated title.",
                            "source_id": "doc-a-chunk-013",
                            "file_path": "doc.pdf",
                        }
                    ],
                },
                {},
            )
        ],
        candidate_entity_types=[],
        file_path="doc.pdf",
    )

    filtered = filter_hierarchy_candidates_by_grounding(
        candidates,
        "Title: Verified Resource Title: Norms, Theory, and Values",
    )

    assert set(filtered) == {"verified resource title: norms, theory, and values"}


def test_hard_candidate_filter_removes_obvious_non_knowledge_names():
    candidates = _candidate_map("课时01", "2017年1月3日", "2024-01-03", "123")

    filtered = filter_hierarchy_candidates_by_hard_rules(candidates)

    assert filtered == {}


def test_hard_candidate_filter_keeps_numbered_real_concepts():
    candidates = _candidate_map("二叉树", "HTTP 2", "第3范式")

    filtered = filter_hierarchy_candidates_by_hard_rules(candidates)

    assert set(filtered) == {"二叉树", "http 2", "第3范式"}


def test_hard_candidate_filter_keeps_cjk_single_character_concepts():
    candidates = _candidate_map("熵", "力")

    filtered = filter_hierarchy_candidates_by_hard_rules(candidates)

    assert set(filtered) == {"熵", "力"}


@pytest.mark.asyncio
async def test_resource_graph_chunk_results_trim_merged_node_to_current_doc_chunks():
    class FakeGraph:
        async def get_all_nodes(self):
            return [
                {
                    "entity_id": "Country",
                    "entity_name": "Country",
                    "entity_type": "location",
                    "description": "Current document description<SEP>Other document description",
                    "source_id": "doc-a-chunk-001<SEP>doc-b-chunk-002",
                    "file_path": "a.pdf<SEP>b.pdf",
                }
            ]

        async def get_all_edges(self):
            return []

    nodes, _edges = (
        await build_chunk_results_from_resource_graph(
            FakeGraph(),
            file_path="a.pdf",
            chunk_ids=["doc-a-chunk-001"],
        )
    )[0]

    record = nodes["Country"][0]
    assert record["source_id"] == "doc-a-chunk-001"
    assert record["file_path"] == "a.pdf"
    assert record["description"] == "Current document description"


def _candidate_map(*names: str):
    return collect_hierarchy_candidates(
        [
            (
                {
                    name: [
                        {
                            "entity_name": name,
                            "entity_type": "concept",
                            "description": f"{name} description",
                            "source_id": f"chunk-{idx}",
                            "file_path": "doc.pdf",
                        }
                    ]
                    for idx, name in enumerate(names)
                },
                {},
            )
        ],
        candidate_entity_types=[],
        file_path="doc.pdf",
    )


def test_parse_parent_assignments_accepts_valid_json_list():
    assignments = parse_parent_assignments(
        """
        [
          {
            "child": "Police Agency",
            "parent": "Legal Agency",
            "confidence": 0.92,
            "reason": "Police Agency is a kind of Legal Agency"
          },
          {
            "child": "Standalone Concept",
            "parent": null,
            "confidence": 0.1,
            "reason": "No clear parent"
          }
        ]
        """
    )

    assert assignments[0].child == "Police Agency"
    assert assignments[0].parent == "Legal Agency"
    assert assignments[0].confidence == 0.92
    assert assignments[1].child == "Standalone Concept"
    assert assignments[1].parent is None


def test_parse_parent_assignments_accepts_decision_field():
    assignments = parse_parent_assignments(
        {
            "assignments": [
                {
                    "child": "课时01",
                    "parent": None,
                    "confidence": 0.99,
                    "decision": "reject",
                    "reason": "课时标记，不是知识点",
                }
            ]
        }
    )

    assert assignments[0].child == "课时01"
    assert assignments[0].parent is None
    assert assignments[0].confidence == 0.99
    assert assignments[0].decision == "reject"


def test_parse_parent_assignments_ignores_invalid_rows():
    assignments = parse_parent_assignments(
        {
            "assignments": [
                {"child": "A", "parent": "B", "confidence": "0.8"},
                {"child": "", "parent": "B", "confidence": 0.9},
                {"parent": "B", "confidence": 0.9},
                "not a row",
            ]
        }
    )

    assert [(item.child, item.parent, item.confidence) for item in assignments] == [
        ("A", "B", 0.8)
    ]


def test_normalize_assignments_covers_all_candidates_and_roots_unassigned():
    candidates = _candidate_map("Parent", "Child", "Orphan")

    hierarchy = normalize_assignments_to_hierarchy(
        root_id="resource:doc-a",
        root_title="doc.pdf",
        file_path="doc.pdf",
        candidates=candidates,
        assignments=[
            ParentAssignment(child="Child", parent="Parent", confidence=0.9),
        ],
        max_depth=5,
        min_confidence=0.6,
    )

    edge_pairs = {(src, tgt) for src, tgt, _data in hierarchy.edges}
    assert edge_pairs == {
        ("resource:doc-a", "Parent"),
        ("Parent", "Child"),
        ("resource:doc-a", "Orphan"),
    }


def test_normalize_assignments_removes_semantically_rejected_candidates():
    candidates = _candidate_map("二叉树", "课时01")
    hierarchy = normalize_assignments_to_hierarchy(
        root_id="resource:doc-a",
        root_title="trees.pdf",
        file_path="trees.pdf",
        candidates=candidates,
        assignments=[
            ParentAssignment(
                child="课时01",
                parent=None,
                confidence=0.99,
                decision="reject",
                reason="课时标记，不是知识点",
            )
        ],
        max_depth=5,
        min_confidence=0.6,
    )

    edge_children = {target for _source, target, _data in hierarchy.edges}
    assert edge_children == {"二叉树"}


def test_normalize_assignments_keeps_all_candidates_when_filter_rejects_everything():
    # An over-aggressive semantic filter must not drop the entire hierarchy:
    # if every candidate is rejected we fall back to rooting them so the
    # resource still gets a (flat) hierarchy instead of failing with
    # "no valid hierarchy edges matched existing graph nodes".
    candidates = _candidate_map("邻补角", "补角")
    hierarchy = normalize_assignments_to_hierarchy(
        root_id="resource:doc-a",
        root_title="什么是邻居补角.mp4",
        file_path="什么是邻居补角.mp4",
        candidates=candidates,
        assignments=[
            ParentAssignment(
                child="邻补角",
                parent=None,
                confidence=0.99,
                decision="reject",
                reason="误判",
            ),
            ParentAssignment(
                child="补角",
                parent=None,
                confidence=0.99,
                decision="reject",
                reason="误判",
            ),
        ],
        max_depth=5,
        min_confidence=0.6,
    )

    edge_children = {target for _source, target, _data in hierarchy.edges}
    assert edge_children == {"邻补角", "补角"}
    # All fall back to root-level attachment.
    assert all(source == "resource:doc-a" for source, _target, _data in hierarchy.edges)


def test_normalize_assignments_keeps_missing_decision_as_root_candidate():
    candidates = _candidate_map("二叉树")
    hierarchy = normalize_assignments_to_hierarchy(
        root_id="resource:doc-a",
        root_title="trees.pdf",
        file_path="trees.pdf",
        candidates=candidates,
        assignments=[
            ParentAssignment(
                child="二叉树",
                parent=None,
                confidence=0.2,
            )
        ],
        max_depth=5,
        min_confidence=0.6,
    )

    assert len(hierarchy.edges) == 1
    assert hierarchy.edges[0][1] == "二叉树"


def test_normalize_assignments_treats_root_decision_with_parent_as_root():
    candidates = _candidate_map("Parent", "Child")
    hierarchy = normalize_assignments_to_hierarchy(
        root_id="resource:doc-a",
        root_title="doc.pdf",
        file_path="doc.pdf",
        candidates=candidates,
        assignments=[
            ParentAssignment(
                child="Child",
                parent="Parent",
                confidence=0.99,
                decision="root",
            )
        ],
        max_depth=5,
        min_confidence=0.6,
    )

    edge_by_child = {target: source for source, target, _data in hierarchy.edges}
    assert edge_by_child["Child"] == "resource:doc-a"


def test_parent_assignment_prompt_includes_reject_decisions():
    candidates = _candidate_map("二叉树", "课时01")

    prompt = _build_parent_assignment_prompt(
        root_title="trees.pdf",
        children=[candidates["二叉树"]],
        parent_candidates=[candidates["二叉树"]],
        relations=[],
        current_assignments=[
            ParentAssignment(
                child="课时01",
                parent=None,
                confidence=0.99,
                decision="reject",
                reason="课时标记，不是知识点",
            )
        ],
    )

    current_assignments_json = prompt.split("Current Assignments JSON:\n", 1)[1].split(
        "\nParent Assignment JSON:",
        1,
    )[0]

    assert json.loads(current_assignments_json) == [
        {
            "child": "课时01",
            "parent": None,
            "confidence": 0.99,
            "decision": "reject",
        }
    ]


def test_normalize_assignments_rejects_cycles_and_low_confidence():
    candidates = _candidate_map("A", "B", "C")

    hierarchy = normalize_assignments_to_hierarchy(
        root_id="resource:doc-a",
        root_title="doc.pdf",
        file_path="doc.pdf",
        candidates=candidates,
        assignments=[
            ParentAssignment(child="A", parent="B", confidence=0.95),
            ParentAssignment(child="B", parent="A", confidence=0.95),
            ParentAssignment(child="C", parent="A", confidence=0.2),
        ],
        max_depth=5,
        min_confidence=0.6,
    )

    edge_pairs = {(src, tgt) for src, tgt, _data in hierarchy.edges}
    assert ("A", "B") not in edge_pairs or ("B", "A") not in edge_pairs
    assert ("resource:doc-a", "C") in edge_pairs
    assert {tgt for _src, tgt, _data in hierarchy.edges} == {"A", "B", "C"}


def test_normalize_assignments_rejects_parent_child_compound_sibling_conflict():
    candidates = _candidate_map("Country", "Country Constitution", "Constitution")

    hierarchy = normalize_assignments_to_hierarchy(
        root_id="resource:doc-a",
        root_title="doc.pdf",
        file_path="doc.pdf",
        candidates=candidates,
        assignments=[
            ParentAssignment(child="Constitution", parent="Country", confidence=0.9),
        ],
        max_depth=5,
        min_confidence=0.6,
    )

    edge_pairs = {(src, tgt) for src, tgt, _data in hierarchy.edges}
    assert ("Country", "Constitution") not in edge_pairs
    assert ("resource:doc-a", "Constitution") in edge_pairs


def test_assign_parents_from_relations_prefers_hierarchy_keywords():
    candidates = _candidate_map("Parent Agency", "Child Agency", "Related Node")
    relations = [
        {
            "source": "Parent Agency",
            "target": "Child Agency",
            "description": "Child Agency belongs to Parent Agency.",
            "keywords": "belongs,classification",
        },
        {
            "source": "Parent Agency",
            "target": "Related Node",
            "description": "Parent Agency is related to Related Node.",
            "keywords": "related",
        },
    ]

    assignments = assign_parents_from_relations(candidates, relations)

    assert [(item.child, item.parent) for item in assignments] == [
        ("Child Agency", "Parent Agency")
    ]


def test_tree_normalization_uses_original_kg_nodes_for_hierarchy_edges():
    candidates = collect_hierarchy_candidates(
        [
            (
                {
                    "Tree": [
                        {
                            "entity_name": "Tree",
                            "entity_type": "concept",
                            "description": "A hierarchical data structure.",
                            "source_id": "chunk-a",
                            "file_path": "trees.pdf",
                        }
                    ],
                    "Binary Tree": [
                        {
                            "entity_name": "Binary Tree",
                            "entity_type": "concept",
                            "description": "A tree with at most two children per node.",
                            "source_id": "chunk-b",
                            "file_path": "trees.pdf",
                        }
                    ],
                },
                {},
            )
        ],
        candidate_entity_types=["concept"],
        file_path="trees.pdf",
    )

    normalized = normalize_hierarchy_tree(
        root_id="resource:doc-a",
        root_title="trees.pdf",
        file_path="trees.pdf",
        candidates=candidates,
        llm_tree={
            "nodes": [
                {
                    "name": "Tree",
                    "children": [{"name": "Binary Tree", "children": []}],
                }
            ]
        },
        max_depth=5,
    )

    assert normalized.nodes == []
    assert {(src, tgt) for src, tgt, _data in normalized.edges} == {
        ("resource:doc-a", "Tree"),
        ("Tree", "Binary Tree"),
    }


def test_tree_normalization_accepts_string_child_names():
    candidates = collect_hierarchy_candidates(
        [
            (
                {
                    "Parent": [
                        {
                            "entity_name": "Parent",
                            "entity_type": "concept",
                            "description": "Parent concept.",
                            "source_id": "chunk-a",
                            "file_path": "tree.pdf",
                        }
                    ],
                    "Child": [
                        {
                            "entity_name": "Child",
                            "entity_type": "concept",
                            "description": "Child concept.",
                            "source_id": "chunk-b",
                            "file_path": "tree.pdf",
                        }
                    ],
                },
                {},
            )
        ],
        candidate_entity_types=["concept"],
        file_path="tree.pdf",
    )

    normalized = normalize_hierarchy_tree(
        root_id="resource:doc-string-child",
        root_title="tree.pdf",
        file_path="tree.pdf",
        candidates=candidates,
        llm_tree={
            "nodes": [
                {
                    "name": "Parent",
                    "children": ["Child"],
                }
            ]
        },
        max_depth=5,
    )

    edge_pairs = [(src, tgt) for src, tgt, _data in normalized.edges]
    assert ("Parent", "Child") in edge_pairs
    assert ("resource:doc-string-child", "Child") not in edge_pairs


def test_tree_normalization_keeps_parent_when_child_appears_before_parent():
    candidates = collect_hierarchy_candidates(
        [
            (
                {
                    "Parent": [
                        {
                            "entity_name": "Parent",
                            "entity_type": "concept",
                            "description": "Parent concept.",
                            "source_id": "chunk-a",
                            "file_path": "tree.pdf",
                        }
                    ],
                    "Child": [
                        {
                            "entity_name": "Child",
                            "entity_type": "concept",
                            "description": "Child concept.",
                            "source_id": "chunk-b",
                            "file_path": "tree.pdf",
                        }
                    ],
                },
                {},
            )
        ],
        candidate_entity_types=["concept"],
        file_path="tree.pdf",
    )

    normalized = normalize_hierarchy_tree(
        root_id="resource:doc-order",
        root_title="tree.pdf",
        file_path="tree.pdf",
        candidates=candidates,
        llm_tree={
            "nodes": [
                {
                    "name": "Child",
                    "children": [],
                },
                {
                    "name": "Parent",
                    "children": [{"name": "Child", "children": []}],
                },
            ]
        },
        max_depth=5,
    )

    assert ("Parent", "Child") in [
        (src, tgt) for src, tgt, _data in normalized.edges
    ]
    assert ("resource:doc-order", "Child") not in [
        (src, tgt) for src, tgt, _data in normalized.edges
    ]


def test_tree_normalization_discards_external_nodes_and_attaches_orphans():
    candidates = collect_hierarchy_candidates(
        [
            (
                {
                    "Tree": [
                        {
                            "entity_name": "Tree",
                            "entity_type": "concept",
                            "description": "A hierarchical data structure.",
                            "source_id": "chunk-a",
                            "file_path": "trees.pdf",
                        }
                    ],
                    "Binary Tree": [
                        {
                            "entity_name": "Binary Tree",
                            "entity_type": "concept",
                            "description": "A tree with at most two children per node.",
                            "source_id": "chunk-b",
                            "file_path": "trees.pdf",
                        }
                    ],
                },
                {},
            )
        ],
        candidate_entity_types=["concept"],
        file_path="trees.pdf",
    )
    llm_tree = {
        "root_title": "trees.pdf",
        "nodes": [
            {
                "name": "Tree",
                "description": "Root concept",
                "source_chunk_ids": ["chunk-a"],
                "children": [
                    {
                        "name": "Invented Node",
                        "description": "Not allowed",
                        "source_chunk_ids": ["chunk-x"],
                        "children": [],
                    }
                ],
            }
        ],
    }

    normalized = normalize_hierarchy_tree(
        root_id="resource:doc-a",
        root_title="trees.pdf",
        file_path="trees.pdf",
        candidates=candidates,
        llm_tree=llm_tree,
        max_depth=5,
    )

    names = {node.entity_name for node in normalized.nodes}
    assert names == set()
    assert ("resource:doc-a", "Binary Tree") in [
        (src, tgt) for src, tgt, _data in normalized.edges
    ]


def test_tree_normalization_enforces_max_depth():
    candidates = collect_hierarchy_candidates(
        [
            (
                {
                    name: [
                        {
                            "entity_name": name,
                            "entity_type": "concept",
                            "description": name,
                            "source_id": f"chunk-{idx}",
                            "file_path": "deep.pdf",
                        }
                    ]
                    for idx, name in enumerate(["A", "B", "C"], start=1)
                },
                {},
            )
        ],
        candidate_entity_types=["concept"],
        file_path="deep.pdf",
    )
    llm_tree = {
        "nodes": [
            {
                "name": "A",
                "children": [
                    {
                        "name": "B",
                        "children": [
                            {"name": "C", "children": []},
                        ],
                    }
                ],
            }
        ],
    }

    normalized = normalize_hierarchy_tree(
        root_id="resource:doc-depth",
        root_title="deep.pdf",
        file_path="deep.pdf",
        candidates=candidates,
        llm_tree=llm_tree,
        max_depth=1,
    )

    assert normalized.nodes == []
    assert {src for src, _tgt, _data in normalized.edges} == {"resource:doc-depth"}
    assert GRAPH_FIELD_SEP not in normalized.root.entity_id


def test_tree_normalization_does_not_repair_flat_tree_from_weak_relations():
    candidates = collect_hierarchy_candidates(
        [
            (
                {
                    f"Concept {idx}": [
                        {
                            "entity_name": f"Concept {idx}",
                            "entity_type": "concept",
                            "description": f"Concept {idx}",
                            "source_id": f"chunk-{idx}",
                            "file_path": "weak.pdf",
                        }
                    ]
                    for idx in range(12)
                },
                {},
            )
        ],
        candidate_entity_types=["concept"],
        file_path="weak.pdf",
    )

    normalized = normalize_hierarchy_tree(
        root_id="resource:doc-weak",
        root_title="weak.pdf",
        file_path="weak.pdf",
        candidates=candidates,
        llm_tree={
            "nodes": [
                {"name": f"Concept {idx}", "children": []}
                for idx in range(12)
            ]
        },
        max_depth=5,
        relations=[
            {
                "source": "Concept 0",
                "target": "Concept 1",
                "description": "Concept 0 is related to Concept 1.",
                "keywords": "related",
            }
        ],
    )

    assert {
        src for src, _tgt, _data in normalized.edges
    } == {"resource:doc-weak"}


def test_tree_normalization_repairs_flat_tree_from_hierarchy_relations():
    candidates = collect_hierarchy_candidates(
        [
            (
                {
                    **{
                        f"Concept {idx}": [
                            {
                                "entity_name": f"Concept {idx}",
                                "entity_type": "concept",
                                "description": f"Concept {idx}",
                                "source_id": f"chunk-{idx}",
                                "file_path": "rule.pdf",
                            }
                        ]
                        for idx in range(10)
                    },
                    "Resource-Specific Rule": [
                        {
                            "entity_name": "Resource-Specific Rule",
                            "entity_type": "concept",
                            "description": "Rule body.",
                            "source_id": "chunk-a",
                            "file_path": "rule.pdf",
                        }
                    ],
                    "Governing Statute": [
                        {
                            "entity_name": "Governing Statute",
                            "entity_type": "concept",
                            "description": "Upper-level legal basis.",
                            "source_id": "chunk-b",
                            "file_path": "rule.pdf",
                        }
                    ],
                },
                {},
            )
        ],
        candidate_entity_types=["concept"],
        file_path="rule.pdf",
    )

    normalized = normalize_hierarchy_tree(
        root_id="resource:doc-rule",
        root_title="rule.pdf",
        file_path="rule.pdf",
        candidates=candidates,
        llm_tree={
            "nodes": [
                {"name": name, "children": []}
                for name in [
                    *[f"Concept {idx}" for idx in range(10)],
                    "Resource-Specific Rule",
                    "Governing Statute",
                ]
            ]
        },
        max_depth=5,
        relations=[
            {
                "source": "Governing Statute",
                "target": "Resource-Specific Rule",
                "description": "Resource-Specific Rule is a subtopic of Governing Statute.",
                "keywords": "subtopic,contains",
            }
        ],
    )

    assert ("Governing Statute", "Resource-Specific Rule") in [
        (src, tgt) for src, tgt, _data in normalized.edges
    ]


@pytest.mark.asyncio
async def test_build_resource_knowledge_hierarchy_calls_llm_and_normalizes_json():
    async def fake_llm(prompt: str, **kwargs):
        assert "Binary Tree" in prompt
        return {
            "assignments": [
                {
                    "child": "Binary Tree",
                    "parent": "Tree",
                    "confidence": 0.9,
                    "reason": "Binary Tree is a kind of Tree",
                }
            ]
        }

    chunk_results = [
        (
            {
                "Tree": [
                    {
                        "entity_name": "Tree",
                        "entity_type": "concept",
                        "description": "A hierarchical data structure.",
                        "source_id": "chunk-a",
                        "file_path": "trees.pdf",
                    }
                ],
                "Binary Tree": [
                    {
                        "entity_name": "Binary Tree",
                        "entity_type": "concept",
                        "description": "A tree with at most two children per node.",
                        "source_id": "chunk-b",
                        "file_path": "trees.pdf",
                    }
                ],
            },
            {},
        )
    ]

    hierarchy = await build_resource_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="trees.pdf",
        chunk_results=chunk_results,
        global_config={
            "enable_knowledge_hierarchy": True,
            "hierarchy_candidate_entity_types": ["concept"],
            "hierarchy_max_depth": 5,
            "llm_model_func": fake_llm,
        },
    )

    assert hierarchy is not None
    assert hierarchy.root.entity_id == "resource:doc-a"
    assert hierarchy.nodes == []
    assert ("Tree", "Binary Tree") in [
        (src, tgt) for src, tgt, _data in hierarchy.edges
    ]


@pytest.mark.asyncio
async def test_build_resource_knowledge_hierarchy_falls_back_when_grounding_strips_all():
    # Regression: media transcripts often have extracted entity names that do
    # not appear verbatim in the grounding text. The grounding filter must not
    # drop every candidate and cause "No hierarchy was generated".
    async def fake_llm(prompt: str, **kwargs):
        return {
            "assignments": [
                {
                    "child": "作物育种",
                    "parent": None,
                    "confidence": 0.9,
                    "reason": "root-level knowledge point",
                }
            ]
        }

    chunk_results = [
        (
            {
                "作物育种": [
                    {
                        "entity_name": "作物育种",
                        "entity_type": "knowledgepoint",
                        "description": "crop breeding",
                        "source_id": "chunk-a",
                        "file_path": "制订作物育种目标的原则2.mp4",
                    }
                ]
            },
            {},
        )
    ]

    hierarchy = await build_resource_knowledge_hierarchy(
        doc_id="doc-breeding",
        file_path="制订作物育种目标的原则2.mp4",
        chunk_results=chunk_results,
        global_config={
            "enable_knowledge_hierarchy": True,
            "hierarchy_candidate_entity_types": ["knowledgepoint", "concept"],
            "hierarchy_max_depth": 5,
            "llm_model_func": fake_llm,
        },
        # Grounding text that does NOT contain the candidate name -> the
        # grounding filter would remove everything without the fallback.
        grounding_text="这是一段与知识点名称不完全匹配的转写文本。",
    )

    assert hierarchy is not None
    edge_children = {tgt for _src, tgt, _data in hierarchy.edges}
    assert edge_children == {"作物育种"}


@pytest.mark.asyncio
async def test_build_resource_knowledge_hierarchy_keeps_low_frequency_candidates():
    async def fake_llm(prompt: str, **_kwargs):
        return {
            "assignments": [
                {
                    "child": "二叉树",
                    "parent": None,
                    "confidence": 0.8,
                    "decision": "root",
                    "reason": "valid concept",
                },
                {
                    "child": "红黑树",
                    "parent": None,
                    "confidence": 0.8,
                    "decision": "root",
                    "reason": "valid concept even when mentioned once",
                }
            ]
        }

    chunk_results = [
        (
            {
                "二叉树": [
                    {
                        "entity_name": "二叉树",
                        "entity_type": "concept",
                        "description": "二叉树知识点",
                        "source_id": "chunk-a",
                        "file_path": "trees.pdf",
                    }
                ],
                "红黑树": [
                    {
                        "entity_name": "红黑树",
                        "entity_type": "concept",
                        "description": "低频知识点",
                        "source_id": "chunk-a",
                        "file_path": "trees.pdf",
                    }
                ],
            },
            {},
        )
    ]

    hierarchy = await build_resource_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="trees.pdf",
        chunk_results=chunk_results,
        global_config={
            "enable_knowledge_hierarchy": True,
            "hierarchy_candidate_entity_types": ["concept"],
            "hierarchy_enable_hard_candidate_filter": True,
            "hierarchy_enable_semantic_candidate_filter": True,
            "llm_model_func": fake_llm,
        },
        grounding_text="二叉树是一种树。二叉树可以递归遍历。二叉树常用于搜索。红黑树只出现一次。",
    )

    assert hierarchy is not None
    assert {target for _source, target, _data in hierarchy.edges} == {"二叉树", "红黑树"}


@pytest.mark.asyncio
async def test_build_resource_knowledge_hierarchy_applies_hard_filter_before_llm():
    captured_prompts = []

    async def fake_llm(prompt: str, **_kwargs):
        captured_prompts.append(prompt)
        return {"assignments": []}

    chunk_results = [
        (
            {
                "二叉树": [
                    {
                        "entity_name": "二叉树",
                        "entity_type": "concept",
                        "description": "二叉树知识点",
                        "source_id": "chunk-a",
                        "file_path": "trees.pdf",
                    }
                ],
                "课时01": [
                    {
                        "entity_name": "课时01",
                        "entity_type": "concept",
                        "description": "课时标记",
                        "source_id": "chunk-b",
                        "file_path": "trees.pdf",
                    }
                ],
            },
            {},
        )
    ]

    hierarchy = await build_resource_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="trees.pdf",
        chunk_results=chunk_results,
        global_config={
            "enable_knowledge_hierarchy": True,
            "hierarchy_candidate_entity_types": ["concept"],
            "hierarchy_enable_hard_candidate_filter": True,
            "hierarchy_enable_semantic_candidate_filter": True,
            "llm_model_func": fake_llm,
        },
        grounding_text="二叉树 课时01",
    )

    assert hierarchy is not None
    assert "课时01" not in captured_prompts[0]
    assert {target for _source, target, _data in hierarchy.edges} == {"二叉树"}


@pytest.mark.asyncio
async def test_build_resource_knowledge_hierarchy_includes_extracted_relations_in_prompt():
    captured = {}

    async def fake_llm(prompt: str, **kwargs):
        captured["prompt"] = prompt
        return {"nodes": []}

    chunk_results = [
        (
            {
                "Tree": [
                    {
                        "entity_name": "Tree",
                        "entity_type": "concept",
                        "description": "A hierarchical data structure.",
                        "source_id": "chunk-a",
                        "file_path": "trees.pdf",
                    }
                ],
                "Binary Tree": [
                    {
                        "entity_name": "Binary Tree",
                        "entity_type": "concept",
                        "description": "A tree with at most two children per node.",
                        "source_id": "chunk-b",
                        "file_path": "trees.pdf",
                    }
                ],
            },
            {
                ("Tree", "Binary Tree"): [
                    {
                        "src_id": "Tree",
                        "tgt_id": "Binary Tree",
                        "description": "Binary Tree is a specialized kind of Tree.",
                        "keywords": "is-a,classification",
                        "source_id": "chunk-r",
                    }
                ]
            },
        )
    ]

    await build_resource_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="trees.pdf",
        chunk_results=chunk_results,
        global_config={
            "enable_knowledge_hierarchy": True,
            "hierarchy_candidate_entity_types": ["concept"],
            "hierarchy_max_depth": 5,
            "llm_model_func": fake_llm,
        },
    )

    assert "Relations JSON:" in captured["prompt"]
    assert "Tree" in captured["prompt"]
    assert "Binary Tree" in captured["prompt"]
    assert "specialized kind of Tree" in captured["prompt"]
    assert (
        "Use the document context, entity descriptions, and extracted relations"
        in captured["prompt"]
    )
    assert "Parent Assignment JSON:" in captured["prompt"]


@pytest.mark.asyncio
async def test_build_resource_knowledge_hierarchy_repairs_mostly_flat_llm_tree_from_relations():
    async def flat_llm(prompt: str, **kwargs):
        return {
            "root_title": "large.pdf",
            "nodes": [
                {"name": f"Concept {idx}", "children": []}
                for idx in range(82)
            ],
        }

    nodes = {
        f"Concept {idx}": [
            {
                "entity_name": f"Concept {idx}",
                "entity_type": "concept",
                "description": f"Concept {idx}",
                "source_id": f"chunk-{idx}",
                "file_path": "large.pdf",
            }
        ]
        for idx in range(82)
    }
    relations = {
        (f"Concept {idx}", f"Concept {idx + 1}"): [
            {
                "src_id": f"Concept {idx}",
                "tgt_id": f"Concept {idx + 1}",
                "description": (
                    f"Concept {idx + 1} is a subtopic of Concept {idx}."
                ),
                "keywords": "subtopic,contains",
                "source_id": f"chunk-r-{idx}",
            }
        ]
        for idx in range(40)
    }

    hierarchy = await build_resource_knowledge_hierarchy(
        doc_id="doc-large",
        file_path="large.pdf",
        chunk_results=[(nodes, relations)],
        global_config={
            "enable_knowledge_hierarchy": True,
            "hierarchy_candidate_entity_types": ["concept"],
            "hierarchy_max_depth": 5,
            "hierarchy_max_candidates": 82,
            "llm_model_func": flat_llm,
        },
    )

    assert hierarchy is not None
    root_edges = [
        (src, tgt) for src, tgt, _data in hierarchy.edges if src == "resource:doc-large"
    ]
    non_root_edges = [
        (src, tgt) for src, tgt, _data in hierarchy.edges if src != "resource:doc-large"
    ]
    assert len(non_root_edges) >= 20
    assert len(root_edges) <= 62
    assert ("Concept 0", "Concept 1") in non_root_edges


@pytest.mark.asyncio
async def test_build_resource_knowledge_hierarchy_roots_all_candidates_on_llm_failure():
    async def broken_llm(prompt: str, **kwargs):
        raise RuntimeError("provider down")

    nodes = {
        f"Concept {idx}": [
            {
                "entity_name": f"Concept {idx}",
                "entity_type": "concept",
                "description": f"Concept {idx}",
                "source_id": f"chunk-{idx}",
                "file_path": "trees.pdf",
            }
        ]
        for idx in range(12)
    }
    relations = {
        ("Concept 0", "Concept 1"): [
            {
                "src_id": "Concept 0",
                "tgt_id": "Concept 1",
                "description": "Concept 0 contains Concept 1.",
                "keywords": "contains",
                "source_id": "chunk-r",
            }
        ]
    }

    hierarchy = await build_resource_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="trees.pdf",
        chunk_results=[(nodes, relations)],
        global_config={
            "enable_knowledge_hierarchy": True,
            "hierarchy_candidate_entity_types": ["concept"],
            "hierarchy_max_depth": 5,
            "llm_model_func": broken_llm,
        },
    )

    assert hierarchy is not None
    edge_pairs = {(src, tgt) for src, tgt, _data in hierarchy.edges}
    assert len(hierarchy.edges) == 12
    assert ("Concept 0", "Concept 1") in edge_pairs


@pytest.mark.asyncio
async def test_build_resource_knowledge_hierarchy_batches_all_candidates():
    prompts = []

    async def fake_llm(prompt: str, **kwargs):
        prompts.append(prompt)
        if "Parent Assignment JSON" in prompt:
            return {
                "assignments": [
                    {
                        "child": "Concept 1",
                        "parent": "Concept 0",
                        "confidence": 0.9,
                        "reason": "Concept 1 is part of Concept 0",
                    },
                    {
                        "child": "Concept 3",
                        "parent": "Concept 2",
                        "confidence": 0.9,
                        "reason": "Concept 3 is part of Concept 2",
                    },
                ]
            }
        return {"nodes": []}

    chunk_results = [
        (
            {
                f"Concept {idx}": [
                    {
                        "entity_name": f"Concept {idx}",
                        "entity_type": "concept",
                        "description": f"Concept {idx}",
                        "source_id": f"chunk-{idx}",
                        "file_path": "many.pdf",
                    }
                ]
                for idx in range(5)
            },
            {},
        )
    ]

    hierarchy = await build_resource_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="many.pdf",
        chunk_results=chunk_results,
        global_config={
            "enable_knowledge_hierarchy": True,
            "hierarchy_candidate_entity_types": [],
            "hierarchy_max_depth": 5,
            "hierarchy_batch_size": 2,
            "hierarchy_max_rounds": 1,
            "hierarchy_parent_candidate_limit": 4,
            "hierarchy_min_parent_confidence": 0.6,
            "llm_model_func": fake_llm,
        },
    )

    assert hierarchy is not None
    edge_pairs = {(src, tgt) for src, tgt, _data in hierarchy.edges}
    assert {tgt for _src, tgt, _data in hierarchy.edges} == {
        f"Concept {idx}" for idx in range(5)
    }
    assert ("Concept 0", "Concept 1") in edge_pairs
    assert ("Concept 2", "Concept 3") in edge_pairs
    assert ("resource:doc-a", "Concept 4") in edge_pairs
    assert any("Parent Assignment JSON" in prompt for prompt in prompts)


@pytest.mark.asyncio
async def test_build_resource_knowledge_hierarchy_does_not_retry_rejected_candidates():
    prompts = []

    async def fake_llm(prompt: str, **_kwargs):
        prompts.append(prompt)
        if len(prompts) > 1:
            return {"assignments": []}
        return {
            "assignments": [
                {
                    "child": "课时01",
                    "parent": None,
                    "confidence": 0.99,
                    "decision": "reject",
                    "reason": "课时标记，不是知识点",
                },
                {
                    "child": "二叉树",
                    "parent": "树",
                    "confidence": 0.9,
                    "decision": "assign",
                    "reason": "二叉树是一种树",
                },
            ]
        }

    chunk_results = [
        (
            {
                "课时01": [
                    {
                        "entity_name": "课时01",
                        "entity_type": "concept",
                        "description": "课时标记",
                        "source_id": "chunk-a",
                        "file_path": "doc.pdf",
                    }
                ],
                "树": [
                    {
                        "entity_name": "树",
                        "entity_type": "concept",
                        "description": "树结构",
                        "source_id": "chunk-b",
                        "file_path": "doc.pdf",
                    }
                ],
                "二叉树": [
                    {
                        "entity_name": "二叉树",
                        "entity_type": "concept",
                        "description": "二叉树知识点",
                        "source_id": "chunk-c",
                        "file_path": "doc.pdf",
                    }
                ],
            },
            {},
        )
    ]

    hierarchy = await build_resource_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="doc.pdf",
        chunk_results=chunk_results,
        global_config={
            "enable_knowledge_hierarchy": True,
            "hierarchy_candidate_entity_types": ["concept"],
            "hierarchy_enable_hard_candidate_filter": False,
            "hierarchy_enable_semantic_candidate_filter": True,
            "hierarchy_max_rounds": 2,
            "llm_model_func": fake_llm,
        },
        grounding_text="课时01 课时01 课时01 树 二叉树",
    )

    assert hierarchy is not None
    edge_children = {target for _source, target, _data in hierarchy.edges}
    assert edge_children == {"树", "二叉树"}
    assert len(prompts) == 2

    second_unresolved_json = prompts[1].split("Unresolved Children JSON:\n", 1)[
        1
    ].split("\nParent Candidates JSON:", 1)[0]
    second_unresolved_names = {
        item["name"] for item in json.loads(second_unresolved_json)
    }
    assert "课时01" not in second_unresolved_names


@pytest.mark.asyncio
async def test_build_resource_knowledge_hierarchy_ignores_reject_when_semantic_filter_disabled():
    prompts = []

    async def fake_llm(prompt: str, **_kwargs):
        prompts.append(prompt)
        return {
            "assignments": [
                {
                    "child": "课时01",
                    "parent": None,
                    "confidence": 0.99,
                    "decision": "reject",
                    "reason": "ignored because semantic filter disabled",
                }
            ]
        }

    chunk_results = [
        (
            {
                "课时01": [
                    {
                        "entity_name": "课时01",
                        "entity_type": "concept",
                        "description": "课时标记",
                        "source_id": "chunk-a",
                        "file_path": "doc.pdf",
                    }
                ]
            },
            {},
        )
    ]

    hierarchy = await build_resource_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="doc.pdf",
        chunk_results=chunk_results,
        global_config={
            "enable_knowledge_hierarchy": True,
            "hierarchy_candidate_entity_types": ["concept"],
            "hierarchy_enable_hard_candidate_filter": False,
            "hierarchy_enable_semantic_candidate_filter": False,
            "llm_model_func": fake_llm,
        },
        grounding_text="课时01 课时01 课时01",
    )

    assert hierarchy is not None
    assert len(prompts) == 1
    assert '"reject"' not in prompts[0]
    assert len(hierarchy.edges) == 1
    assert hierarchy.edges[0][0] == "resource:doc-a"
    assert hierarchy.edges[0][1] == "课时01"


@pytest.mark.asyncio
async def test_build_resource_knowledge_hierarchy_counts_reject_as_round_progress():
    prompts = []

    async def fake_llm(prompt: str, **_kwargs):
        prompts.append(prompt)
        if len(prompts) == 1:
            return {
                "assignments": [
                    {
                        "child": "课时01",
                        "parent": None,
                        "confidence": 0.99,
                        "decision": "reject",
                        "reason": "课时标记，不是知识点",
                    }
                ]
            }
        return {
            "assignments": [
                {
                    "child": "二叉树",
                    "parent": "树",
                    "confidence": 0.9,
                    "decision": "assign",
                    "reason": "二叉树是一种树",
                }
            ]
        }

    chunk_results = [
        (
            {
                "课时01": [
                    {
                        "entity_name": "课时01",
                        "entity_type": "concept",
                        "description": "课时标记",
                        "source_id": "chunk-a",
                        "file_path": "doc.pdf",
                    }
                ],
                "树": [
                    {
                        "entity_name": "树",
                        "entity_type": "concept",
                        "description": "树结构",
                        "source_id": "chunk-b",
                        "file_path": "doc.pdf",
                    }
                ],
                "二叉树": [
                    {
                        "entity_name": "二叉树",
                        "entity_type": "concept",
                        "description": "二叉树知识点",
                        "source_id": "chunk-c",
                        "file_path": "doc.pdf",
                    }
                ],
            },
            {},
        )
    ]

    hierarchy = await build_resource_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="doc.pdf",
        chunk_results=chunk_results,
        global_config={
            "enable_knowledge_hierarchy": True,
            "hierarchy_candidate_entity_types": ["concept"],
            "hierarchy_enable_hard_candidate_filter": False,
            "hierarchy_enable_semantic_candidate_filter": True,
            "hierarchy_batch_size": 3,
            "hierarchy_max_rounds": 2,
            "llm_model_func": fake_llm,
        },
        grounding_text="课时01 树 二叉树",
    )

    assert hierarchy is not None
    edge_children = {target for _source, target, _data in hierarchy.edges}
    assert edge_children == {"树", "二叉树"}
    assert len(prompts) == 2

    second_unresolved_json = prompts[1].split("Unresolved Children JSON:\n", 1)[
        1
    ].split("\nParent Candidates JSON:", 1)[0]
    second_unresolved_names = {
        item["name"] for item in json.loads(second_unresolved_json)
    }
    assert second_unresolved_names == {"树", "二叉树"}


@pytest.mark.asyncio
async def test_build_resource_knowledge_hierarchy_does_not_retry_root_decisions():
    prompts = []

    async def fake_llm(prompt: str, **_kwargs):
        prompts.append(prompt)
        if len(prompts) > 1:
            return {"assignments": []}
        return {
            "assignments": [
                {
                    "child": "Standalone",
                    "parent": None,
                    "confidence": 0.99,
                    "decision": "root",
                    "reason": "Valid root concept",
                },
                {
                    "child": "Child",
                    "parent": "Parent",
                    "confidence": 0.9,
                    "decision": "assign",
                    "reason": "Child belongs to Parent",
                },
            ]
        }

    chunk_results = [
        (
            {
                "Parent": [
                    {
                        "entity_name": "Parent",
                        "entity_type": "concept",
                        "description": "Parent concept",
                        "source_id": "chunk-a",
                        "file_path": "doc.pdf",
                    }
                ],
                "Child": [
                    {
                        "entity_name": "Child",
                        "entity_type": "concept",
                        "description": "Child concept",
                        "source_id": "chunk-b",
                        "file_path": "doc.pdf",
                    }
                ],
                "Standalone": [
                    {
                        "entity_name": "Standalone",
                        "entity_type": "concept",
                        "description": "Standalone concept",
                        "source_id": "chunk-c",
                        "file_path": "doc.pdf",
                    }
                ],
            },
            {},
        )
    ]

    hierarchy = await build_resource_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="doc.pdf",
        chunk_results=chunk_results,
        global_config={
            "enable_knowledge_hierarchy": True,
            "hierarchy_candidate_entity_types": ["concept"],
            "hierarchy_enable_hard_candidate_filter": False,
            "hierarchy_enable_semantic_candidate_filter": True,
            "hierarchy_max_rounds": 2,
            "llm_model_func": fake_llm,
        },
        grounding_text="Parent Child Standalone",
    )

    assert hierarchy is not None
    assert len(prompts) == 2

    second_unresolved_json = prompts[1].split("Unresolved Children JSON:\n", 1)[
        1
    ].split("\nParent Candidates JSON:", 1)[0]
    second_unresolved_names = {
        item["name"] for item in json.loads(second_unresolved_json)
    }
    assert "Standalone" not in second_unresolved_names


@pytest.mark.asyncio
async def test_build_resource_knowledge_hierarchy_roots_failed_batch_children():
    calls = 0

    async def flaky_llm(prompt: str, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("provider timeout")
        return {
            "assignments": [
                {
                    "child": "Concept 3",
                    "parent": "Concept 2",
                    "confidence": 0.9,
                    "reason": "Concept 3 belongs to Concept 2",
                }
            ]
        }

    chunk_results = [
        (
            {
                f"Concept {idx}": [
                    {
                        "entity_name": f"Concept {idx}",
                        "entity_type": "concept",
                        "description": f"Concept {idx}",
                        "source_id": f"chunk-{idx}",
                        "file_path": "many.pdf",
                    }
                ]
                for idx in range(4)
            },
            {},
        )
    ]

    hierarchy = await build_resource_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="many.pdf",
        chunk_results=chunk_results,
        global_config={
            "enable_knowledge_hierarchy": True,
            "hierarchy_candidate_entity_types": [],
            "hierarchy_max_depth": 5,
            "hierarchy_batch_size": 2,
            "hierarchy_max_rounds": 1,
            "hierarchy_parent_candidate_limit": 4,
            "hierarchy_min_parent_confidence": 0.6,
            "llm_model_func": flaky_llm,
        },
    )

    assert hierarchy is not None
    edge_pairs = {(src, tgt) for src, tgt, _data in hierarchy.edges}
    assert {tgt for _src, tgt, _data in hierarchy.edges} == {
        "Concept 0",
        "Concept 1",
        "Concept 2",
        "Concept 3",
    }
    assert ("Concept 2", "Concept 3") in edge_pairs
    assert ("resource:doc-a", "Concept 0") in edge_pairs
    assert ("resource:doc-a", "Concept 1") in edge_pairs


@pytest.mark.asyncio
async def test_build_resource_knowledge_hierarchy_limits_parallel_batches():
    active = 0
    max_active = 0

    async def slow_llm(prompt: str, **kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {"assignments": []}

    chunk_results = [
        (
            {
                f"Concept {idx}": [
                    {
                        "entity_name": f"Concept {idx}",
                        "entity_type": "concept",
                        "description": f"Concept {idx}",
                        "source_id": f"chunk-{idx}",
                        "file_path": "many.pdf",
                    }
                ]
                for idx in range(6)
            },
            {},
        )
    ]

    hierarchy = await build_resource_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="many.pdf",
        chunk_results=chunk_results,
        global_config={
            "enable_knowledge_hierarchy": True,
            "hierarchy_candidate_entity_types": [],
            "hierarchy_max_depth": 5,
            "hierarchy_batch_size": 1,
            "hierarchy_max_rounds": 1,
            "hierarchy_parent_candidate_limit": 4,
            "hierarchy_min_parent_confidence": 0.6,
            "hierarchy_max_parallel_batches": 2,
            "llm_model_func": slow_llm,
        },
    )

    assert hierarchy is not None
    assert len(hierarchy.edges) == 6
    assert max_active == 2


@pytest.mark.asyncio
async def test_build_resource_knowledge_hierarchy_prefers_extract_role_llm():
    calls = []

    async def extract_role_llm(prompt: str, **kwargs):
        calls.append("extract")
        return {"nodes": []}

    async def base_llm(prompt: str, **kwargs):
        calls.append("base")
        return {"nodes": []}

    chunk_results = [
        (
            {
                "Concept": [
                    {
                        "entity_name": "Concept",
                        "entity_type": "concept",
                        "description": "A concept.",
                        "source_id": "chunk-a",
                        "file_path": "many.pdf",
                    }
                ]
            },
            {},
        )
    ]

    hierarchy = await build_resource_knowledge_hierarchy(
        doc_id="doc-a",
        file_path="many.pdf",
        chunk_results=chunk_results,
        global_config={
            "enable_knowledge_hierarchy": True,
            "hierarchy_candidate_entity_types": ["concept"],
            "hierarchy_max_depth": 5,
            "role_llm_funcs": {"extract": extract_role_llm},
            "llm_model_func": base_llm,
        },
    )

    assert hierarchy is not None
    assert calls == ["extract"]
