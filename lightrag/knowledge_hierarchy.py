from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
import inspect

import json
import json_repair

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import (
    get_llm_cache_identity,
    logger,
    split_string_by_multi_markers,
    use_llm_func_with_cache,
)


HIERARCHY_EDGE_TYPE = "hierarchy"
HIERARCHY_RELATION_TYPE = "contains"
HIERARCHY_ROOT_KIND = "root"
HIERARCHY_KP_KIND = "knowledge_point"
HIERARCHY_STATUS_PROCESSING = "processing"
HIERARCHY_STATUS_SUCCESS = "success"
HIERARCHY_STATUS_FAILED = "failed"


class KnowledgeHierarchyBuildError(RuntimeError):
    """Raised when the dedicated hierarchy LLM step fails."""


@dataclass
class HierarchyCandidate:
    key: str
    name: str
    description: str
    source_ids: list[str]
    file_path: str
    entity_type: str


@dataclass
class ParentAssignment:
    child: str
    parent: str | None
    confidence: float
    reason: str = ""


@dataclass
class HierarchyNode:
    entity_id: str
    entity_name: str
    entity_type: str
    description: str
    source_id: str
    file_path: str
    hierarchy_kind: str
    root_id: str
    parent_id: str | None
    level: int
    hierarchy_confidence: float = 1.0

    def to_graph_data(self) -> dict[str, Any]:
        data = {
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "description": self.description,
            "source_id": self.source_id,
            "file_path": self.file_path,
            "hierarchy_kind": self.hierarchy_kind,
            "root_id": self.root_id,
            "level": self.level,
        }
        if self.hierarchy_kind == HIERARCHY_ROOT_KIND and self.entity_id.startswith(
            "resource:"
        ):
            data["doc_id"] = self.entity_id.removeprefix("resource:")
        if self.parent_id is not None:
            data["parent_id"] = self.parent_id
            data["hierarchy_confidence"] = self.hierarchy_confidence
        return data


@dataclass
class NormalizedHierarchy:
    root: HierarchyNode
    nodes: list[HierarchyNode] = field(default_factory=list)
    edges: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)


def normalize_knowledge_point_name(name: str) -> str:
    return " ".join(str(name or "").strip().lower().split())


def make_resource_root_id(doc_id: str) -> str:
    return f"resource:{doc_id}"


def _merge_source_ids(left: list[str], right: str | list[str] | None) -> list[str]:
    seen = set(left)
    result = list(left)
    raw_items: list[str]
    if isinstance(right, list):
        raw_items = right
    elif isinstance(right, str):
        raw_items = split_string_by_multi_markers(right, [GRAPH_FIELD_SEP])
    else:
        raw_items = []
    for item in raw_items:
        item = str(item).strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def collect_hierarchy_candidates(
    chunk_results: list[tuple[dict[str, list[dict]], dict]],
    candidate_entity_types: list[str],
    file_path: str,
) -> dict[str, HierarchyCandidate]:
    allowed = {item.strip().lower() for item in candidate_entity_types if item.strip()}
    candidates: dict[str, HierarchyCandidate] = {}

    for maybe_nodes, _maybe_edges in chunk_results:
        for entity_name, entity_records in maybe_nodes.items():
            for record in entity_records:
                entity_type = str(record.get("entity_type", "")).strip().lower()
                if allowed and entity_type not in allowed:
                    continue
                key = normalize_knowledge_point_name(entity_name)
                if not key:
                    continue
                record_file_path = str(record.get("file_path") or file_path)
                description = str(record.get("description") or "").strip()
                if key not in candidates:
                    candidates[key] = HierarchyCandidate(
                        key=key,
                        name=str(record.get("entity_name") or entity_name),
                        description=description,
                        source_ids=[],
                        file_path=record_file_path,
                        entity_type=entity_type or "knowledgepoint",
                    )
                candidate = candidates[key]
                candidate.source_ids = _merge_source_ids(
                    candidate.source_ids, record.get("source_id")
                )
                if description and description not in candidate.description:
                    if candidate.description:
                        candidate.description = (
                            f"{candidate.description}\n{description}"
                        )
                    else:
                        candidate.description = description
                elif len(description) > len(candidate.description):
                    candidate.description = description

    return candidates


def _grounding_text_key(text: str) -> str:
    return "".join(str(text or "").lower().split())


def _candidate_name_is_grounded(name: str, grounding_text: str) -> bool:
    name_key = _grounding_text_key(name)
    if not name_key:
        return False
    grounding_key = _grounding_text_key(grounding_text)
    if name_key in grounding_key:
        return True
    if len(name_key) <= 4:
        return name_key in grounding_key

    separators = "：:，,。；;（）()《》[]【】“”\"'、"
    normalized = name_key
    for sep in separators:
        normalized = normalized.replace(sep, " ")
    parts = [part for part in normalized.split() if len(part) >= 2]
    if not parts:
        return False
    matched = sum(1 for part in parts if part in grounding_key)
    return matched == len(parts)


def filter_hierarchy_candidates_by_grounding(
    candidates: dict[str, HierarchyCandidate],
    grounding_text: str,
) -> dict[str, HierarchyCandidate]:
    if not grounding_text:
        return candidates

    filtered = {
        key: candidate
        for key, candidate in candidates.items()
        if _candidate_name_is_grounded(candidate.name, grounding_text)
    }
    removed = len(candidates) - len(filtered)
    if removed:
        logger.info(
            "Filtered ungrounded hierarchy candidates: kept=%d, removed=%d",
            len(filtered),
            removed,
        )
    return filtered


def _iter_llm_tree_nodes(nodes: Any, parent_key: str | None, level: int):
    if not isinstance(nodes, list):
        return
    for item in nodes:
        if isinstance(item, str):
            key = normalize_knowledge_point_name(item)
            if key:
                yield key, {"name": item, "children": []}, parent_key, level
            continue
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        key = normalize_knowledge_point_name(str(name or ""))
        if key:
            yield key, item, parent_key, level
            yield from _iter_llm_tree_nodes(item.get("children", []), key, level + 1)


def _source_id_from_candidate(candidate: HierarchyCandidate) -> str:
    return GRAPH_FIELD_SEP.join(candidate.source_ids)


def collect_hierarchy_relations(
    chunk_results: list[tuple[dict[str, list[dict]], dict]],
    candidates: dict[str, HierarchyCandidate],
    max_relations: int = 120,
) -> list[dict[str, Any]]:
    if max_relations <= 0:
        return []

    candidate_key_to_name = {key: candidate.name for key, candidate in candidates.items()}
    relations_by_pair: dict[tuple[str, str], dict[str, Any]] = {}

    for _maybe_nodes, maybe_edges in chunk_results:
        for edge_key, edge_records in (maybe_edges or {}).items():
            if (
                isinstance(edge_key, tuple)
                and len(edge_key) == 2
                and all(isinstance(item, str) for item in edge_key)
            ):
                raw_src, raw_tgt = edge_key
            else:
                raw_src, raw_tgt = "", ""

            for record in edge_records:
                src = str(record.get("src_id") or raw_src).strip()
                tgt = str(record.get("tgt_id") or raw_tgt).strip()
                src_key = normalize_knowledge_point_name(src)
                tgt_key = normalize_knowledge_point_name(tgt)
                if (
                    not src_key
                    or not tgt_key
                    or src_key == tgt_key
                    or src_key not in candidates
                    or tgt_key not in candidates
                ):
                    continue

                pair = (src_key, tgt_key)
                description = str(record.get("description") or "").strip()
                keywords = str(record.get("keywords") or "").strip()
                source_id = str(record.get("source_id") or "").strip()
                if pair not in relations_by_pair:
                    relations_by_pair[pair] = {
                        "source": candidate_key_to_name[src_key],
                        "target": candidate_key_to_name[tgt_key],
                        "description": description,
                        "keywords": keywords,
                        "source_id": source_id,
                    }
                    continue

                relation = relations_by_pair[pair]
                if description and description not in relation["description"]:
                    relation["description"] = (
                        f"{relation['description']}\n{description}"
                        if relation["description"]
                        else description
                    )
                if keywords and keywords not in relation["keywords"]:
                    relation["keywords"] = (
                        f"{relation['keywords']},{keywords}"
                        if relation["keywords"]
                        else keywords
                    )

    relations = sorted(
        relations_by_pair.values(),
        key=lambda item: (
            -len(item.get("description") or ""),
            item.get("source") or "",
            item.get("target") or "",
        ),
    )
    return relations[:max_relations]


_HIERARCHY_RELATION_HINTS = {
    "contain",
    "contains",
    "subtopic",
    "sub-topic",
    "subclass",
    "is-a",
    "kind of",
    "type of",
    "part of",
    "component",
    "step",
    "prerequisite",
    "depends on",
    "includes",
    "consists",
    "belongs",
    "scope",
    "taxonomy",
    "classification",
    "包含",
    "包括",
    "属于",
    "组成",
    "构成",
    "部分",
    "依据",
    "根据",
    "上位",
    "下位",
    "分类",
    "类型",
    "子类",
    "步骤",
    "流程",
    "前置",
    "依赖",
    "范围",
    "立法依据",
    "法律效力",
}


def _relation_supports_hierarchy(relation: dict[str, Any]) -> bool:
    text = " ".join(
        str(relation.get(field) or "").lower()
        for field in ("description", "keywords")
    )
    return any(hint in text for hint in _HIERARCHY_RELATION_HINTS)


def assign_parents_from_relations(
    candidates: dict[str, HierarchyCandidate],
    relations: list[dict[str, Any]],
) -> list[ParentAssignment]:
    candidate_by_name = {
        normalize_knowledge_point_name(candidate.name): candidate
        for candidate in candidates.values()
    }
    assignments: list[ParentAssignment] = []
    seen_children: set[str] = set()

    for relation in relations:
        if not _relation_supports_hierarchy(relation):
            continue
        source_key = normalize_knowledge_point_name(relation.get("source"))
        target_key = normalize_knowledge_point_name(relation.get("target"))
        source = candidate_by_name.get(source_key)
        target = candidate_by_name.get(target_key)
        if source is None or target is None or source.key == target.key:
            continue
        if target.key in seen_children:
            continue
        seen_children.add(target.key)
        assignments.append(
            ParentAssignment(
                child=target.name,
                parent=source.name,
                confidence=0.95,
                reason=str(
                    relation.get("description") or relation.get("keywords") or ""
                ),
            )
        )

    return assignments


def _tree_depth_for_key(
    key: str,
    parent_by_key: dict[str, str],
    key_by_entity_name: dict[str, str],
    root_id: str,
) -> int:
    depth = 1
    visited = {key}
    parent_id = parent_by_key.get(key)
    while parent_id and parent_id != root_id:
        parent_key = key_by_entity_name.get(parent_id)
        if parent_key is None or parent_key in visited:
            return 9999
        visited.add(parent_key)
        depth += 1
        parent_id = parent_by_key.get(parent_key)
    return depth


def _would_create_cycle(
    parent_key: str,
    child_key: str,
    parent_by_key: dict[str, str],
    key_by_entity_name: dict[str, str],
    root_id: str,
) -> bool:
    current_key = parent_key
    visited = {child_key}
    while current_key:
        if current_key == child_key:
            return True
        if current_key in visited:
            return True
        visited.add(current_key)
        parent_id = parent_by_key.get(current_key)
        if not parent_id or parent_id == root_id:
            return False
        current_key = key_by_entity_name.get(parent_id)
    return False


def _repair_flat_hierarchy_from_relations(
    *,
    parent_by_key: dict[str, str],
    level_by_key: dict[str, int],
    candidates: dict[str, HierarchyCandidate],
    relations: list[dict[str, Any]],
    root_id: str,
    max_depth: int,
) -> int:
    if len(parent_by_key) < 10 or not relations:
        return 0

    root_child_count = sum(
        1 for parent_id in parent_by_key.values() if parent_id == root_id
    )
    if root_child_count / len(parent_by_key) < 0.8:
        return 0

    key_by_entity_name = {
        candidate.name: key for key, candidate in candidates.items()
    }
    repaired = 0
    for relation in relations:
        if not _relation_supports_hierarchy(relation):
            continue
        parent_key = normalize_knowledge_point_name(relation.get("source"))
        child_key = normalize_knowledge_point_name(relation.get("target"))
        if (
            not parent_key
            or not child_key
            or parent_key == child_key
            or parent_key not in parent_by_key
            or child_key not in parent_by_key
            or parent_by_key.get(child_key) != root_id
        ):
            continue
        if _would_create_cycle(
            parent_key, child_key, parent_by_key, key_by_entity_name, root_id
        ):
            continue

        parent_depth = _tree_depth_for_key(
            parent_key, parent_by_key, key_by_entity_name, root_id
        )
        child_depth = parent_depth + 1
        if child_depth > max_depth:
            continue

        parent_by_key[child_key] = candidates[parent_key].name
        level_by_key[child_key] = child_depth
        repaired += 1

    return repaired


def _make_hierarchy_edge(
    parent_id: str,
    child: HierarchyNode,
) -> tuple[str, str, dict[str, Any]]:
    return (
        parent_id,
        child.entity_id,
        {
            "description": "Parent contains child in the resource knowledge hierarchy.",
            "keywords": "contains,hierarchy",
            "weight": 1.0,
            "source_id": child.source_id,
            "file_path": child.file_path,
            "edge_type": HIERARCHY_EDGE_TYPE,
            "relation_type": HIERARCHY_RELATION_TYPE,
            "root_id": child.root_id,
            "parent_id": parent_id,
            "child_id": child.entity_id,
        },
    )


def normalize_hierarchy_tree(
    root_id: str,
    root_title: str,
    file_path: str,
    candidates: dict[str, HierarchyCandidate],
    llm_tree: dict[str, Any],
    max_depth: int,
    relations: list[dict[str, Any]] | None = None,
) -> NormalizedHierarchy:
    root = HierarchyNode(
        entity_id=root_id,
        entity_name=root_title,
        entity_type="Resource",
        description=root_title,
        source_id="",
        file_path=file_path,
        hierarchy_kind=HIERARCHY_ROOT_KIND,
        root_id=root_id,
        parent_id=None,
        level=0,
    )

    seen: set[str] = set()
    llm_parent_by_key: dict[str, str | None] = {}
    llm_level_by_key: dict[str, int] = {}
    parent_by_key: dict[str, str] = {}
    level_by_key: dict[str, int] = {}

    for key, _item, parent_key, level in _iter_llm_tree_nodes(
        llm_tree.get("nodes", []), None, 1
    ):
        if key not in candidates:
            continue
        seen.add(key)
        current_level = llm_level_by_key.get(key, 0)
        if key not in llm_parent_by_key or level > current_level:
            llm_parent_by_key[key] = parent_key if parent_key in candidates else None
            llm_level_by_key[key] = level

    for key in candidates:
        if key not in seen:
            seen.add(key)
            llm_parent_by_key[key] = None
            llm_level_by_key[key] = 1

    for key in seen:
        parent_key = llm_parent_by_key.get(key)
        level = llm_level_by_key.get(key, 1)
        if level > max_depth or not parent_key or parent_key not in seen:
            parent_by_key[key] = root_id
            level_by_key[key] = 1
        else:
            parent_by_key[key] = candidates[parent_key].name
            level_by_key[key] = level

    repaired_count = _repair_flat_hierarchy_from_relations(
        parent_by_key=parent_by_key,
        level_by_key=level_by_key,
        candidates=candidates,
        relations=relations or [],
        root_id=root_id,
        max_depth=max_depth,
    )
    if repaired_count:
        logger.info(
            "Repaired flat knowledge hierarchy for `%s` using extracted relations: repaired_edges=%d",
            file_path,
            repaired_count,
        )

    edges: list[tuple[str, str, dict[str, Any]]] = []
    for key in sorted(seen):
        candidate = candidates[key]
        child = HierarchyNode(
            entity_id=candidate.name,
            entity_name=candidate.name,
            entity_type=candidate.entity_type or "KnowledgePoint",
            description=candidate.description,
            source_id=_source_id_from_candidate(candidate),
            file_path=candidate.file_path,
            hierarchy_kind=HIERARCHY_KP_KIND,
            root_id=root_id,
            parent_id=parent_by_key[key],
            level=level_by_key[key],
            hierarchy_confidence=1.0,
        )
        edges.append(_make_hierarchy_edge(parent_by_key[key], child))

    if len(edges) >= 10:
        root_edge_count = sum(1 for src, _tgt, _data in edges if src == root_id)
        root_edge_ratio = root_edge_count / len(edges)
        if root_edge_ratio >= 0.8:
            logger.warning(
                "Knowledge hierarchy for `%s` is mostly flat: root_edges=%d, total_edges=%d",
                file_path,
                root_edge_count,
                len(edges),
            )

    return NormalizedHierarchy(root=root, nodes=[], edges=edges)


def _assignment_key(name: str) -> str:
    return normalize_knowledge_point_name(name)


def _would_assignment_create_cycle(
    child_key: str,
    parent_key: str,
    parent_by_key: dict[str, str],
) -> bool:
    current = parent_key
    seen = {child_key}
    while current:
        if current == child_key or current in seen:
            return True
        seen.add(current)
        current = parent_by_key.get(current, "")
    return False


def _assignment_depth(
    key: str,
    parent_by_key: dict[str, str],
) -> int:
    depth = 1
    current = key
    seen = {key}
    while current in parent_by_key:
        current = parent_by_key[current]
        if current in seen:
            return 9999
        seen.add(current)
        depth += 1
    return depth


def _has_compound_candidate_conflict(
    child_key: str,
    parent_key: str,
    candidate_by_key: dict[str, HierarchyCandidate],
) -> bool:
    child = candidate_by_key.get(child_key)
    parent = candidate_by_key.get(parent_key)
    if child is None or parent is None:
        return False
    child_name = _grounding_text_key(child.name)
    parent_name = _grounding_text_key(parent.name)
    if not child_name or not parent_name:
        return False
    compound_names = {
        _grounding_text_key(candidate.name)
        for key, candidate in candidate_by_key.items()
        if key not in {child_key, parent_key}
    }
    return f"{parent_name}{child_name}" in compound_names


def normalize_assignments_to_hierarchy(
    *,
    root_id: str,
    root_title: str,
    file_path: str,
    candidates: dict[str, HierarchyCandidate],
    assignments: list[ParentAssignment],
    max_depth: int,
    min_confidence: float,
) -> NormalizedHierarchy:
    root = HierarchyNode(
        entity_id=root_id,
        entity_name=root_title,
        entity_type="Resource",
        description=root_title,
        source_id="",
        file_path=file_path,
        hierarchy_kind=HIERARCHY_ROOT_KIND,
        root_id=root_id,
        parent_id=None,
        level=0,
    )

    candidate_by_name = {
        normalize_knowledge_point_name(candidate.name): key
        for key, candidate in candidates.items()
    }
    parent_by_key: dict[str, str] = {}

    ranked = sorted(assignments, key=lambda item: item.confidence, reverse=True)
    for assignment in ranked:
        if assignment.confidence < min_confidence or not assignment.parent:
            continue
        child_key = candidate_by_name.get(_assignment_key(assignment.child))
        parent_key = candidate_by_name.get(_assignment_key(assignment.parent))
        if not child_key or not parent_key or child_key == parent_key:
            continue
        if child_key in parent_by_key:
            continue
        if _would_assignment_create_cycle(child_key, parent_key, parent_by_key):
            continue
        if _has_compound_candidate_conflict(child_key, parent_key, candidates):
            continue
        parent_by_key[child_key] = parent_key
        if _assignment_depth(child_key, parent_by_key) > max_depth:
            parent_by_key.pop(child_key, None)

    edges: list[tuple[str, str, dict[str, Any]]] = []
    for key in sorted(candidates):
        candidate = candidates[key]
        parent_key = parent_by_key.get(key)
        if parent_key:
            parent_id = candidates[parent_key].name
            level = _assignment_depth(key, parent_by_key)
        else:
            parent_id = root_id
            level = 1
        child = HierarchyNode(
            entity_id=candidate.name,
            entity_name=candidate.name,
            entity_type=candidate.entity_type or "KnowledgePoint",
            description=candidate.description,
            source_id=_source_id_from_candidate(candidate),
            file_path=candidate.file_path,
            hierarchy_kind=HIERARCHY_KP_KIND,
            root_id=root_id,
            parent_id=parent_id,
            level=level,
            hierarchy_confidence=1.0,
        )
        edges.append(_make_hierarchy_edge(parent_id, child))

    return NormalizedHierarchy(root=root, nodes=[], edges=edges)


def _build_hierarchy_prompt(
    root_title: str,
    candidates: dict[str, HierarchyCandidate],
    max_depth: int,
    relations: list[dict[str, Any]] | None = None,
) -> str:
    payload = [
        {
            "name": candidate.name,
            "description": candidate.description,
            "source_chunk_ids": candidate.source_ids,
        }
        for candidate in candidates.values()
    ]
    relations = relations or []
    return (
        "Organize the following candidate knowledge points into a strict tree.\n"
        "Use only the provided candidate names. Do not invent new nodes.\n"
        "Each node may appear at most once and may have only one parent.\n"
        "Prefer parent-child edges supported by these relations when the "
        "relation means containment, prerequisite, taxonomy, component, "
        "process step, or conceptual scope.\n"
        "Avoid attaching many nodes directly to the root unless the provided "
        "relations and descriptions truly do not support deeper structure.\n"
        f"Maximum depth is {max_depth}.\n"
        "Return JSON only with shape: "
        '{"root_title": string, "nodes": [{"name": string, "description": string, '
        '"source_chunk_ids": [string], "children": [{"name": string, '
        '"description": string, "source_chunk_ids": [string], "children": []}]}.\n'
        "The `children` field must contain node objects, not plain strings.\n"
        f"Root title: {root_title}\n"
        f"Candidates JSON:\n{json.dumps(payload, ensure_ascii=False)}\n"
        f"Relations JSON:\n{json.dumps(relations, ensure_ascii=False)}"
    )


def _build_parent_assignment_prompt(
    *,
    root_title: str,
    children: list[HierarchyCandidate],
    parent_candidates: list[HierarchyCandidate],
    relations: list[dict[str, Any]],
    current_assignments: list[ParentAssignment],
) -> str:
    children_payload = [
        {
            "name": item.name,
            "description": item.description,
            "source_chunk_ids": item.source_ids,
            "entity_type": item.entity_type,
        }
        for item in children
    ]
    parents_payload = [
        {
            "name": item.name,
            "description": item.description,
            "source_chunk_ids": item.source_ids,
            "entity_type": item.entity_type,
        }
        for item in parent_candidates
    ]
    assigned_payload = [
        {"child": item.child, "parent": item.parent, "confidence": item.confidence}
        for item in current_assignments
        if item.parent
    ]
    return (
        "Assign each unresolved child knowledge point to the best parent.\n"
        "Use the document context, entity descriptions, and extracted relations to "
        "infer semantic hierarchy.\n"
        "Use only provided parent candidate names, or null when no parent is clear.\n"
        "Do not invent nodes. Do not assign a child to itself.\n"
        "Return JSON only with shape: "
        '{"assignments": [{"child": string, "parent": string|null, '
        '"confidence": number, "reason": string}]}.\n'
        f"Root title: {root_title}\n"
        f"Unresolved Children JSON:\n{json.dumps(children_payload, ensure_ascii=False)}\n"
        f"Parent Candidates JSON:\n{json.dumps(parents_payload, ensure_ascii=False)}\n"
        f"Relevant Relations JSON:\n{json.dumps(relations, ensure_ascii=False)}\n"
        f"Current Assignments JSON:\n{json.dumps(assigned_payload, ensure_ascii=False)}\n"
        "Parent Assignment JSON:"
    )


def _chunk_list(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        return [items]
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _select_parent_candidates(
    *,
    candidates: dict[str, HierarchyCandidate],
    children: list[HierarchyCandidate],
    assignments: list[ParentAssignment],
    limit: int,
) -> list[HierarchyCandidate]:
    child_keys = {child.key for child in children}
    assigned_parent_names = {
        normalize_knowledge_point_name(item.parent)
        for item in assignments
        if item.parent
    }
    batch_candidates = [item for item in children if item.key in candidates]
    external_candidates = [
        item for item in candidates.values() if item.key not in child_keys
    ]
    external_candidates.sort(
        key=lambda item: (
            -(item.key in assigned_parent_names),
            -len(item.source_ids),
            -len(item.description or ""),
            item.key,
        )
    )
    if limit <= 0:
        selected = [*batch_candidates, *external_candidates]
    else:
        selected = [*batch_candidates, *external_candidates[: max(0, limit - len(batch_candidates))]]
    seen: set[str] = set()
    unique: list[HierarchyCandidate] = []
    for item in selected:
        if item.key in seen:
            continue
        seen.add(item.key)
        unique.append(item)
    return unique


def _limit_hierarchy_candidates(
    candidates: dict[str, HierarchyCandidate],
    max_candidates: int,
) -> dict[str, HierarchyCandidate]:
    if max_candidates <= 0 or len(candidates) <= max_candidates:
        return candidates

    ranked = sorted(
        candidates.items(),
        key=lambda item: (
            -len(item[1].source_ids),
            -len(item[1].description or ""),
            item[0],
        ),
    )
    return dict(ranked[:max_candidates])


def _parse_hierarchy_json(raw: str) -> dict[str, Any]:
    parsed = json_repair.loads(str(raw or "").strip())
    if not isinstance(parsed, dict):
        raise ValueError("Hierarchy LLM output must be a JSON object")
    if not isinstance(parsed.get("nodes", []), list):
        raise ValueError("Hierarchy LLM output field `nodes` must be a list")
    return parsed


def parse_parent_assignments(raw: Any) -> list[ParentAssignment]:
    if isinstance(raw, str):
        parsed = json_repair.loads(raw)
    else:
        parsed = raw

    if isinstance(parsed, dict):
        rows = parsed.get("assignments", [])
    else:
        rows = parsed

    if not isinstance(rows, list):
        return []

    assignments: list[ParentAssignment] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        child = str(row.get("child") or "").strip()
        if not child:
            continue
        parent_raw = row.get("parent")
        parent = str(parent_raw).strip() if parent_raw not in (None, "") else None
        try:
            confidence = float(row.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        reason = str(row.get("reason") or "").strip()
        assignments.append(
            ParentAssignment(
                child=child,
                parent=parent,
                confidence=max(0.0, min(confidence, 1.0)),
                reason=reason,
            )
        )
    return assignments


def _resolve_hierarchy_llm_func(global_config: dict[str, Any]):
    explicit_func = global_config.get("hierarchy_llm_model_func")
    if explicit_func is not None:
        return explicit_func, "hierarchy_llm_model_func"

    role_llm_funcs = global_config.get("role_llm_funcs") or {}
    extract_func = role_llm_funcs.get("extract")
    if extract_func is not None:
        return extract_func, "role_llm_funcs.extract"

    return global_config.get("llm_model_func"), "llm_model_func"


async def build_resource_knowledge_hierarchy(
    doc_id: str,
    file_path: str,
    chunk_results: list[tuple[dict[str, list[dict]], dict]],
    global_config: dict[str, Any],
    llm_response_cache=None,
    grounding_text: str = "",
) -> NormalizedHierarchy | None:
    if not global_config.get("enable_knowledge_hierarchy", True):
        return None

    root_id = make_resource_root_id(doc_id)
    candidate_types = global_config.get(
        "hierarchy_candidate_entity_types",
        ["knowledgepoint", "concept", "content", "method"],
    )
    candidates = collect_hierarchy_candidates(
        chunk_results,
        candidate_entity_types=[str(item).lower() for item in candidate_types],
        file_path=file_path,
    )
    candidates = filter_hierarchy_candidates_by_grounding(
        candidates,
        grounding_text,
    )
    if not candidates:
        logger.info(
            "No hierarchy candidates found for `%s` (candidate_entity_types=%s)",
            file_path,
            candidate_types,
        )
        return None

    all_candidates = candidates
    max_depth = int(global_config.get("hierarchy_max_depth", 5))
    relations = collect_hierarchy_relations(chunk_results, all_candidates)
    root_title = file_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] or file_path
    llm_func, llm_source = _resolve_hierarchy_llm_func(global_config)
    if llm_func is None:
        logger.warning(
            "Hierarchy generation skipped for `%s`: no LLM function", file_path
        )
        return None

    logger.info(
        "Building knowledge hierarchy for `%s`: root_id=%s, candidates=%d, max_depth=%d, llm_source=%s",
        file_path,
        root_id,
        len(all_candidates),
        max_depth,
        llm_source,
    )

    batch_size = int(
        global_config.get(
            "hierarchy_batch_size",
            global_config.get("hierarchy_max_candidates", 80),
        )
    )
    max_rounds = int(global_config.get("hierarchy_max_rounds", 2))
    parent_candidate_limit = int(
        global_config.get("hierarchy_parent_candidate_limit", 40)
    )
    min_confidence = float(global_config.get("hierarchy_min_parent_confidence", 0.6))
    max_parallel_batches = int(global_config.get("hierarchy_max_parallel_batches", 3))

    assignments = assign_parents_from_relations(all_candidates, relations)
    assigned_children = {
        normalize_knowledge_point_name(item.child)
        for item in assignments
        if item.parent and item.confidence >= min_confidence
    }

    for _round in range(max(0, max_rounds)):
        assigned_count_before_round = len(assigned_children)
        unresolved = [
            candidate
            for key, candidate in all_candidates.items()
            if key not in assigned_children
        ]
        if not unresolved:
            break

        round_assignments = list(assignments)
        batch_semaphore = asyncio.Semaphore(max(1, max_parallel_batches))

        async def assign_batch(
            batch: list[HierarchyCandidate],
        ) -> list[ParentAssignment]:
            parent_candidates = _select_parent_candidates(
                candidates=all_candidates,
                children=batch,
                assignments=round_assignments,
                limit=parent_candidate_limit,
            )
            prompt = _build_parent_assignment_prompt(
                root_title=root_title,
                children=batch,
                parent_candidates=parent_candidates,
                relations=relations,
                current_assignments=round_assignments,
            )
            try:
                async with batch_semaphore:
                    if llm_response_cache is not None:
                        raw, _timestamp = await use_llm_func_with_cache(
                            prompt,
                            llm_func,
                            llm_response_cache=llm_response_cache,
                            cache_type="hierarchy",
                            response_format={"type": "json_object"},
                            llm_cache_identity=get_llm_cache_identity(
                                global_config, "extract"
                            ),
                        )
                    else:
                        raw = await llm_func(prompt)
                    return parse_parent_assignments(raw)
            except Exception as exc:
                logger.warning(
                    "Hierarchy parent assignment batch failed for `%s`: %s",
                    file_path,
                    exc,
                )
                return []

        batch_results = await asyncio.gather(
            *[assign_batch(batch) for batch in _chunk_list(unresolved, batch_size)]
        )

        for parsed_assignments in batch_results:
            assignments.extend(parsed_assignments)
            for item in parsed_assignments:
                if item.parent and item.confidence >= min_confidence:
                    assigned_children.add(normalize_knowledge_point_name(item.child))

        if len(assigned_children) == assigned_count_before_round:
            break

    return normalize_assignments_to_hierarchy(
        root_id=root_id,
        root_title=root_title,
        file_path=file_path,
        candidates=all_candidates,
        assignments=assignments,
        max_depth=max_depth,
        min_confidence=min_confidence,
    )


def build_flat_resource_knowledge_hierarchy(
    doc_id: str,
    file_path: str,
    chunk_results: list[tuple[dict[str, list[dict]], dict]],
    global_config: dict[str, Any],
) -> NormalizedHierarchy | None:
    candidate_types = global_config.get(
        "hierarchy_candidate_entity_types",
        [],
    )
    candidates = collect_hierarchy_candidates(
        chunk_results,
        candidate_entity_types=[str(item).lower() for item in candidate_types],
        file_path=file_path,
    )
    if not candidates:
        return None

    max_candidates = int(global_config.get("hierarchy_max_candidates", 80))
    candidates = _limit_hierarchy_candidates(candidates, max_candidates)
    root_id = make_resource_root_id(doc_id)
    root_title = file_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] or file_path
    return normalize_hierarchy_tree(
        root_id=root_id,
        root_title=root_title,
        file_path=file_path,
        candidates=candidates,
        llm_tree={"root_title": root_title, "nodes": []},
        max_depth=1,
    )


async def persist_resource_knowledge_hierarchy(
    hierarchy: NormalizedHierarchy,
    knowledge_graph_inst,
    entity_vdb,
) -> bool:
    root_id = hierarchy.root.entity_id
    endpoint_ids = {
        node_id
        for src, tgt, _data in hierarchy.edges
        for node_id in (src, tgt)
        if node_id != root_id
    }
    existing_endpoints = (
        await knowledge_graph_inst.has_nodes_batch(list(endpoint_ids))
        if endpoint_ids
        else set()
    )
    valid_edges = [
        (src, tgt, data)
        for src, tgt, data in hierarchy.edges
        if (src == root_id or src in existing_endpoints)
        and (tgt == root_id or tgt in existing_endpoints)
    ]
    if not valid_edges:
        logger.warning(
            "Knowledge hierarchy skipped: root_id=%s has no valid hierarchy edges",
            root_id,
        )
        return False

    existing_edges = await knowledge_graph_inst.get_all_edges()
    stale_edges = []
    for edge in existing_edges:
        if edge.get("edge_type") != HIERARCHY_EDGE_TYPE or edge.get("root_id") != root_id:
            continue
        source = edge.get("source") or edge.get("source_node_id") or edge.get("src_id")
        target = edge.get("target") or edge.get("target_node_id") or edge.get("tgt_id")
        if source and target:
            stale_edges.append((str(source), str(target)))
    if stale_edges:
        await knowledge_graph_inst.remove_edges(stale_edges)

    nodes = [(hierarchy.root.entity_id, hierarchy.root.to_graph_data())]
    await knowledge_graph_inst.upsert_nodes_batch(nodes)
    await knowledge_graph_inst.upsert_edges_batch(valid_edges)

    logger.info(
        "Persisted knowledge hierarchy: root_id=%s, nodes=%d, edges=%d",
        hierarchy.root.entity_id,
        len(hierarchy.nodes) + 1,
        len(valid_edges),
    )
    return True


async def build_chunk_results_from_resource_graph(
    knowledge_graph_inst,
    file_path: str,
    chunk_ids: list[str] | None = None,
) -> list[tuple[dict[str, list[dict]], dict]]:
    chunk_filter = {str(chunk_id) for chunk_id in (chunk_ids or []) if chunk_id}

    def _split_field(value: Any) -> list[str]:
        return split_string_by_multi_markers(str(value or ""), [GRAPH_FIELD_SEP])

    def _trim_doc_scoped_fields(data: dict[str, Any]) -> dict[str, Any] | None:
        source_ids = _split_field(data.get("source_id"))
        file_paths = _split_field(data.get("file_path"))
        descriptions = _split_field(data.get("description"))
        if not source_ids:
            if data.get("file_path") == file_path:
                return {**data, "file_path": file_path}
            return None

        keep_indexes = [
            idx
            for idx, source_id in enumerate(source_ids)
            if (not chunk_filter or source_id in chunk_filter)
            and (
                idx >= len(file_paths)
                or not file_paths[idx]
                or file_paths[idx] == file_path
            )
        ]
        if not keep_indexes:
            return None

        def pick(items: list[str]) -> list[str]:
            return [
                items[idx]
                for idx in keep_indexes
                if idx < len(items) and str(items[idx]).strip()
            ]

        trimmed = dict(data)
        trimmed["source_id"] = GRAPH_FIELD_SEP.join(pick(source_ids))
        trimmed["file_path"] = GRAPH_FIELD_SEP.join(pick(file_paths)) or file_path
        picked_descriptions = pick(descriptions)
        if picked_descriptions:
            trimmed["description"] = GRAPH_FIELD_SEP.join(picked_descriptions)
        return trimmed

    def belongs_to_doc(data: dict[str, Any]) -> bool:
        if _trim_doc_scoped_fields(data) is not None:
            return True
        return False

    nodes: dict[str, list[dict]] = {}
    for node in await knowledge_graph_inst.get_all_nodes():
        if node.get("hierarchy_kind"):
            continue
        entity_name = str(
            node.get("entity_name") or node.get("entity_id") or node.get("id") or ""
        ).strip()
        scoped_node = _trim_doc_scoped_fields(node)
        if not entity_name or scoped_node is None:
            continue
        nodes.setdefault(entity_name, []).append(
            {
                "entity_name": entity_name,
                "entity_type": scoped_node.get("entity_type", "concept"),
                "description": scoped_node.get("description", ""),
                "source_id": scoped_node.get("source_id", ""),
                "file_path": scoped_node.get("file_path") or file_path,
            }
        )

    edges: dict[tuple[str, str], list[dict]] = {}
    for edge in await knowledge_graph_inst.get_all_edges():
        if edge.get("edge_type") == HIERARCHY_EDGE_TYPE:
            continue
        scoped_edge = _trim_doc_scoped_fields(edge)
        if scoped_edge is None:
            continue
        src = str(
            edge.get("source") or edge.get("source_node_id") or edge.get("src_id") or ""
        ).strip()
        tgt = str(
            edge.get("target") or edge.get("target_node_id") or edge.get("tgt_id") or ""
        ).strip()
        if not src or not tgt:
            continue
        edges.setdefault((src, tgt), []).append(
            {
                "src_id": src,
                "tgt_id": tgt,
                "description": scoped_edge.get("description", ""),
                "keywords": scoped_edge.get("keywords", ""),
                "source_id": scoped_edge.get("source_id", ""),
                "file_path": scoped_edge.get("file_path") or file_path,
            }
        )

    return [(nodes, edges)]


def _is_legacy_hierarchy_duplicate_id(node_id: str) -> bool:
    return node_id.startswith("resource:") and ":kp:" in node_id


async def cleanup_legacy_hierarchy_duplicates(
    knowledge_graph_inst,
    entity_vdb=None,
) -> list[str]:
    nodes = await knowledge_graph_inst.get_all_nodes()
    edges = await knowledge_graph_inst.get_all_edges()
    hierarchy_roots_with_edges = {
        str(edge.get("root_id") or "")
        for edge in edges
        if edge.get("edge_type") == HIERARCHY_EDGE_TYPE and edge.get("root_id")
    }
    legacy_ids: list[str] = []
    for node in nodes:
        node_id = str(node.get("entity_id") or node.get("id") or "")
        if _is_legacy_hierarchy_duplicate_id(node_id):
            legacy_ids.append(node_id)
        elif (
            node.get("hierarchy_kind") == HIERARCHY_ROOT_KIND
            and node_id.startswith("resource:")
            and node_id not in hierarchy_roots_with_edges
        ):
            legacy_ids.append(node_id)

    for node_id in legacy_ids:
        await knowledge_graph_inst.delete_node(node_id)

    legacy_vdb_ids = [
        node_id for node_id in legacy_ids if _is_legacy_hierarchy_duplicate_id(node_id)
    ]
    if entity_vdb is not None and legacy_vdb_ids:
        await entity_vdb.delete(legacy_vdb_ids)

    if legacy_ids:
        logger.info(
            "Cleaned legacy hierarchy duplicate nodes: count=%d", len(legacy_ids)
        )
    return legacy_ids


async def _safe_get_node(graph, node_id: str) -> dict[str, Any] | None:
    node = await graph.get_node(node_id)
    if node is None:
        return None
    return {**node, "entity_id": node.get("entity_id") or node_id}


async def _safe_get_edge(graph, src: str, tgt: str) -> dict[str, Any] | None:
    edge = await graph.get_edge(src, tgt)
    if inspect.isawaitable(edge):
        edge = await edge
    if edge is None:
        return None
    return edge if isinstance(edge, dict) else None


async def _collect_children(
    graph, parent_id: str, root_id: str
) -> list[dict[str, Any]]:
    edges = await graph.get_node_edges(parent_id) or []
    children = []
    for src, tgt in edges:
        edge_data = await _safe_get_edge(graph, src, tgt)
        if not edge_data and src != parent_id:
            edge_data = await _safe_get_edge(graph, tgt, src)
        child_id = tgt if src == parent_id else src
        child = await _safe_get_node(graph, child_id)
        if not child:
            continue
        if edge_data:
            if (
                edge_data.get("edge_type") != HIERARCHY_EDGE_TYPE
                or edge_data.get("root_id") != root_id
                or edge_data.get("parent_id") != parent_id
            ):
                continue
            child = {
                **child,
                "root_id": root_id,
                "parent_id": parent_id,
                "file_path": child.get("file_path") or edge_data.get("file_path"),
                "source_id": child.get("source_id") or edge_data.get("source_id", ""),
            }
        elif not (
            child.get("root_id") == root_id and child.get("parent_id") == parent_id
        ):
            continue
        if child:
            children.append(child)
    return children


def _edge_endpoint(edge: dict[str, Any], *names: str) -> str:
    for name in names:
        value = edge.get(name)
        if value:
            return str(value)
    return ""


async def _collect_hierarchy_children_by_root(
    graph,
    root_id: str,
) -> dict[str, list[dict[str, Any]]]:
    try:
        edges = await graph.get_all_edges()
    except Exception as exc:
        logger.debug(
            "Failed to collect hierarchy edges by root `%s`: %s", root_id, exc
        )
        return {}

    children_by_parent: dict[str, list[dict[str, Any]]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for edge in edges:
        if (
            edge.get("edge_type") != HIERARCHY_EDGE_TYPE
            or str(edge.get("root_id") or "") != root_id
        ):
            continue
        parent_id = _edge_endpoint(edge, "parent_id", "source", "source_node_id", "src_id")
        child_id = _edge_endpoint(edge, "child_id", "target", "target_node_id", "tgt_id")
        if not parent_id or not child_id:
            continue
        pair = (parent_id, child_id)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        child = await _safe_get_node(graph, child_id)
        if not child:
            continue
        child = {
            **child,
            "root_id": root_id,
            "parent_id": parent_id,
            "file_path": child.get("file_path") or edge.get("file_path"),
            "source_id": child.get("source_id") or edge.get("source_id", ""),
        }
        children_by_parent.setdefault(parent_id, []).append(child)

    return children_by_parent


async def _collect_descendants(
    graph,
    parent_id: str,
    root_id: str,
    depth: int,
    limit_per_parent: int,
) -> list[dict[str, Any]]:
    if depth <= 0:
        return []

    children = await _collect_children(graph, parent_id, root_id)
    if limit_per_parent > 0:
        children = children[:limit_per_parent]

    descendants: list[dict[str, Any]] = []
    for child in children:
        descendants.append(child)
        child_id = child.get("entity_id")
        if child_id:
            descendants.extend(
                await _collect_descendants(
                    graph,
                    child_id,
                    root_id,
                    depth - 1,
                    limit_per_parent,
                )
            )
    return descendants


async def _collect_parent_chain(
    graph,
    node: dict[str, Any],
    depth: int,
) -> list[dict[str, Any]]:
    chain = []
    current = node
    for _ in range(depth):
        parent_id = current.get("parent_id")
        root_id = node.get("root_id")
        if not parent_id or parent_id == current.get("entity_id"):
            break
        parent = await _safe_get_node(graph, parent_id)
        if not parent or parent.get("root_id") != root_id:
            break
        chain.append(parent)
        current = parent
    return list(reversed(chain))


async def _resolve_hierarchy_node_context(
    graph,
    entity_id: str,
    expected_root_id: str | None = None,
) -> dict[str, Any] | None:
    node = await _safe_get_node(graph, entity_id)
    if not node:
        return None
    if node.get("root_id"):
        return node

    edges = await graph.get_node_edges(entity_id) or []
    for src, tgt in edges:
        edge_data = await _safe_get_edge(graph, src, tgt)
        if not edge_data and src != entity_id:
            edge_data = await _safe_get_edge(graph, tgt, src)
        if not edge_data or edge_data.get("edge_type") != HIERARCHY_EDGE_TYPE:
            continue
        root_id = edge_data.get("root_id")
        if expected_root_id and root_id != expected_root_id:
            continue
        if edge_data.get("child_id") == entity_id or tgt == entity_id:
            parent_id = edge_data.get("parent_id") or src
        else:
            parent_id = None
        return {
            **node,
            "root_id": root_id,
            "parent_id": parent_id,
            "file_path": node.get("file_path") or edge_data.get("file_path"),
            "source_id": node.get("source_id") or edge_data.get("source_id", ""),
        }
    return None


async def build_hierarchy_context(
    matched_entities: list[dict[str, Any]],
    knowledge_graph_inst,
    query_param,
    tokenizer=None,
) -> str:
    if not getattr(query_param, "enable_hierarchy_context", False):
        return ""

    sections: list[str] = []
    seen: set[str] = set()
    for entity in matched_entities:
        root_id = entity.get("root_id")
        entity_id = entity.get("entity_id") or entity.get("entity_name")
        if not root_id or not entity_id or entity_id in seen:
            if not entity_id or entity_id in seen:
                continue
        seen.add(entity_id)
        node = await _resolve_hierarchy_node_context(
            knowledge_graph_inst, entity_id, root_id
        )
        if not node or not node.get("root_id"):
            continue
        root_id = node.get("root_id")

        parent_chain = await _collect_parent_chain(
            knowledge_graph_inst,
            node,
            int(getattr(query_param, "hierarchy_parent_depth", 2)),
        )
        limit = int(getattr(query_param, "hierarchy_sibling_limit", 5))
        children = await _collect_descendants(
            knowledge_graph_inst,
            entity_id,
            root_id,
            int(getattr(query_param, "hierarchy_child_depth", 1)),
            limit,
        )
        siblings = []
        parent_id = node.get("parent_id")
        if parent_id:
            siblings = await _collect_children(knowledge_graph_inst, parent_id, root_id)
            siblings = [
                item
                for item in siblings
                if item.get("entity_id") != node.get("entity_id")
            ][:limit]

        path = " -> ".join(
            [item.get("entity_name", item.get("entity_id", "")) for item in parent_chain]
            + [node.get("entity_name", entity_id)]
        )
        sections.append(
            "\n".join(
                [
                    f"Resource: {root_id}",
                    f"Path: {path}",
                    "Children: "
                    + ", ".join(child.get("entity_name", "") for child in children),
                    "Sibling Concepts: "
                    + ", ".join(
                        sibling.get("entity_name", "") for sibling in siblings
                    ),
                    f"Source Chunks: {node.get('source_id', '')}",
                    f"File Path: {node.get('file_path', 'unknown_source')}",
                ]
            )
        )

    if not sections:
        return ""

    result = "-----Knowledge Hierarchy-----\n" + "\n\n".join(sections)
    max_tokens = int(getattr(query_param, "max_hierarchy_tokens", 2000))
    if tokenizer is not None and max_tokens > 0:
        tokens = tokenizer.encode(result)
        if len(tokens) > max_tokens:
            result = tokenizer.decode(tokens[:max_tokens])
    return result


async def get_hierarchy_tree(
    knowledge_graph_inst,
    root_id: str,
    max_depth: int = 20,
) -> dict[str, Any] | None:
    root = await _safe_get_node(knowledge_graph_inst, root_id)
    if not root or root.get("root_id") != root_id:
        return None
    children_by_parent = await _collect_hierarchy_children_by_root(
        knowledge_graph_inst, root_id
    )
    root_children = children_by_parent.get(root_id)
    if root_children is None:
        root_children = await _collect_children(knowledge_graph_inst, root_id, root_id)
    if not root_children:
        return None

    async def build(node: dict[str, Any], depth: int) -> dict[str, Any]:
        payload = dict(node)
        payload["children"] = []
        if depth >= max_depth:
            return payload
        if node["entity_id"] == root_id:
            children = root_children
        else:
            children = children_by_parent.get(node["entity_id"])
            if children is None:
                children = await _collect_children(
                    knowledge_graph_inst,
                    node["entity_id"],
                    root_id,
                )
        children = sorted(
            children,
            key=lambda item: (
                int(item.get("level", 0)),
                str(item.get("entity_name", item.get("entity_id", ""))),
            ),
        )
        for child in children:
            payload["children"].append(await build(child, depth + 1))
        return payload

    return await build(root, 0)


async def get_all_hierarchy_trees(
    knowledge_graph_inst,
    max_depth: int = 20,
) -> list[dict[str, Any]]:
    nodes = await knowledge_graph_inst.get_all_nodes()
    roots: list[tuple[str, str]] = []
    for node in nodes:
        if node.get("hierarchy_kind") != HIERARCHY_ROOT_KIND:
            continue
        root_id = str(node.get("entity_id") or node.get("id") or node.get("root_id") or "")
        if not root_id:
            continue
        roots.append((root_id, str(node.get("entity_name") or root_id)))

    trees: list[dict[str, Any]] = []
    for root_id, _name in sorted(roots, key=lambda item: (item[1], item[0])):
        tree = await get_hierarchy_tree(
            knowledge_graph_inst,
            root_id=root_id,
            max_depth=max_depth,
        )
        if tree is not None:
            trees.append(tree)
    return trees
