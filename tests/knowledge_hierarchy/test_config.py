from unittest.mock import AsyncMock

import numpy as np
import pytest

from lightrag import LightRAG, QueryParam
from lightrag.base import QueryResult
from lightrag.utils import EmbeddingFunc


async def fake_embed(texts: list[str]) -> np.ndarray:
    return np.zeros((len(texts), 4))


async def fake_llm(prompt: str, **kwargs) -> str:
    return "ok"


def make_rag(**kwargs):
    return LightRAG(
        working_dir="./tmp-hierarchy-config",
        llm_model_func=fake_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=4,
            max_token_size=16,
            func=fake_embed,
        ),
        **kwargs,
    )


def test_knowledge_hierarchy_is_enabled_by_default_without_type_filter():
    rag = make_rag()

    assert rag.enable_knowledge_hierarchy is True
    assert rag.hierarchy_max_depth == 5
    assert rag.hierarchy_batch_size == 80
    assert rag.hierarchy_max_rounds == 2
    assert rag.hierarchy_parent_candidate_limit == 40
    assert rag.hierarchy_min_parent_confidence == 0.6
    assert rag.hierarchy_max_parallel_batches == 3
    assert rag.hierarchy_candidate_entity_types == []


def test_knowledge_hierarchy_config_is_in_global_config():
    rag = make_rag(
        enable_knowledge_hierarchy=True,
        hierarchy_max_depth=3,
        hierarchy_batch_size=24,
        hierarchy_max_rounds=4,
        hierarchy_parent_candidate_limit=12,
        hierarchy_min_parent_confidence=0.7,
        hierarchy_max_parallel_batches=2,
        hierarchy_candidate_entity_types=["Concept", "Skill"],
    )

    config = rag._build_global_config()

    assert config["enable_knowledge_hierarchy"] is True
    assert config["hierarchy_max_depth"] == 3
    assert config["hierarchy_batch_size"] == 24
    assert config["hierarchy_max_rounds"] == 4
    assert config["hierarchy_parent_candidate_limit"] == 12
    assert config["hierarchy_min_parent_confidence"] == 0.7
    assert config["hierarchy_max_parallel_batches"] == 2
    assert config["hierarchy_candidate_entity_types"] == ["concept", "skill"]


def test_query_param_hierarchy_defaults():
    param = QueryParam()

    assert param.enable_hierarchy_context is False
    assert param.hierarchy_parent_depth == 2
    assert param.hierarchy_child_depth == 1
    assert param.hierarchy_sibling_limit == 5
    assert param.max_hierarchy_tokens == 2000


def test_entities_vector_storage_keeps_hierarchy_metadata():
    rag = make_rag()

    assert {
        "hierarchy_kind",
        "root_id",
        "parent_id",
        "level",
        "entity_type",
        "description",
    }.issubset(rag.entities_vdb.meta_fields)


@pytest.mark.asyncio
async def test_aquery_data_preserves_hierarchy_query_params(monkeypatch):
    captured = {}

    async def fake_kg_query(*args, **kwargs):
        captured["param"] = args[5]
        return QueryResult(
            content="",
            raw_data={
                "status": "success",
                "message": "ok",
                "data": {},
                "metadata": {},
            },
        )

    monkeypatch.setattr("lightrag.lightrag.kg_query", fake_kg_query)

    rag = object.__new__(LightRAG)
    rag.chunk_entity_relation_graph = object()
    rag.entities_vdb = object()
    rag.relationships_vdb = object()
    rag.text_chunks = object()
    rag.llm_response_cache = object()
    rag.chunks_vdb = object()
    rag._build_global_config = lambda: {}
    rag._query_done = AsyncMock()

    await rag.aquery_data(
        "tree",
        QueryParam(
            mode="local",
            enable_hierarchy_context=True,
            hierarchy_parent_depth=4,
            hierarchy_child_depth=3,
            hierarchy_sibling_limit=7,
            max_hierarchy_tokens=123,
        ),
    )

    param = captured["param"]
    assert param.enable_hierarchy_context is True
    assert param.hierarchy_parent_depth == 4
    assert param.hierarchy_child_depth == 3
    assert param.hierarchy_sibling_limit == 7
    assert param.max_hierarchy_tokens == 123
