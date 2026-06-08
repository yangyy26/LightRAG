from types import SimpleNamespace

import numpy as np
import pytest

import lightrag.llm.openai as openai_module


@pytest.mark.asyncio
async def test_openai_embed_omits_encoding_format_when_configured(monkeypatch):
    monkeypatch.setattr(openai_module, "EMBEDDING_ENCODING_FORMAT", None)
    captured = {}

    class FakeEmbeddings:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.1, 0.2])],
                usage=SimpleNamespace(prompt_tokens=1, total_tokens=1),
            )

    class FakeClient:
        def __init__(self):
            self.embeddings = FakeEmbeddings()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    monkeypatch.setattr(
        openai_module,
        "create_openai_async_client",
        lambda **_kwargs: FakeClient(),
    )

    result = await openai_module.openai_embed.func(
        ["hello"],
        model="fake-embedding",
        api_key="key",
        base_url="https://example.test/v1",
    )

    assert isinstance(result, np.ndarray)
    assert "encoding_format" not in captured
