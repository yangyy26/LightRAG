import importlib
import asyncio
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from lightrag.base import DocStatus
from lightrag.base import DocProcessingStatus
from lightrag.utils import compute_mdhash_id

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_dr = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

DocumentManager = _dr.DocumentManager

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _ensure_shared_storage_initialized():
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    shared_storage.initialize_share_data()
    yield
    if shared_storage._shared_dicts is not None:
        for key in list(shared_storage._shared_dicts.keys()):
            if key.endswith("pipeline_status") or key == "pipeline_status":
                ns = shared_storage._shared_dicts[key]
                if isinstance(ns, dict):
                    ns["busy"] = False
                    ns["scanning"] = False


class _MemoryKV:
    def __init__(self):
        self.rows = {}
        self.index_done_calls = 0

    async def upsert(self, rows):
        self.rows.update(rows)

    async def get_by_id(self, key):
        return self.rows.get(key)

    async def index_done_callback(self):
        self.index_done_calls += 1


class _MemoryDocStatus(_MemoryKV):
    async def get_doc_by_file_basename(self, basename):
        for doc_id, row in self.rows.items():
            if row.get("file_path") == basename:
                return doc_id, row
        return None

    async def delete(self, ids):
        for doc_id in ids:
            self.rows.pop(doc_id, None)


class _FakeRag:
    def __init__(self, workspace="default"):
        self.workspace = workspace
        self.full_docs = _MemoryKV()
        self.doc_status = _MemoryDocStatus()
        self.process_calls = 0

    async def apipeline_process_enqueue_documents(self):
        self.process_calls += 1

    async def apipeline_enqueue_error_documents(self, error_files, track_id):
        rows = {}
        now = datetime.now(timezone.utc).isoformat()
        for item in error_files:
            doc_id = compute_mdhash_id(item["file_path"], prefix="doc-")
            rows[doc_id] = {
                "status": DocStatus.FAILED,
                "content_summary": item.get("error_description", ""),
                "content_length": item.get("file_size", 0),
                "chunks_count": 0,
                "chunks_list": [],
                "created_at": now,
                "updated_at": now,
                "file_path": item["file_path"],
                "track_id": track_id,
                "error_msg": item.get("original_error", ""),
                "metadata": {},
            }
        await self.doc_status.upsert(rows)


@pytest.mark.parametrize(
    "filename",
    [
        "legacy.doc",
        "lecture.mp4",
        "lecture.avi",
        "lecture.mov",
        "lecture.wmv",
        "lecture.flv",
        "lecture.mkv",
        "voice.mp3",
        "voice.wav",
        "voice.m4a",
        "voice.aac",
        "voice.flac",
    ],
)
def test_document_manager_accepts_doc_and_media_extensions(tmp_path, filename):
    manager = DocumentManager(str(tmp_path))

    assert manager.is_supported_file(filename)


def test_media_suffix_detection():
    from lightrag.api.media_transcription import is_media_file

    assert is_media_file("lecture.mp4")
    assert is_media_file("voice.MP3")
    assert not is_media_file("report.docx")


def test_transcription_payload_uses_explicit_public_base():
    from lightrag.api.media_transcription import (
        TranscriptionConfig,
        build_transcription_payload,
    )

    cfg = TranscriptionConfig(
        endpoint="http://220.180.237.78:4021/transcribe/async",
        bucket_name="whisper",
        public_base_url="http://10.88.88.50:9621",
    )

    payload = build_transcription_payload(cfg, workspace="default", doc_id="doc-abc")

    assert payload == {
        "task_id": "doc-abc",
        "workspace": "default",
        "url": "http://10.88.88.50:9621/documents/media/files/default/doc-abc",
        "bucketName": "whisper",
        "callback": "http://10.88.88.50:9621/documents/media/transcribe_callback",
    }


def test_transcription_payload_can_use_external_media_url():
    from lightrag.api.media_transcription import (
        TranscriptionConfig,
        build_transcription_payload,
    )

    cfg = TranscriptionConfig(
        endpoint="http://220.180.237.78:4021/transcribe/async",
        bucket_name="whisper",
        public_base_url="http://10.88.88.50:9621",
    )

    payload = build_transcription_payload(
        cfg,
        workspace="default",
        doc_id="doc-abc",
        media_url="http://192.168.60.143:6692/whisper/lecture.mp4",
    )

    assert payload["url"] == "http://192.168.60.143:6692/whisper/lecture.mp4"
    assert payload["callback"] == (
        "http://10.88.88.50:9621/documents/media/transcribe_callback"
    )


def test_gfkd_upload_config_is_disabled_without_base_url(monkeypatch):
    from lightrag.api.media_transcription import load_gfkd_upload_config

    monkeypatch.delenv("GFKD_UPLOAD_BASE_URL", raising=False)

    assert load_gfkd_upload_config() is None


def test_upload_media_to_gfkd_runs_multipart_flow(monkeypatch, tmp_path):
    asyncio.run(_assert_upload_media_to_gfkd_runs_multipart_flow(monkeypatch, tmp_path))


async def _assert_upload_media_to_gfkd_runs_multipart_flow(monkeypatch, tmp_path):
    import httpx

    from lightrag.api.media_transcription import (
        GfkdUploadConfig,
        upload_media_to_gfkd,
    )

    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"abcdef")
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path == "/gfkb/resource/rbs/multipart/check":
            return httpx.Response(200, json={"code": 0})
        if request.url.path == "/gfkb/resource/rbs/multipart/init":
            return httpx.Response(
                200,
                json={
                    "code": 1,
                    "result": {
                        "uploadId": "upload-1",
                        "url": "http://objects/whisper/lecture.mp4",
                        "urlList": [
                            "http://upload.local/chunk-1",
                            "http://upload.local/chunk-2",
                        ],
                    },
                },
            )
        if request.url.host == "upload.local":
            return httpx.Response(200)
        if request.url.path == "/gfkb/resource/rbs/multipart/merge":
            return httpx.Response(
                200,
                json={
                    "code": 1,
                    "url": "http://objects/whisper/lecture.mp4",
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    original_async_client = httpx.AsyncClient

    class Client:
        def __init__(self, *args, **kwargs):
            self._client = original_async_client(transport=transport)

        async def __aenter__(self):
            return self._client

        async def __aexit__(self, *args):
            await self._client.aclose()

    monkeypatch.setattr("httpx.AsyncClient", Client)

    result = await upload_media_to_gfkd(
        GfkdUploadConfig(
            base_url="http://gfkd.local",
            token="token-1",
            domain="192.168.10.47",
            chunk_size=3,
            timeout=10,
        ),
        source,
        filename="lecture.mp4",
        content_type="video/mp4",
    )

    assert result == "http://objects/whisper/lecture.mp4"
    assert [request.method for request in requests] == [
        "GET",
        "POST",
        "PUT",
        "PUT",
        "POST",
    ]
    assert requests[0].headers["token"] == "token-1"
    assert requests[0].headers["domain"] == "192.168.10.47"
    assert requests[2].content == b"abc"
    assert requests[3].content == b"def"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"task_id": "doc-a", "text": "hello world"}, "hello world"),
        ({"task_id": "doc-a", "transcription": "hello\nworld"}, "hello\nworld"),
        (
            {"task_id": "doc-a", "segments": [{"text": "hello"}, {"text": "world"}]},
            "hello\nworld",
        ),
        (
            {
                "task_id": "doc-a",
                "subtitle": [{"content": "hello"}, {"content": "world"}],
            },
            "hello\nworld",
        ),
        ({"task_id": "doc-a", "data": {"text": "nested text"}}, "nested text"),
    ],
)
def test_extract_transcription_text_common_shapes(payload, expected):
    from lightrag.api.media_transcription import extract_transcription_text

    assert extract_transcription_text(payload) == expected


def test_extract_transcription_text_rejects_empty_payload():
    from lightrag.api.media_transcription import (
        TranscriptionPayloadError,
        extract_transcription_text,
    )

    with pytest.raises(TranscriptionPayloadError, match="No transcription text"):
        extract_transcription_text({"task_id": "doc-a", "segments": []})


def test_enqueue_media_transcription_creates_parsing_status(monkeypatch, tmp_path):
    asyncio.run(_assert_enqueue_media_transcription_creates_parsing_status(monkeypatch, tmp_path))


def test_pipeline_enqueue_file_routes_mp3_to_media_transcription(monkeypatch, tmp_path):
    asyncio.run(
        _assert_pipeline_enqueue_file_routes_mp3_to_media_transcription(
            monkeypatch, tmp_path
        )
    )


def test_scan_preserves_pending_media_transcription_status(monkeypatch, tmp_path):
    asyncio.run(
        _assert_scan_preserves_pending_media_transcription_status(monkeypatch, tmp_path)
    )


def test_pipeline_consistency_preserves_pending_media_transcription():
    asyncio.run(_assert_pipeline_consistency_preserves_pending_media_transcription())


async def _assert_pipeline_consistency_preserves_pending_media_transcription():
    from lightrag.pipeline import _PipelineMixin

    class PipelineProbe(_PipelineMixin):
        def __init__(self):
            self.full_docs = _MemoryKV()
            self.doc_status = _MemoryDocStatus()

    probe = PipelineProbe()
    doc_id = compute_mdhash_id("voice.mp3", prefix="doc-")
    status_doc = DocProcessingStatus(
        status=DocStatus.PARSING,
        content_summary="Media transcription pending",
        content_length=5,
        chunks_count=0,
        chunks_list=[],
        created_at="2026-06-24T00:00:00+00:00",
        updated_at="2026-06-24T00:00:00+00:00",
        file_path="voice.mp3",
        track_id="upload-media",
        metadata={
            "media_transcription_status": "pending",
            "media_source_file": "voice.mp3",
        },
    )
    await probe.doc_status.upsert({doc_id: status_doc.__dict__.copy()})
    pipeline_status = {"history_messages": []}
    pipeline_status_lock = asyncio.Lock()

    result = await probe._validate_and_fix_document_consistency(
        {doc_id: status_doc},
        pipeline_status,
        pipeline_status_lock,
    )

    assert result == {}
    assert await probe.doc_status.get_by_id(doc_id) is not None
    assert pipeline_status["history_messages"] == [
        "Preserving 1 pending media transcription document entries"
    ]


async def _assert_scan_preserves_pending_media_transcription_status(
    monkeypatch, tmp_path
):
    from lightrag.api.routers.document_routes import run_scanning_process

    media_path = tmp_path / "voice.mp3"
    media_path.write_bytes(b"audio")
    media_doc_id = compute_mdhash_id("voice.mp3", prefix="doc-")
    rag = _FakeRag()
    await rag.doc_status.upsert(
        {
            media_doc_id: {
                "status": DocStatus.PARSING,
                "content_summary": "Media transcription pending",
                "content_length": media_path.stat().st_size,
                "chunks_count": 0,
                "chunks_list": [],
                "created_at": "2026-06-24T00:00:00+00:00",
                "updated_at": "2026-06-24T00:00:00+00:00",
                "file_path": "voice.mp3",
                "track_id": "upload-media",
                "metadata": {
                    "media_transcription_status": "pending",
                    "media_source_file": "voice.mp3",
                },
            }
        }
    )

    async def fail_if_pipeline_index_files_called(*args, **kwargs):
        raise AssertionError("pending media transcription must not be re-enqueued")

    monkeypatch.setattr(
        _dr, "pipeline_index_files", fail_if_pipeline_index_files_called
    )

    await run_scanning_process(rag, DocumentManager(str(tmp_path)), "scan-track")

    status = await rag.doc_status.get_by_id(media_doc_id)
    assert status is not None
    assert status["status"] == DocStatus.PARSING
    assert status["metadata"]["media_transcription_status"] == "pending"
    assert rag.process_calls == 0
    assert media_path.exists()


async def _assert_pipeline_enqueue_file_routes_mp3_to_media_transcription(
    monkeypatch, tmp_path
):
    from lightrag.api.routers.document_routes import pipeline_enqueue_file

    calls = []

    async def fake_trigger(config, payload):
        calls.append(payload)

    monkeypatch.setenv(
        "TRANSCRIBE_ENDPOINT", "http://220.180.237.78:4021/transcribe/async"
    )
    monkeypatch.setenv("TRANSCRIBE_BUCKET_NAME", "whisper")
    monkeypatch.setenv("LIGHTRAG_PUBLIC_CALLBACK_BASE_URL", "http://10.88.88.50:9621")
    monkeypatch.delenv("GFKD_UPLOAD_BASE_URL", raising=False)
    monkeypatch.setattr(
        "lightrag.api.media_transcription.trigger_transcription", fake_trigger
    )
    monkeypatch.setattr(_dr, "_get_global_args", lambda: SimpleNamespace())

    rag = _FakeRag()
    source = tmp_path / "voice.mp3"
    source.write_bytes(b"audio")
    doc_id = compute_mdhash_id("voice.mp3", prefix="doc-")

    success, track_id = await pipeline_enqueue_file(rag, source, "scan-track")

    status = await rag.doc_status.get_by_id(doc_id)
    assert success is True
    assert track_id == "scan-track"
    assert status["status"] == DocStatus.PARSING
    assert status["file_path"] == "voice.mp3"
    assert status["metadata"]["media_transcription_status"] == "pending"
    assert calls[0]["task_id"] == doc_id
    assert source.exists()


async def _assert_enqueue_media_transcription_creates_parsing_status(
    monkeypatch, tmp_path
):
    from lightrag.api.routers.document_routes import enqueue_media_transcription

    calls = []

    async def fake_trigger(config, payload):
        calls.append((config, payload))

    monkeypatch.setenv(
        "TRANSCRIBE_ENDPOINT", "http://220.180.237.78:4021/transcribe/async"
    )
    monkeypatch.setenv("TRANSCRIBE_BUCKET_NAME", "whisper")
    monkeypatch.setenv("LIGHTRAG_PUBLIC_CALLBACK_BASE_URL", "http://10.88.88.50:9621")
    monkeypatch.delenv("GFKD_UPLOAD_BASE_URL", raising=False)
    monkeypatch.setattr(
        "lightrag.api.media_transcription.trigger_transcription", fake_trigger
    )

    rag = _FakeRag()
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"media")
    doc_id = compute_mdhash_id("lecture.mp4", prefix="doc-")

    await enqueue_media_transcription(
        rag=rag,
        file_path=source,
        canonical_file_path="lecture.mp4",
        doc_id=doc_id,
        track_id="upload-track",
        mime_type="video/mp4",
    )

    status = await rag.doc_status.get_by_id(doc_id)
    assert status["status"] == DocStatus.PARSING
    assert status["file_path"] == "lecture.mp4"
    assert status["track_id"] == "upload-track"
    assert status["metadata"]["media_transcription_status"] == "pending"
    assert status["metadata"]["media_source_file"] == "lecture.mp4"
    assert rag.doc_status.index_done_calls == 1
    assert calls[0][1]["task_id"] == doc_id
    assert calls[0][1]["workspace"] == "default"
    assert (
        calls[0][1]["url"]
        == f"http://10.88.88.50:9621/documents/media/files/default/{doc_id}"
    )


def test_enqueue_media_transcription_uses_gfkd_uploaded_url(monkeypatch, tmp_path):
    asyncio.run(
        _assert_enqueue_media_transcription_uses_gfkd_uploaded_url(monkeypatch, tmp_path)
    )


async def _assert_enqueue_media_transcription_uses_gfkd_uploaded_url(
    monkeypatch, tmp_path
):
    from lightrag.api.routers.document_routes import enqueue_media_transcription

    calls = []

    async def fake_trigger(config, payload):
        calls.append((config, payload))

    async def fake_upload(config, file_path, *, filename, content_type):
        calls.append(("upload", filename, content_type, file_path.read_bytes()))
        return "http://192.168.60.143:6692/whisper/lecture.mp4"

    monkeypatch.setenv(
        "TRANSCRIBE_ENDPOINT", "http://220.180.237.78:4021/transcribe/async"
    )
    monkeypatch.setenv("TRANSCRIBE_BUCKET_NAME", "whisper")
    monkeypatch.setenv("LIGHTRAG_PUBLIC_CALLBACK_BASE_URL", "http://10.88.88.50:9621")
    monkeypatch.setenv("GFKD_UPLOAD_BASE_URL", "http://127.0.0.1:8498")
    monkeypatch.setattr(
        "lightrag.api.media_transcription.upload_media_to_gfkd", fake_upload
    )
    monkeypatch.setattr(
        "lightrag.api.media_transcription.trigger_transcription", fake_trigger
    )

    rag = _FakeRag()
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"media")
    doc_id = compute_mdhash_id("lecture.mp4", prefix="doc-")

    await enqueue_media_transcription(
        rag=rag,
        file_path=source,
        canonical_file_path="lecture.mp4",
        doc_id=doc_id,
        track_id="upload-track",
        mime_type="video/mp4",
    )

    status = await rag.doc_status.get_by_id(doc_id)
    assert status["status"] == DocStatus.PARSING
    assert status["metadata"]["media_uploaded_url"] == (
        "http://192.168.60.143:6692/whisper/lecture.mp4"
    )
    assert calls[0] == ("upload", "lecture.mp4", "video/mp4", b"media")
    assert calls[1][1]["url"] == "http://192.168.60.143:6692/whisper/lecture.mp4"


def test_enqueue_media_transcription_failure_marks_failed(monkeypatch, tmp_path):
    asyncio.run(_assert_enqueue_media_transcription_failure_marks_failed(monkeypatch, tmp_path))


async def _assert_enqueue_media_transcription_failure_marks_failed(monkeypatch, tmp_path):
    from lightrag.api.routers.document_routes import enqueue_media_transcription

    async def fake_trigger(config, payload):
        raise RuntimeError("transcribe unavailable")

    monkeypatch.setenv(
        "TRANSCRIBE_ENDPOINT", "http://220.180.237.78:4021/transcribe/async"
    )
    monkeypatch.setenv("LIGHTRAG_PUBLIC_CALLBACK_BASE_URL", "http://10.88.88.50:9621")
    monkeypatch.delenv("GFKD_UPLOAD_BASE_URL", raising=False)
    monkeypatch.setattr(
        "lightrag.api.media_transcription.trigger_transcription", fake_trigger
    )

    rag = _FakeRag()
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"media")
    doc_id = compute_mdhash_id("lecture.mp4", prefix="doc-")

    await enqueue_media_transcription(
        rag=rag,
        file_path=source,
        canonical_file_path="lecture.mp4",
        doc_id=doc_id,
        track_id="upload-track",
        mime_type="video/mp4",
    )

    status = await rag.doc_status.get_by_id(doc_id)
    assert status["status"] == DocStatus.FAILED
    assert status["metadata"]["media_transcription_status"] == "failed"
    assert status["error_msg"] == "transcribe unavailable"
    assert rag.doc_status.index_done_calls == 2


def test_apply_media_transcription_callback_success_enqueues_text():
    asyncio.run(_assert_apply_media_transcription_callback_success_enqueues_text())


async def _assert_apply_media_transcription_callback_success_enqueues_text():
    from lightrag.api.routers.document_routes import apply_media_transcription_callback

    rag = _FakeRag()
    doc_id = compute_mdhash_id("lecture.mp4", prefix="doc-")
    await rag.doc_status.upsert(
        {
            doc_id: {
                "status": DocStatus.PARSING,
                "content_summary": "Media transcription pending",
                "content_length": 5,
                "chunks_count": 0,
                "chunks_list": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "file_path": "lecture.mp4",
                "track_id": "upload-track",
                "metadata": {"media_transcription_status": "pending"},
            }
        }
    )

    await apply_media_transcription_callback(
        rag,
        {"task_id": doc_id, "segments": [{"text": "hello"}, {"text": "world"}]},
    )

    full_doc = await rag.full_docs.get_by_id(doc_id)
    status = await rag.doc_status.get_by_id(doc_id)
    assert full_doc["content"] == "hello\nworld"
    assert full_doc["file_path"] == "lecture.mp4"
    assert status["status"] == DocStatus.PENDING
    assert status["metadata"]["media_transcription_status"] == "completed"
    assert rag.doc_status.index_done_calls == 1
    assert rag.process_calls == 1


def test_apply_media_transcription_callback_failure_marks_failed():
    asyncio.run(_assert_apply_media_transcription_callback_failure_marks_failed())


async def _assert_apply_media_transcription_callback_failure_marks_failed():
    from lightrag.api.routers.document_routes import apply_media_transcription_callback

    rag = _FakeRag()
    doc_id = compute_mdhash_id("lecture.mp4", prefix="doc-")
    await rag.doc_status.upsert(
        {
            doc_id: {
                "status": DocStatus.PARSING,
                "content_summary": "Media transcription pending",
                "content_length": 5,
                "chunks_count": 0,
                "chunks_list": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "file_path": "lecture.mp4",
                "track_id": "upload-track",
                "metadata": {"media_transcription_status": "pending"},
            }
        }
    )

    await apply_media_transcription_callback(
        rag,
        {"task_id": doc_id, "status": "failed", "error": "asr failed"},
    )

    status = await rag.doc_status.get_by_id(doc_id)
    assert status["status"] == DocStatus.FAILED
    assert status["error_msg"] == "asr failed"
    assert status["metadata"]["media_transcription_status"] == "failed"
    assert rag.doc_status.index_done_calls == 1
    assert rag.process_calls == 0


def test_pipeline_enqueue_doc_without_converter_records_clear_failure(
    monkeypatch, tmp_path
):
    asyncio.run(
        _assert_pipeline_enqueue_doc_without_converter_records_clear_failure(
            monkeypatch, tmp_path
        )
    )


async def _assert_pipeline_enqueue_doc_without_converter_records_clear_failure(
    monkeypatch, tmp_path
):
    from lightrag.api.routers.document_routes import pipeline_enqueue_file

    monkeypatch.setattr(
        "lightrag.api.routers.document_routes._get_global_args",
        lambda: SimpleNamespace(),
    )
    monkeypatch.delenv("DOC_CONVERT_ENDPOINT", raising=False)
    monkeypatch.setenv("LIGHTRAG_PARSER", "*:legacy-F")

    rag = _FakeRag()
    source = tmp_path / "legacy.doc"
    source.write_bytes(b"binary-doc")

    success, track_id = await pipeline_enqueue_file(rag, source, track_id="upload-doc")

    assert success is False
    assert track_id == "upload-doc"
    failed_rows = list(rag.doc_status.rows.values())
    assert failed_rows
    assert failed_rows[0]["status"] == DocStatus.FAILED
    assert "DOC conversion is not configured" in failed_rows[0]["error_msg"]
