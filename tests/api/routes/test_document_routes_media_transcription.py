import importlib
import asyncio
import json
import sys
from datetime import datetime, timezone
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.base import DocStatus
from lightrag.base import DocProcessingStatus
from lightrag.utils import compute_mdhash_id

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_dr = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

DocumentManager = _dr.DocumentManager
WorkspaceContext = _dr.WorkspaceContext
create_document_routes = _dr.create_document_routes

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


def test_transcription_payload_requires_uploaded_media_url():
    from lightrag.api.media_transcription import (
        TranscriptionConfig,
        TranscriptionPayloadError,
        build_transcription_payload,
    )

    cfg = TranscriptionConfig(
        endpoint="http://220.180.237.78:4021/transcribe/async",
        bucket_name="whisper",
        public_base_url="http://10.88.88.50:9621",
    )

    with pytest.raises(TranscriptionPayloadError, match="uploaded media URL"):
        build_transcription_payload(cfg, workspace="default", doc_id="doc-abc")


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
        media_url="http://objects.local/whisper/lecture.mp4",
    )

    assert payload["url"] == "http://objects.local/whisper/lecture.mp4"
    assert "bucket_name" not in payload
    assert payload["callback"] == (
        "http://10.88.88.50:9621/documents/media/transcribe_callback"
        "?workspace=default"
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"workspace": "default"}, "default"),
        ({"workspace_id": "test"}, "test"),
        ({"data": {"workspace": "nested"}}, "nested"),
        ({"result": {"workspace_id": "result_ws"}}, "result_ws"),
        ({"task_id": "doc-a"}, None),
    ],
)
def test_callback_workspace_id_common_shapes(payload, expected):
    from lightrag.api.routers.document_routes import get_callback_workspace_id

    assert get_callback_workspace_id(payload) == expected


def test_summarize_media_callback_payload_does_not_log_text():
    from lightrag.api.routers.document_routes import summarize_media_callback_payload

    summary = summarize_media_callback_payload(
        {
            "task_id": "doc-a",
            "text": "secret transcript",
            "data": {"segments": [{"text": "hidden"}]},
        }
    )

    assert summary == {
        "top_level_keys": ["data", "task_id", "text"],
        "data_keys": ["segments"],
        "top.text_length": 17,
        "data.segments_count": 1,
    }
    assert "secret transcript" not in str(summary)
    assert "hidden" not in str(summary)


def test_media_resource_config_is_disabled_without_base_url(monkeypatch):
    from lightrag.api.media_transcription import load_media_resource_config

    monkeypatch.delenv("MEDIA_RESOURCE_BASE_URL", raising=False)

    assert load_media_resource_config() is None


def test_media_resource_config_reads_paths_and_download_origin(monkeypatch):
    from lightrag.api.media_transcription import load_media_resource_config

    monkeypatch.setenv("MEDIA_RESOURCE_BASE_URL", "http://media.local")
    monkeypatch.setenv("MEDIA_RESOURCE_MULTIPART_CHECK_PATH", "custom/check")
    monkeypatch.setenv("MEDIA_RESOURCE_MULTIPART_INIT_PATH", "/custom/init")
    monkeypatch.setenv("MEDIA_RESOURCE_MULTIPART_MERGE_PATH", "/custom/merge")
    monkeypatch.setenv("MEDIA_RESOURCE_PREVIEW_PATH", "/custom/preview")
    monkeypatch.setenv("MEDIA_RESOURCE_DOWNLOAD_ORIGIN", "http://public-object:19000")

    config = load_media_resource_config()

    assert config.multipart_check_path == "/custom/check"
    assert config.multipart_init_path == "/custom/init"
    assert config.multipart_merge_path == "/custom/merge"
    assert config.preview_path == "/custom/preview"
    assert config.download_origin == "http://public-object:19000"


def test_media_resource_config_defaults_to_multipart_upload_mode(monkeypatch):
    from lightrag.api.media_transcription import load_media_resource_config

    monkeypatch.setenv("MEDIA_RESOURCE_BASE_URL", "http://media.local")
    monkeypatch.delenv("MEDIA_RESOURCE_UPLOAD_MODE", raising=False)

    config = load_media_resource_config()

    assert config.upload_mode == "multipart"
    assert config.oss_upload_info_path == "/gatewayApi/resource/upload/getResUploadInfo"
    assert config.oss_callback_path == "/gatewayApi/resource/upload/dfsCallback"


def test_media_resource_config_reads_oss_upload_mode_and_paths(monkeypatch):
    from lightrag.api.media_transcription import load_media_resource_config

    monkeypatch.setenv("MEDIA_RESOURCE_BASE_URL", "http://media.local")
    monkeypatch.setenv("MEDIA_RESOURCE_UPLOAD_MODE", "OSS")
    monkeypatch.setenv("MEDIA_RESOURCE_OSS_UPLOAD_INFO_PATH", "custom/getInfo")
    monkeypatch.setenv("MEDIA_RESOURCE_OSS_CALLBACK_PATH", "/custom/callback")

    config = load_media_resource_config()

    assert config.upload_mode == "oss"
    assert config.oss_upload_info_path == "/custom/getInfo"
    assert config.oss_callback_path == "/custom/callback"


def test_media_resource_config_rejects_invalid_upload_mode(monkeypatch):
    from lightrag.api.media_transcription import (
        TranscriptionConfigError,
        load_media_resource_config,
    )

    monkeypatch.setenv("MEDIA_RESOURCE_BASE_URL", "http://media.local")
    monkeypatch.setenv("MEDIA_RESOURCE_UPLOAD_MODE", "ftp")

    with pytest.raises(TranscriptionConfigError):
        load_media_resource_config()


def test_upload_media_to_resource_service_runs_oss_flow(monkeypatch, tmp_path):
    asyncio.run(
        _assert_upload_media_to_resource_service_runs_oss_flow(monkeypatch, tmp_path)
    )


def test_upload_media_to_resource_service_runs_multipart_flow(monkeypatch, tmp_path):
    asyncio.run(_assert_upload_media_to_resource_service_runs_multipart_flow(monkeypatch, tmp_path))


def test_upload_media_to_resource_service_infers_mp3_content_type(monkeypatch, tmp_path):
    asyncio.run(
        _assert_upload_media_to_resource_service_infers_mp3_content_type(
            monkeypatch, tmp_path
        )
    )


def test_upload_media_to_resource_service_reuploads_fast_check_url_with_wrong_suffix(
    monkeypatch, tmp_path
):
    asyncio.run(
        _assert_upload_media_to_resource_service_reuploads_fast_check_url_with_wrong_suffix(
            monkeypatch, tmp_path
        )
    )


async def _assert_upload_media_to_resource_service_runs_multipart_flow(monkeypatch, tmp_path):
    import httpx

    from lightrag.api.media_transcription import (
        MediaResourceConfig,
        upload_media_to_resource_service,
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

    result = await upload_media_to_resource_service(
        MediaResourceConfig(
            base_url="http://media.local",
            token="token-1",
            domain="media-domain",
            chunk_size=3,
            timeout=10,
            multipart_check_path="/gfkb/resource/rbs/multipart/check",
            multipart_init_path="/gfkb/resource/rbs/multipart/init",
            multipart_merge_path="/gfkb/resource/rbs/multipart/merge",
            preview_path="/gfkb/resource/preview/getMeterialPreviewByUrl",
            download_origin="",
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
    assert requests[0].headers["domain"] == "media-domain"
    assert requests[2].content == b"abc"
    assert requests[3].content == b"def"


async def _assert_upload_media_to_resource_service_infers_mp3_content_type(
    monkeypatch, tmp_path
):
    import httpx

    from lightrag.api.media_transcription import (
        MediaResourceConfig,
        upload_media_to_resource_service,
    )

    source = tmp_path / "voice.mp3"
    source.write_bytes(b"abcdef")
    init_payloads = []
    upload_content_types = []

    def handler(request):
        if request.url.path == "/check":
            return httpx.Response(200, json={"code": 0})
        if request.url.path == "/init":
            init_payloads.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                200,
                json={
                    "code": 1,
                    "result": {
                        "uploadId": "upload-1",
                        "url": "http://objects/whisper/voice.mp3",
                        "urlList": ["http://upload.local/chunk-1"],
                    },
                },
            )
        if request.url.host == "upload.local":
            upload_content_types.append(request.headers["content-type"])
            return httpx.Response(200)
        if request.url.path == "/merge":
            return httpx.Response(
                200,
                json={"code": 1, "url": "http://objects/whisper/voice.mp3"},
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

    await upload_media_to_resource_service(
        MediaResourceConfig(
            base_url="http://media.local",
            token="",
            domain="",
            chunk_size=10,
            timeout=10,
            multipart_check_path="/check",
            multipart_init_path="/init",
            multipart_merge_path="/merge",
            preview_path="/preview",
            download_origin="",
        ),
        source,
        filename="voice.mp3",
        content_type=None,
    )

    assert init_payloads[0]["fileType"] == "audio"
    assert init_payloads[0]["contentType"] == "audio/mpeg"
    assert upload_content_types == ["audio/mpeg"]


async def _assert_upload_media_to_resource_service_reuploads_fast_check_url_with_wrong_suffix(
    monkeypatch, tmp_path
):
    import httpx

    from lightrag.api.media_transcription import (
        MediaResourceConfig,
        upload_media_to_resource_service,
    )

    source = tmp_path / "voice.mp3"
    source.write_bytes(b"abcdef")
    requested_paths = []

    def handler(request):
        requested_paths.append(request.url.path)
        if request.url.path == "/check":
            return httpx.Response(
                200,
                json={"code": 200, "result": {"url": "http://objects/whisper/voice"}},
            )
        if request.url.path == "/init":
            return httpx.Response(
                200,
                json={
                    "code": 1,
                    "result": {
                        "uploadId": "upload-1",
                        "url": "http://objects/whisper/voice.mp3",
                        "urlList": ["http://upload.local/chunk-1"],
                    },
                },
            )
        if request.url.host == "upload.local":
            return httpx.Response(200)
        if request.url.path == "/merge":
            return httpx.Response(
                200,
                json={"code": 1, "url": "http://objects/whisper/voice.mp3"},
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

    result = await upload_media_to_resource_service(
        MediaResourceConfig(
            base_url="http://media.local",
            token="",
            domain="",
            chunk_size=10,
            timeout=10,
            multipart_check_path="/check",
            multipart_init_path="/init",
            multipart_merge_path="/merge",
            preview_path="/preview",
            download_origin="",
        ),
        source,
        filename="voice.mp3",
        content_type=None,
    )

    assert result == "http://objects/whisper/voice.mp3"
    assert requested_paths == ["/check", "/init", "/chunk-1", "/merge"]


async def _assert_upload_media_to_resource_service_runs_oss_flow(monkeypatch, tmp_path):
    import httpx

    from lightrag.api.media_transcription import (
        MediaResourceConfig,
        upload_media_to_resource_service,
    )

    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"abcdef")
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path == "/gatewayApi/resource/upload/getResUploadInfo":
            return httpx.Response(
                200,
                json={
                    "code": 200,
                    "msg": "操作成功",
                    "result": {
                        "host": "https://gj-doc.oss-cn-hangzhou.aliyuncs.com",
                        "signature": {
                            "ossAccessKeyId": "ak-1",
                            "signature": "sig-1",
                            "callbackUrl": "https://public.example.com/gatewayApi/resource/upload/dfsCallback?bucket=gj-doc&strategy=oss",
                            "strategy": "2",
                            "key": "doc_mk/2026/07/21/tp@ABC.mp4",
                            "policy": "policy-1",
                        },
                    },
                },
            )
        if request.url.host == "gj-doc.oss-cn-hangzhou.aliyuncs.com":
            return httpx.Response(200, headers={"ETag": '"ETAG-1"'})
        if request.url.path == "/gatewayApi/resource/upload/dfsCallback":
            return httpx.Response(
                200,
                json={
                    "msg": "Success",
                    "bucket": "gj-doc",
                    "code": 1,
                    "size": 6,
                    "strategy": "oss",
                    "url": "doc_mk/2026/07/21/tp@ABC.mp4",
                    "key": "doc_mk/2026/07/21/tp@ABC.mp4",
                    "md5": "ABC",
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

    result = await upload_media_to_resource_service(
        MediaResourceConfig(
            base_url="http://media.local",
            token="token-1",
            domain="media-domain",
            chunk_size=1024,
            timeout=10,
            multipart_check_path="/check",
            multipart_init_path="/init",
            multipart_merge_path="/merge",
            preview_path="/preview",
            download_origin="",
            upload_mode="oss",
            oss_upload_info_path="/gatewayApi/resource/upload/getResUploadInfo",
            oss_callback_path="/gatewayApi/resource/upload/dfsCallback",
        ),
        source,
        filename="lecture.mp4",
        content_type="video/mp4",
    )

    assert result == "doc_mk/2026/07/21/tp@ABC.mp4"
    assert [request.method for request in requests] == ["POST", "POST", "POST"]

    # Interface 1: form-urlencoded getResUploadInfo
    info_request = requests[0]
    assert info_request.headers["token"] == "token-1"
    assert info_request.headers["domain"] == "media-domain"
    assert (
        info_request.headers["content-type"] == "application/x-www-form-urlencoded"
    )
    info_fields = dict(_parse_urlencoded(info_request.content.decode("utf-8")))
    assert info_fields["fileName"] == "lecture.mp4"
    assert info_fields["size"] == "6"

    # Interface 2: multipart POST to the OSS host
    oss_request = requests[1]
    assert oss_request.url.host == "gj-doc.oss-cn-hangzhou.aliyuncs.com"
    assert oss_request.headers["content-type"].startswith("multipart/form-data")
    body = oss_request.content.decode("utf-8", errors="ignore")
    assert 'name="OSSAccessKeyId"' in body
    assert "ak-1" in body
    assert "sig-1" in body
    assert "doc_mk/2026/07/21/tp@ABC.mp4" in body
    assert 'name="success_action_status"' in body

    # Interface 3: dfsCallback, origin rewritten to base_url with etag/key merged
    callback_request = requests[2]
    assert callback_request.url.host == "media.local"
    assert callback_request.url.path == "/gatewayApi/resource/upload/dfsCallback"
    callback_params = dict(callback_request.url.params)
    assert callback_params["bucket"] == "gj-doc"
    assert callback_params["strategy"] == "oss"
    assert callback_params["etag"] == '"ETAG-1"'
    assert callback_params["key"] == "doc_mk/2026/07/21/tp@ABC.mp4"


def test_upload_media_to_resource_service_runs_oss_flow_existing_file(monkeypatch, tmp_path):
    asyncio.run(
        _assert_upload_media_to_resource_service_runs_oss_flow_existing_file(
            monkeypatch, tmp_path
        )
    )


async def _assert_upload_media_to_resource_service_runs_oss_flow_existing_file(
    monkeypatch, tmp_path
):
    import httpx

    from lightrag.api.media_transcription import (
        MediaResourceConfig,
        upload_media_to_resource_service,
    )

    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"abcdef")
    requests = []

    def handler(request):
        requests.append(request)
        # The file already exists (matched by md5): the endpoint returns the
        # existing resource metadata without a presigned host/signature.
        if request.url.path == "/gatewayApi/resource/upload/getResUploadInfo":
            return httpx.Response(
                200,
                json={
                    "code": 200,
                    "msg": "操作成功",
                    "result": {
                        "bucket": None,
                        "strategy": None,
                        "md5": "ABC",
                        "fileName": "lecture.mp4",
                        "size": 6,
                        "duration": 70.0,
                        "url": "doc_mk/2025/04/11/2969A158/tp@ABC.mp4",
                        "category": "video",
                        "transferState": 2,
                    },
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

    result = await upload_media_to_resource_service(
        MediaResourceConfig(
            base_url="http://media.local",
            token="token-1",
            domain="media-domain",
            chunk_size=1024,
            timeout=10,
            multipart_check_path="/check",
            multipart_init_path="/init",
            multipart_merge_path="/merge",
            preview_path="/preview",
            download_origin="",
            upload_mode="oss",
            oss_upload_info_path="/gatewayApi/resource/upload/getResUploadInfo",
            oss_callback_path="/gatewayApi/resource/upload/dfsCallback",
        ),
        source,
        filename="lecture.mp4",
        content_type="video/mp4",
    )

    # Reuses the existing object key, no OSS POST / dfsCallback performed.
    assert result == "doc_mk/2025/04/11/2969A158/tp@ABC.mp4"
    assert [request.method for request in requests] == ["POST"]
    assert requests[0].url.path == "/gatewayApi/resource/upload/getResUploadInfo"


def _parse_urlencoded(raw: str):
    from urllib.parse import parse_qsl

    return parse_qsl(raw, keep_blank_values=True)


def test_rewrite_media_download_url_replaces_any_http_origin_with_configured_origin():
    from lightrag.api.media_transcription import MediaResourceConfig, rewrite_media_download_url

    config = MediaResourceConfig(
        base_url="http://media.local",
        token="",
        domain="",
        chunk_size=3,
        timeout=10,
        multipart_check_path="/check",
        multipart_init_path="/init",
        multipart_merge_path="/merge",
        preview_path="/preview",
        download_origin="http://public-object:19000",
    )

    assert rewrite_media_download_url(
        "http://internal-object:19000/whisper/lecture.json?token=1",
        config,
    ) == "http://public-object:19000/whisper/lecture.json?token=1"


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


@pytest.mark.parametrize(
    ("media_url", "expected"),
    [
        ("http://host/whisper/voice.mp3", "http://host/whisper/voice.json"),
        (
            "http://host/whisper/voice.mp3?token=1",
            "http://host/whisper/voice.json?token=1",
        ),
        ("http://host/whisper/voice", "http://host/whisper/voice.json"),
    ],
)
def test_media_url_to_transcription_json_url(media_url, expected):
    from lightrag.api.media_transcription import media_url_to_transcription_json_url

    assert media_url_to_transcription_json_url(media_url) == expected


def test_fetch_transcription_text_from_media_preview_rewrites_down_url(monkeypatch):
    asyncio.run(_assert_fetch_transcription_text_from_media_preview_rewrites_down_url(monkeypatch))


def test_fetch_transcription_text_from_media_preview_preserves_encoded_down_url(
    monkeypatch,
):
    asyncio.run(
        _assert_fetch_transcription_text_from_media_preview_preserves_encoded_down_url(
            monkeypatch
        )
    )


def test_fetch_transcription_text_from_media_preview_reports_download_error_body(
    monkeypatch,
):
    asyncio.run(
        _assert_fetch_transcription_text_from_media_preview_reports_download_error_body(
            monkeypatch
        )
    )


def test_fetch_transcription_text_from_media_preview_retries_unsigned_on_signature_mismatch(
    monkeypatch,
):
    asyncio.run(
        _assert_fetch_transcription_text_from_media_preview_retries_unsigned_on_signature_mismatch(
            monkeypatch
        )
    )


async def _assert_fetch_transcription_text_from_media_preview_rewrites_down_url(
    monkeypatch,
):
    import httpx

    from lightrag.api.media_transcription import (
        MediaResourceConfig,
        fetch_transcription_text_from_media_preview,
    )

    requested_urls = []

    def handler(request):
        requested_urls.append(str(request.url))
        if request.url.path == "/custom/preview":
            assert request.url.params["url"] == "http://objects/whisper/lecture.json"
            return httpx.Response(
                200,
                json={
                    "code": 200,
                    "result": {
                        "downUrl": (
                            "http://internal-object:19000/whisper/lecture.json"
                        )
                    },
                },
            )
        if str(request.url) == "http://public-object:19000/whisper/lecture.json":
            return httpx.Response(200, json={"text": "hello rewritten"})
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

    text, json_url = await fetch_transcription_text_from_media_preview(
        MediaResourceConfig(
            base_url="http://media.local",
            token="",
            domain="",
            chunk_size=3,
            timeout=10,
            multipart_check_path="/check",
            multipart_init_path="/init",
            multipart_merge_path="/merge",
            preview_path="/custom/preview",
            download_origin="http://public-object:19000",
        ),
        "http://objects/whisper/lecture.mp4",
    )

    assert text == "hello rewritten"
    assert json_url == "http://objects/whisper/lecture.json"
    assert requested_urls == [
        (
            "http://media.local/custom/preview?"
            "url=http%3A%2F%2Fobjects%2Fwhisper%2Flecture.json"
        ),
        "http://public-object:19000/whisper/lecture.json",
    ]


async def _assert_fetch_transcription_text_from_media_preview_preserves_encoded_down_url(
    monkeypatch,
):
    import httpx

    from lightrag.api.media_transcription import (
        MediaResourceConfig,
        fetch_transcription_text_from_media_preview,
    )

    requested_urls = []
    encoded_down_url = (
        "http://internal-object:19000/gfkd/ss/2026/06/24/"
        "tp%403E7A857311302350CE3A8C40F2864F48.json"
        "?X-Amz-SignedHeaders=host&X-Amz-Signature=abc"
    )
    expected_rewritten_url = (
        "http://public-object:19000/gfkd/ss/2026/06/24/"
        "tp%403E7A857311302350CE3A8C40F2864F48.json"
        "?X-Amz-SignedHeaders=host&X-Amz-Signature=abc"
    )

    def handler(request):
        requested_urls.append(str(request.url))
        if request.url.path == "/custom/preview":
            return httpx.Response(
                200,
                json={"code": 200, "result": {"downUrl": encoded_down_url}},
            )
        if str(request.url) == expected_rewritten_url:
            return httpx.Response(200, json={"text": "encoded ok"})
        return httpx.Response(403, text="bad canonical request")

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

    text, _ = await fetch_transcription_text_from_media_preview(
        MediaResourceConfig(
            base_url="http://media.local",
            token="",
            domain="",
            chunk_size=3,
            timeout=10,
            multipart_check_path="/check",
            multipart_init_path="/init",
            multipart_merge_path="/merge",
            preview_path="/custom/preview",
            download_origin="http://public-object:19000",
        ),
        "http://objects/gfkd/ss/2026/06/24/tp@3E7A857311302350CE3A8C40F2864F48.mp3",
    )

    assert text == "encoded ok"
    assert requested_urls[1] == expected_rewritten_url


async def _assert_fetch_transcription_text_from_media_preview_reports_download_error_body(
    monkeypatch,
):
    import httpx

    from lightrag.api.media_transcription import (
        MediaResourceConfig,
        MediaResourceError,
        fetch_transcription_text_from_media_preview,
    )

    def handler(request):
        if request.url.path == "/custom/preview":
            return httpx.Response(
                200,
                json={
                    "code": 200,
                    "result": {
                        "downUrl": "http://internal-object:19000/gfkd/result.json"
                    },
                },
            )
        return httpx.Response(
            403,
            text="<Error><Code>SignatureDoesNotMatch</Code></Error>",
        )


async def _assert_fetch_transcription_text_from_media_preview_retries_unsigned_on_signature_mismatch(
    monkeypatch,
):
    import httpx

    from lightrag.api.media_transcription import (
        MediaResourceConfig,
        fetch_transcription_text_from_media_preview,
    )

    requested_urls = []
    signed_down_url = (
        "http://internal-object:19000/gfkd/ss/2026/06/24/"
        "tp%403E7A857311302350CE3A8C40F2864F48.json"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
        "&X-Amz-Signature=bad"
    )
    signed_rewritten_url = (
        "http://public-object:19000/gfkd/ss/2026/06/24/"
        "tp%403E7A857311302350CE3A8C40F2864F48.json"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
        "&X-Amz-Signature=bad"
    )
    unsigned_rewritten_url = (
        "http://public-object:19000/gfkd/ss/2026/06/24/"
        "tp%403E7A857311302350CE3A8C40F2864F48.json"
    )

    def handler(request):
        requested_urls.append(str(request.url))
        if request.url.path == "/custom/preview":
            return httpx.Response(
                200,
                json={"code": 200, "result": {"downUrl": signed_down_url}},
            )
        if str(request.url) == signed_rewritten_url:
            return httpx.Response(
                403,
                text="<Error><Code>SignatureDoesNotMatch</Code></Error>",
            )
        if str(request.url) == unsigned_rewritten_url:
            return httpx.Response(200, json={"text": "unsigned ok"})
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

    text, _ = await fetch_transcription_text_from_media_preview(
        MediaResourceConfig(
            base_url="http://media.local",
            token="",
            domain="",
            chunk_size=3,
            timeout=10,
            multipart_check_path="/check",
            multipart_init_path="/init",
            multipart_merge_path="/merge",
            preview_path="/custom/preview",
            download_origin="http://public-object:19000",
        ),
        "http://objects/gfkd/ss/2026/06/24/tp@3E7A857311302350CE3A8C40F2864F48.mp3",
    )

    assert text == "unsigned ok"
    assert requested_urls[1:] == [signed_rewritten_url, unsigned_rewritten_url]


def test_enqueue_media_transcription_creates_parsing_status(monkeypatch, tmp_path):
    asyncio.run(_assert_enqueue_media_transcription_creates_parsing_status(monkeypatch, tmp_path))


def test_enqueue_media_transcription_requires_media_resource_upload(
    monkeypatch, tmp_path
):
    asyncio.run(
        _assert_enqueue_media_transcription_requires_media_resource_upload(
            monkeypatch, tmp_path
        )
    )


def test_pipeline_enqueue_file_routes_mp3_to_media_transcription(monkeypatch, tmp_path):
    asyncio.run(
        _assert_pipeline_enqueue_file_routes_mp3_to_media_transcription(
            monkeypatch, tmp_path
        )
    )


def test_media_transcription_uses_deterministic_task_id(monkeypatch, tmp_path):
    asyncio.run(_assert_media_transcription_uses_deterministic_task_id(monkeypatch, tmp_path))


def test_upload_media_returns_deterministic_task_doc_id(monkeypatch, tmp_path):
    asyncio.run(_assert_upload_media_returns_deterministic_task_doc_id(monkeypatch, tmp_path))


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

    async def fake_upload(config, file_path, *, filename, content_type):
        calls.append(("upload", filename, content_type, file_path.read_bytes()))
        return "http://objects.local/gfkd/voice.mp3"

    async def fake_trigger(config, payload):
        calls.append(("trigger", payload))

    monkeypatch.setenv(
        "TRANSCRIBE_ENDPOINT", "http://220.180.237.78:4021/transcribe/async"
    )
    monkeypatch.setenv("TRANSCRIBE_BUCKET_NAME", "whisper")
    monkeypatch.setenv("LIGHTRAG_PUBLIC_CALLBACK_BASE_URL", "http://10.88.88.50:9621")
    monkeypatch.setenv("MEDIA_RESOURCE_BASE_URL", "http://media.local")
    monkeypatch.setattr(
        "lightrag.api.media_transcription.upload_media_to_resource_service",
        fake_upload,
    )
    monkeypatch.setattr(
        "lightrag.api.media_transcription.trigger_transcription", fake_trigger
    )
    monkeypatch.setattr(_dr, "_get_global_args", lambda: SimpleNamespace())

    rag = _FakeRag()
    source = tmp_path / "voice.mp3"
    source.write_bytes(b"audio")

    success, track_id = await pipeline_enqueue_file(rag, source, "scan-track")

    doc_id = calls[1][1]["task_id"]
    status = await rag.doc_status.get_by_id(doc_id)
    assert success is True
    assert track_id == "scan-track"
    assert status["status"] == DocStatus.PARSING
    assert status["file_path"] == "voice.mp3"
    assert status["metadata"]["media_transcription_status"] == "pending"
    assert calls[0] == ("upload", "voice.mp3", None, b"audio")
    assert calls[1][1]["task_id"] == doc_id
    assert calls[1][1]["url"] == "http://objects.local/gfkd/voice.mp3"
    assert source.exists()


async def _assert_media_transcription_uses_deterministic_task_id(
    monkeypatch, tmp_path
):
    from lightrag.api.routers.document_routes import pipeline_enqueue_file

    calls = []

    async def fake_upload(config, file_path, *, filename, content_type):
        return f"http://objects.local/gfkd/{filename}"

    async def fake_trigger(config, payload):
        calls.append(payload)

    monkeypatch.setenv(
        "TRANSCRIBE_ENDPOINT", "http://220.180.237.78:4021/transcribe/async"
    )
    monkeypatch.setenv("TRANSCRIBE_BUCKET_NAME", "whisper")
    monkeypatch.setenv("LIGHTRAG_PUBLIC_CALLBACK_BASE_URL", "http://10.88.88.50:9621")
    monkeypatch.setenv("MEDIA_RESOURCE_BASE_URL", "http://media.local")
    monkeypatch.setattr(
        "lightrag.api.media_transcription.upload_media_to_resource_service",
        fake_upload,
    )
    monkeypatch.setattr(
        "lightrag.api.media_transcription.trigger_transcription", fake_trigger
    )
    monkeypatch.setattr(_dr, "_get_global_args", lambda: SimpleNamespace())

    rag = _FakeRag()
    source = tmp_path / "voice.mp3"
    source.write_bytes(b"audio")

    first_success, _ = await pipeline_enqueue_file(rag, source, "scan-track-1")
    second_success, _ = await pipeline_enqueue_file(rag, source, "scan-track-2")

    assert first_success is True
    assert second_success is True
    assert len(calls) == 2
    # Deterministic doc id: the same canonical file name yields the same task id
    # across enqueues, so business systems can reconcile it (e.g. for
    # /graph/hierarchy?root_id=resource:<doc_id>).
    assert calls[0]["task_id"] == calls[1]["task_id"]
    assert await rag.doc_status.get_by_id(calls[0]["task_id"]) is not None
    assert await rag.doc_status.get_by_id(calls[1]["task_id"]) is not None


async def _assert_upload_media_returns_deterministic_task_doc_id(monkeypatch, tmp_path):
    calls = []

    async def fake_trigger(config, payload):
        calls.append(payload)

    async def fake_upload(config, file_path, *, filename, content_type):
        return f"http://objects.local/gfkd/{filename}"

    monkeypatch.setattr(
        _dr,
        "_get_router_auth_dependency",
        lambda api_key=None: (lambda: None),
    )
    monkeypatch.setattr(
        _dr,
        "_get_global_args",
        lambda: SimpleNamespace(max_upload_size=100 * 1024 * 1024),
    )
    monkeypatch.setenv(
        "TRANSCRIBE_ENDPOINT", "http://220.180.237.78:4021/transcribe/async"
    )
    monkeypatch.setenv("TRANSCRIBE_BUCKET_NAME", "whisper")
    monkeypatch.setenv("LIGHTRAG_PUBLIC_CALLBACK_BASE_URL", "http://10.88.88.50:9621")
    monkeypatch.setenv("MEDIA_RESOURCE_BASE_URL", "http://media.local")
    monkeypatch.setattr(
        "lightrag.api.media_transcription.upload_media_to_resource_service",
        fake_upload,
    )
    monkeypatch.setattr(
        "lightrag.api.media_transcription.trigger_transcription", fake_trigger
    )

    rag = _FakeRag()
    doc_manager = DocumentManager(str(tmp_path))
    router = create_document_routes(rag, doc_manager)
    upload_endpoint = [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "upload_to_input_dir"
    ][-1]
    upload_file = _dr.UploadFile(filename="voice.mp3", file=BytesIO(b"audio"))
    upload_file.size = 5
    bg = _dr.BackgroundTasks()

    response = await upload_endpoint(
        bg,
        upload_file,
        WorkspaceContext("default", rag, doc_manager),
    )

    assert response.status == "success"
    # Deterministic doc id: matches the normal-file formula so the business can
    # reconcile it (e.g. /graph/hierarchy?root_id=resource:<doc_id>).
    assert response.doc_id == compute_mdhash_id("voice.mp3", prefix="doc-")
    assert len(bg.tasks) == 1
    for task in bg.tasks:
        await task.func(*task.args, **task.kwargs)

    assert len(calls) == 1
    assert response.doc_id == calls[0]["task_id"]
    assert await rag.doc_status.get_by_id(response.doc_id) is not None
    assert calls[0]["url"] == "http://objects.local/gfkd/voice.mp3"


async def _assert_enqueue_media_transcription_creates_parsing_status(
    monkeypatch, tmp_path
):
    from lightrag.api.routers.document_routes import enqueue_media_transcription

    calls = []

    async def fake_upload(config, file_path, *, filename, content_type):
        calls.append(("upload", filename, content_type, file_path.read_bytes()))
        return "http://objects.local/gfkd/lecture.mp4"

    async def fake_trigger(config, payload):
        calls.append(("trigger", config, payload))

    monkeypatch.setenv(
        "TRANSCRIBE_ENDPOINT", "http://220.180.237.78:4021/transcribe/async"
    )
    monkeypatch.setenv("TRANSCRIBE_BUCKET_NAME", "whisper")
    monkeypatch.setenv("LIGHTRAG_PUBLIC_CALLBACK_BASE_URL", "http://10.88.88.50:9621")
    monkeypatch.setenv("MEDIA_RESOURCE_BASE_URL", "http://media.local")
    monkeypatch.setattr(
        "lightrag.api.media_transcription.upload_media_to_resource_service",
        fake_upload,
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
    assert status["file_path"] == "lecture.mp4"
    assert status["track_id"] == "upload-track"
    assert status["metadata"]["media_transcription_status"] == "pending"
    assert status["metadata"]["media_source_file"] == "lecture.mp4"
    assert rag.doc_status.index_done_calls == 2
    assert calls[0] == ("upload", "lecture.mp4", "video/mp4", b"media")
    assert calls[1][2]["task_id"] == doc_id
    assert calls[1][2]["workspace"] == "default"
    assert calls[1][2]["url"] == "http://objects.local/gfkd/lecture.mp4"


async def _assert_enqueue_media_transcription_requires_media_resource_upload(
    monkeypatch, tmp_path
):
    from lightrag.api.routers.document_routes import enqueue_media_transcription

    async def fail_if_trigger_called(config, payload):
        raise AssertionError("transcription must not start without gfkb media URL")

    monkeypatch.setenv(
        "TRANSCRIBE_ENDPOINT", "http://220.180.237.78:4021/transcribe/async"
    )
    monkeypatch.setenv("TRANSCRIBE_BUCKET_NAME", "whisper")
    monkeypatch.setenv("LIGHTRAG_PUBLIC_CALLBACK_BASE_URL", "http://10.88.88.50:9621")
    monkeypatch.delenv("MEDIA_RESOURCE_BASE_URL", raising=False)
    monkeypatch.setattr(
        "lightrag.api.media_transcription.trigger_transcription",
        fail_if_trigger_called,
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
    assert "MEDIA_RESOURCE_BASE_URL" in status["error_msg"]


def test_enqueue_media_transcription_uses_media_resource_uploaded_url(monkeypatch, tmp_path):
    asyncio.run(
        _assert_enqueue_media_transcription_uses_media_resource_uploaded_url(monkeypatch, tmp_path)
    )


async def _assert_enqueue_media_transcription_uses_media_resource_uploaded_url(
    monkeypatch, tmp_path
):
    from lightrag.api.routers.document_routes import enqueue_media_transcription

    calls = []

    async def fake_trigger(config, payload):
        calls.append((config, payload))

    async def fake_upload(config, file_path, *, filename, content_type):
        calls.append(("upload", filename, content_type, file_path.read_bytes()))
        return "http://objects.local/whisper/lecture.mp4"

    monkeypatch.setenv(
        "TRANSCRIBE_ENDPOINT", "http://220.180.237.78:4021/transcribe/async"
    )
    monkeypatch.setenv("TRANSCRIBE_BUCKET_NAME", "whisper")
    monkeypatch.setenv("LIGHTRAG_PUBLIC_CALLBACK_BASE_URL", "http://10.88.88.50:9621")
    monkeypatch.setenv("MEDIA_RESOURCE_BASE_URL", "http://media.local")
    monkeypatch.setattr(
        "lightrag.api.media_transcription.upload_media_to_resource_service", fake_upload
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
        "http://objects.local/whisper/lecture.mp4"
    )
    assert calls[0] == ("upload", "lecture.mp4", "video/mp4", b"media")
    assert calls[1][1]["url"] == "http://objects.local/whisper/lecture.mp4"


def test_enqueue_media_transcription_failure_marks_failed(monkeypatch, tmp_path):
    asyncio.run(_assert_enqueue_media_transcription_failure_marks_failed(monkeypatch, tmp_path))


async def _assert_enqueue_media_transcription_failure_marks_failed(monkeypatch, tmp_path):
    from lightrag.api.routers.document_routes import enqueue_media_transcription

    async def fake_trigger(config, payload):
        raise RuntimeError("transcribe unavailable")

    async def fake_upload(config, file_path, *, filename, content_type):
        return "http://objects.local/gfkd/lecture.mp4"

    monkeypatch.setenv(
        "TRANSCRIBE_ENDPOINT", "http://220.180.237.78:4021/transcribe/async"
    )
    monkeypatch.setenv("LIGHTRAG_PUBLIC_CALLBACK_BASE_URL", "http://10.88.88.50:9621")
    monkeypatch.setenv("MEDIA_RESOURCE_BASE_URL", "http://media.local")
    monkeypatch.setattr(
        "lightrag.api.media_transcription.upload_media_to_resource_service",
        fake_upload,
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
    assert status["status"] == DocStatus.FAILED
    assert status["metadata"]["media_transcription_status"] == "failed"
    assert status["error_msg"] == "transcribe unavailable"
    assert rag.doc_status.index_done_calls == 3


def test_apply_media_transcription_callback_success_enqueues_text():
    asyncio.run(_assert_apply_media_transcription_callback_success_enqueues_text())


def test_media_transcribe_callback_uses_default_workspace_when_missing(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        _dr,
        "_get_router_auth_dependency",
        lambda api_key=None: (lambda: None),
    )
    doc_manager = DocumentManager(str(tmp_path))
    rag = _FakeRag(workspace="default")
    doc_id = compute_mdhash_id("lecture.mp4", prefix="doc-")
    asyncio.run(
        rag.doc_status.upsert(
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
    )
    app = FastAPI()
    app.include_router(create_document_routes(rag, doc_manager, api_key="test-key"))
    client = TestClient(app)

    response = client.post(
        "/documents/media/transcribe_callback",
        headers={"X-API-Key": "test-key"},
        json={"task_id": doc_id, "text": "hello world"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    assert rag.full_docs.rows[doc_id]["content"] == "hello world"
    assert rag.process_calls == 1


def test_media_transcribe_callback_uses_workspace_query(monkeypatch, tmp_path):
    monkeypatch.setattr(
        _dr,
        "_get_router_auth_dependency",
        lambda api_key=None: (lambda: None),
    )
    doc_manager = DocumentManager(str(tmp_path))
    rag = _FakeRag(workspace="test")
    doc_id = compute_mdhash_id("lecture.mp4", prefix="doc-")
    asyncio.run(
        rag.doc_status.upsert(
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
    )
    app = FastAPI()
    app.include_router(create_document_routes(rag, doc_manager, api_key="test-key"))
    client = TestClient(app)

    response = client.post(
        "/documents/media/transcribe_callback?workspace=test",
        headers={"X-API-Key": "test-key"},
        json={"task_id": doc_id, "text": "hello from query workspace"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    assert rag.full_docs.rows[doc_id]["content"] == "hello from query workspace"
    assert rag.process_calls == 1


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


def test_apply_media_transcription_callback_waiting_status_keeps_parsing():
    asyncio.run(
        _assert_apply_media_transcription_callback_waiting_status_keeps_parsing()
    )


def test_apply_media_transcription_callback_success_without_text_records_completion(
    monkeypatch,
):
    asyncio.run(
        _assert_apply_media_transcription_callback_success_without_text_records_completion(
            monkeypatch
        )
    )


def test_apply_media_transcription_callback_success_loads_media_json(monkeypatch):
    asyncio.run(
        _assert_apply_media_transcription_callback_success_loads_media_json(monkeypatch)
    )


async def _assert_apply_media_transcription_callback_waiting_status_keeps_parsing():
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
        {"task_id": doc_id, "status": "processing"},
    )

    status = await rag.doc_status.get_by_id(doc_id)
    assert await rag.full_docs.get_by_id(doc_id) is None
    assert status["status"] == DocStatus.PARSING
    assert status["metadata"]["media_transcription_status"] == "running"
    assert status["error_msg"] is None
    assert rag.doc_status.index_done_calls == 1
    assert rag.process_calls == 0


async def _assert_apply_media_transcription_callback_success_without_text_records_completion(
    monkeypatch,
):
    from lightrag.api.routers.document_routes import apply_media_transcription_callback

    monkeypatch.delenv("MEDIA_RESOURCE_BASE_URL", raising=False)
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
        {"task_id": doc_id, "status": "success", "objectName": "test.json"},
    )

    status = await rag.doc_status.get_by_id(doc_id)
    assert await rag.full_docs.get_by_id(doc_id) is None
    assert status["status"] == DocStatus.PARSING
    assert status["metadata"]["media_transcription_status"] == "completed"
    assert status["metadata"]["media_transcription_object_name"] == "test.json"
    assert status["error_msg"] == "Media transcription completed without text payload"
    assert rag.doc_status.index_done_calls == 1
    assert rag.process_calls == 0


async def _assert_apply_media_transcription_callback_success_loads_media_json(
    monkeypatch,
):
    from lightrag.api.routers.document_routes import apply_media_transcription_callback

    calls = []

    async def fake_fetch(config, media_url):
        calls.append((config.base_url, media_url))
        return "hello from media json", "http://host/whisper/lecture.json"

    monkeypatch.setenv("MEDIA_RESOURCE_BASE_URL", "http://media.local")
    monkeypatch.setattr(
        "lightrag.api.media_transcription.fetch_transcription_text_from_media_preview",
        fake_fetch,
    )

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
                "metadata": {
                    "media_transcription_status": "pending",
                    "media_uploaded_url": "http://host/whisper/lecture.mp4",
                },
            }
        }
    )

    await apply_media_transcription_callback(
        rag,
        {"task_id": doc_id, "status": "success", "objectName": "test.json"},
    )

    full_doc = await rag.full_docs.get_by_id(doc_id)
    status = await rag.doc_status.get_by_id(doc_id)
    assert calls == [("http://media.local", "http://host/whisper/lecture.mp4")]
    assert full_doc["content"] == "hello from media json"
    assert status["status"] == DocStatus.PENDING
    assert status["metadata"]["media_transcription_status"] == "completed"
    assert status["metadata"]["media_transcription_json_url"] == (
        "http://host/whisper/lecture.json"
    )
    assert status["metadata"]["media_transcription_object_name"] == "test.json"
    assert status["metadata"]["media_transcription_length"] == 21
    assert status["error_msg"] is None
    assert rag.process_calls == 1


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
