from __future__ import annotations

import os
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Any

MEDIA_VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv"})
MEDIA_AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".m4a", ".aac", ".flac"})
MEDIA_EXTENSIONS = MEDIA_VIDEO_EXTENSIONS | MEDIA_AUDIO_EXTENSIONS


class TranscriptionConfigError(RuntimeError):
    pass


class TranscriptionPayloadError(ValueError):
    pass


class GfkdUploadError(RuntimeError):
    pass


@dataclass(frozen=True)
class TranscriptionConfig:
    endpoint: str
    bucket_name: str
    public_base_url: str


@dataclass(frozen=True)
class GfkdUploadConfig:
    base_url: str
    token: str
    domain: str
    chunk_size: int
    timeout: float


def is_media_file(file_path: str | Path) -> bool:
    return Path(str(file_path)).suffix.lower() in MEDIA_EXTENSIONS


def load_gfkd_upload_config() -> GfkdUploadConfig | None:
    base_url = os.getenv("GFKD_UPLOAD_BASE_URL", "").strip()
    if not base_url:
        return None

    chunk_size_raw = os.getenv("GFKD_UPLOAD_CHUNK_SIZE", "").strip()
    timeout_raw = os.getenv("GFKD_UPLOAD_TIMEOUT", "").strip()
    try:
        chunk_size = int(chunk_size_raw) if chunk_size_raw else 10 * 1024 * 1024
    except ValueError as exc:
        raise TranscriptionConfigError(
            "GFKD_UPLOAD_CHUNK_SIZE must be an integer"
        ) from exc
    try:
        timeout = float(timeout_raw) if timeout_raw else 60.0
    except ValueError as exc:
        raise TranscriptionConfigError("GFKD_UPLOAD_TIMEOUT must be a number") from exc

    if chunk_size <= 0:
        raise TranscriptionConfigError("GFKD_UPLOAD_CHUNK_SIZE must be greater than 0")
    if timeout <= 0:
        raise TranscriptionConfigError("GFKD_UPLOAD_TIMEOUT must be greater than 0")

    return GfkdUploadConfig(
        base_url=base_url.rstrip("/"),
        token=os.getenv("GFKD_UPLOAD_TOKEN", "").strip(),
        domain=os.getenv("GFKD_UPLOAD_DOMAIN", "").strip(),
        chunk_size=chunk_size,
        timeout=timeout,
    )


def load_transcription_config() -> TranscriptionConfig:
    endpoint = os.getenv(
        "TRANSCRIBE_ENDPOINT", "http://220.180.237.78:4021/transcribe/async"
    ).strip()
    bucket_name = os.getenv("TRANSCRIBE_BUCKET_NAME", "whisper").strip() or "whisper"
    public_base_url = os.getenv(
        "LIGHTRAG_PUBLIC_CALLBACK_BASE_URL", "http://10.88.88.50:9621"
    ).strip()
    missing = []
    if not endpoint:
        missing.append("TRANSCRIBE_ENDPOINT")
    if not public_base_url:
        missing.append("LIGHTRAG_PUBLIC_CALLBACK_BASE_URL")
    if missing:
        raise TranscriptionConfigError(
            "Missing media transcription configuration: " + ", ".join(missing)
        )
    return TranscriptionConfig(
        endpoint=endpoint.rstrip("/"),
        bucket_name=bucket_name,
        public_base_url=public_base_url.rstrip("/"),
    )


def build_transcription_payload(
    config: TranscriptionConfig,
    *,
    workspace: str,
    doc_id: str,
    media_url: str | None = None,
) -> dict[str, str]:
    resolved_media_url = (
        media_url
        or f"{config.public_base_url}/documents/media/files/{workspace}/{doc_id}"
    )
    callback_url = f"{config.public_base_url}/documents/media/transcribe_callback"
    return {
        "task_id": doc_id,
        "workspace": workspace,
        "url": resolved_media_url,
        "bucketName": config.bucket_name,
        "callback": callback_url,
    }


def callback_task_id(payload: dict[str, Any]) -> str:
    for key in ("task_id", "taskId", "doc_id", "docId"):
        value = payload.get(key)
        if value:
            return str(value).strip()
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("task_id", "taskId", "doc_id", "docId"):
            value = data.get(key)
            if value:
                return str(value).strip()
    raise TranscriptionPayloadError("Missing task_id in transcription callback")


def callback_failed(payload: dict[str, Any]) -> bool:
    status = str(payload.get("status") or payload.get("state") or "").strip().lower()
    code = payload.get("code")
    if status in {"failed", "fail", "error", "3"}:
        return True
    if isinstance(code, int) and code not in {0, 1, 200}:
        return True
    return False


def callback_error_message(payload: dict[str, Any]) -> str:
    for key in ("error_msg", "error", "message", "msg"):
        value = payload.get(key)
        if value:
            return str(value)
    return "Media transcription failed"


def extract_transcription_text(payload: dict[str, Any]) -> str:
    candidates = [
        payload,
        payload.get("data") if isinstance(payload.get("data"), dict) else None,
        payload.get("result") if isinstance(payload.get("result"), dict) else None,
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key in ("text", "transcription", "content", "subtitleText"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for key in ("segments", "subtitle", "subtitles"):
            value = candidate.get(key)
            if isinstance(value, list):
                text = _text_from_segments(value)
                if text:
                    return text
    raise TranscriptionPayloadError("No transcription text found in callback payload")


def _text_from_segments(segments: list[Any]) -> str:
    pieces: list[str] = []
    for segment in segments:
        if isinstance(segment, str):
            text = segment.strip()
        elif isinstance(segment, dict):
            text = str(
                segment.get("text")
                or segment.get("content")
                or segment.get("sentence")
                or ""
            ).strip()
        else:
            text = ""
        if text:
            pieces.append(text)
    return "\n".join(pieces).strip()


async def trigger_transcription(
    config: TranscriptionConfig, payload: dict[str, str]
) -> None:
    try:
        import httpx
    except ImportError as exc:
        raise TranscriptionConfigError(
            "httpx is required for media transcription"
        ) from exc

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        response = await client.post(config.endpoint, json=payload)
    response.raise_for_status()
    data = response.json() if response.text else {}
    if isinstance(data, dict):
        code = data.get("code")
        if code is not None and code not in (0, 1, 200):
            raise TranscriptionPayloadError(
                str(
                    data.get("msg")
                    or data.get("message")
                    or "Transcription trigger failed"
                )
            )


async def upload_media_to_gfkd(
    config: GfkdUploadConfig,
    file_path: Path,
    *,
    filename: str,
    content_type: str | None,
) -> str:
    try:
        import httpx
    except ImportError as exc:
        raise TranscriptionConfigError("httpx is required for gfkd upload") from exc

    headers = _gfkd_headers(config)
    file_md5 = _file_md5(file_path)
    file_size = file_path.stat().st_size
    file_type = _gfkd_file_type(filename)

    async with httpx.AsyncClient(timeout=config.timeout) as client:
        check_data = await _gfkd_request_json(
            client,
            "GET",
            _gfkd_url(config, "/gfkb/resource/rbs/multipart/check"),
            headers=headers,
            params={"md5": file_md5},
        )
        check_code = check_data.get("code")
        if check_code == 200:
            existing_url = _nested_str(check_data, "result", "url")
            if existing_url:
                return existing_url

        chunk_uploaded_list: list[int] = []
        if check_code == 2:
            uploaded = check_data.get("result", {}).get("chunkUploadedList", [])
            if isinstance(uploaded, list):
                chunk_uploaded_list = [int(item) for item in uploaded]

        chunk_num = max(1, (file_size + config.chunk_size - 1) // config.chunk_size)
        init_payload = {
            "fileName": filename,
            "size": file_size,
            "chunkSize": config.chunk_size,
            "chunkNum": chunk_num,
            "md5": file_md5,
            "contentType": content_type or "application/octet-stream",
            "fileType": file_type,
            "chunkUploadedList": chunk_uploaded_list,
        }
        init_data = await _gfkd_request_json(
            client,
            "POST",
            _gfkd_url(config, "/gfkb/resource/rbs/multipart/init"),
            headers=headers,
            json=init_payload,
        )
        upload_info = init_data.get("result")
        if not isinstance(upload_info, dict):
            raise GfkdUploadError("gfkd upload init response missing result")

        upload_urls = upload_info.get("urlList")
        if not isinstance(upload_urls, list) or len(upload_urls) != chunk_num:
            raise GfkdUploadError("gfkd upload init response has invalid urlList")

        uploaded_set = set(chunk_uploaded_list)
        with file_path.open("rb") as file_obj:
            for index, upload_url in enumerate(upload_urls, start=1):
                chunk = file_obj.read(config.chunk_size)
                if index in uploaded_set:
                    continue
                response = await client.put(
                    str(upload_url),
                    content=chunk,
                    headers={"Content-Type": content_type or "application/octet-stream"},
                )
                response.raise_for_status()

        merge_payload = {
            "uploadId": upload_info.get("uploadId"),
            "fileName": filename,
            "md5": file_md5,
            "fileType": file_type,
            "url": upload_info.get("url"),
            "chunkNum": len(upload_urls),
            "chunkSize": config.chunk_size,
            "size": file_size,
        }
        merge_data = await _gfkd_request_json(
            client,
            "POST",
            _gfkd_url(config, "/gfkb/resource/rbs/multipart/merge"),
            headers=headers,
            json=merge_payload,
        )
        if merge_data.get("code") != 1:
            raise GfkdUploadError(
                str(merge_data.get("msg") or "gfkd upload merge failed")
            )
        merged_url = merge_data.get("url")
        if not isinstance(merged_url, str) or not merged_url.strip():
            raise GfkdUploadError("gfkd upload merge response missing url")
        return merged_url.strip()


def _gfkd_headers(config: GfkdUploadConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if config.token:
        headers["token"] = config.token
    if config.domain:
        headers["domain"] = config.domain
    return headers


def _gfkd_url(config: GfkdUploadConfig, path: str) -> str:
    return f"{config.base_url}{path}"


def _file_md5(file_path: Path) -> str:
    digest = md5()
    with file_path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gfkd_file_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix in {"mp3", "wav", "m4a", "aac", "flac"}:
        return "audio"
    if suffix in {"mp4", "avi", "mov", "wmv", "flv", "mkv"}:
        return "video"
    return suffix


async def _gfkd_request_json(client: Any, method: str, url: str, **kwargs: Any) -> dict:
    response = await client.request(method, url, **kwargs)
    response.raise_for_status()
    data = response.json() if response.text else {}
    if not isinstance(data, dict):
        raise GfkdUploadError("gfkd upload response is not a JSON object")
    return data


def _nested_str(data: dict[str, Any], *keys: str) -> str:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return ""
        value = value.get(key)
    return value.strip() if isinstance(value, str) else ""
