from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from dotenv import load_dotenv

from lightrag.utils import logger

load_dotenv(dotenv_path=".env", override=False)

MEDIA_VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv"})
MEDIA_AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".m4a", ".aac", ".flac"})
MEDIA_EXTENSIONS = MEDIA_VIDEO_EXTENSIONS | MEDIA_AUDIO_EXTENSIONS
MEDIA_CONTENT_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".wmv": "video/x-ms-wmv",
    ".flv": "video/x-flv",
    ".mkv": "video/x-matroska",
}


class TranscriptionConfigError(RuntimeError):
    pass


class TranscriptionPayloadError(ValueError):
    pass


class MediaResourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class TranscriptionConfig:
    endpoint: str
    bucket_name: str
    public_base_url: str


@dataclass(frozen=True)
class MediaResourceConfig:
    base_url: str
    token: str
    domain: str
    chunk_size: int
    timeout: float
    multipart_check_path: str
    multipart_init_path: str
    multipart_merge_path: str
    preview_path: str
    download_origin: str


def is_media_file(file_path: str | Path) -> bool:
    return Path(str(file_path)).suffix.lower() in MEDIA_EXTENSIONS


def load_media_resource_config() -> MediaResourceConfig | None:
    base_url = os.getenv("MEDIA_RESOURCE_BASE_URL", "").strip()
    if not base_url:
        return None

    chunk_size_raw = os.getenv("MEDIA_RESOURCE_CHUNK_SIZE", "").strip()
    timeout_raw = os.getenv("MEDIA_RESOURCE_TIMEOUT", "").strip()
    try:
        chunk_size = int(chunk_size_raw) if chunk_size_raw else 10 * 1024 * 1024
    except ValueError as exc:
        raise TranscriptionConfigError(
            "MEDIA_RESOURCE_CHUNK_SIZE must be an integer"
        ) from exc
    try:
        timeout = float(timeout_raw) if timeout_raw else 60.0
    except ValueError as exc:
        raise TranscriptionConfigError(
            "MEDIA_RESOURCE_TIMEOUT must be a number"
        ) from exc

    if chunk_size <= 0:
        raise TranscriptionConfigError(
            "MEDIA_RESOURCE_CHUNK_SIZE must be greater than 0"
        )
    if timeout <= 0:
        raise TranscriptionConfigError("MEDIA_RESOURCE_TIMEOUT must be greater than 0")

    return MediaResourceConfig(
        base_url=base_url.rstrip("/"),
        token=os.getenv("MEDIA_RESOURCE_TOKEN", "").strip(),
        domain=os.getenv("MEDIA_RESOURCE_DOMAIN", "").strip(),
        chunk_size=chunk_size,
        timeout=timeout,
        multipart_check_path=_env_path(
            "MEDIA_RESOURCE_MULTIPART_CHECK_PATH",
            "/gfkb/resource/rbs/multipart/check",
        ),
        multipart_init_path=_env_path(
            "MEDIA_RESOURCE_MULTIPART_INIT_PATH",
            "/gfkb/resource/rbs/multipart/init",
        ),
        multipart_merge_path=_env_path(
            "MEDIA_RESOURCE_MULTIPART_MERGE_PATH",
            "/gfkb/resource/rbs/multipart/merge",
        ),
        preview_path=_env_path(
            "MEDIA_RESOURCE_PREVIEW_PATH",
            "/gfkb/resource/preview/getMeterialPreviewByUrl",
        ),
        download_origin=os.getenv("MEDIA_RESOURCE_DOWNLOAD_ORIGIN", "").strip(),
    )


def load_transcription_config() -> TranscriptionConfig:
    endpoint = os.getenv("TRANSCRIBE_ENDPOINT", "").strip()
    bucket_name = os.getenv("TRANSCRIBE_BUCKET_NAME", "whisper").strip() or "whisper"
    public_base_url = os.getenv("LIGHTRAG_PUBLIC_CALLBACK_BASE_URL", "").strip()
    missing = []
    if not endpoint:
        missing.append("TRANSCRIBE_ENDPOINT")
    if not public_base_url:
        missing.append("LIGHTRAG_PUBLIC_CALLBACK_BASE_URL")
    if missing:
        raise TranscriptionConfigError(
            "Missing media transcription configuration: " + ", ".join(missing)
        )
    config = TranscriptionConfig(
        endpoint=endpoint.rstrip("/"),
        bucket_name=bucket_name,
        public_base_url=public_base_url.rstrip("/"),
    )
    logger.info(
        "Loaded media transcription config: endpoint=%s bucket_name=%s public_base_url=%s",
        config.endpoint,
        config.bucket_name,
        config.public_base_url,
    )
    return config


def build_transcription_payload(
    config: TranscriptionConfig,
    *,
    workspace: str,
    doc_id: str,
    media_url: str | None = None,
) -> dict[str, str]:
    resolved_media_url = (media_url or "").strip()
    if not resolved_media_url:
        raise TranscriptionPayloadError(
            "Media transcription requires uploaded media URL"
        )
    callback_query = urlencode({"workspace": workspace})
    callback_url = (
        f"{config.public_base_url}/documents/media/transcribe_callback"
        f"?{callback_query}"
    )
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
    status = callback_status(payload)
    code = payload.get("code")
    if status in {"failed", "fail", "error"}:
        return True
    if isinstance(code, int) and code not in {0, 1, 200}:
        return True
    return False


def callback_status(payload: dict[str, Any]) -> str:
    candidates = [
        payload,
        payload.get("data") if isinstance(payload.get("data"), dict) else None,
        payload.get("result") if isinstance(payload.get("result"), dict) else None,
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        raw_status = candidate.get("status") or candidate.get("state")
        if raw_status is None or raw_status == "":
            continue
        status = str(raw_status).strip().lower()
        alias_map = {
            "0": "pending",
            "1": "running",
            "2": "success",
            "3": "failed",
            "wait": "pending",
            "waiting": "pending",
            "queued": "queued",
            "queue": "queued",
            "pending": "pending",
            "processing": "running",
            "running": "running",
            "analyzing": "running",
            "success": "success",
            "done": "success",
            "completed": "success",
            "complete": "success",
            "failed": "failed",
            "fail": "failed",
            "error": "failed",
        }
        return alias_map.get(status, status)
    return ""


def callback_waiting(payload: dict[str, Any]) -> bool:
    return callback_status(payload) in {"pending", "queued", "running"}


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


def media_url_to_transcription_json_url(media_url: str) -> str:
    if not media_url:
        return ""
    if "?" in media_url:
        path, query = media_url.split("?", 1)
        query = "?" + query
    else:
        path, query = media_url, ""
    path_without_fragment, sep, fragment = path.partition("#")
    fragment = sep + fragment if sep else ""
    if "." not in path_without_fragment.rsplit("/", 1)[-1]:
        return f"{path_without_fragment}.json{fragment}{query}"
    base = path_without_fragment.rsplit(".", 1)[0]
    return f"{base}.json{fragment}{query}"


async def fetch_transcription_text_from_media_preview(
    config: MediaResourceConfig,
    media_url: str,
) -> tuple[str, str]:
    json_url = media_url_to_transcription_json_url(media_url)
    if not json_url:
        raise TranscriptionPayloadError("Missing media URL for transcription result")

    try:
        import httpx
    except ImportError as exc:
        raise TranscriptionConfigError(
            "httpx is required for media transcription result download"
        ) from exc

    headers = _media_resource_headers(config)
    async with httpx.AsyncClient(timeout=config.timeout) as client:
        preview_data = await _media_resource_request_json(
            client,
            "GET",
            _media_resource_url(config, config.preview_path),
            headers=headers,
            params={"url": json_url},
        )
        code = preview_data.get("code")
        if code not in {1, 200}:
            raise MediaResourceError(
                str(preview_data.get("msg") or "media preview request failed")
            )
        down_url = (
            _nested_str(preview_data, "data", "downUrl")
            or _nested_str(preview_data, "result", "downUrl")
            or _nested_str(preview_data, "data", "url")
            or _nested_str(preview_data, "result", "url")
        )
        if not down_url:
            raise MediaResourceError("media preview response missing downUrl")
        down_url = rewrite_media_download_url(down_url, config)

        logger.info(
            "Fetching media transcription JSON: json_url=%s down_url=%s",
            json_url,
            down_url,
        )
        response = await client.get(down_url)
        if response.is_error:
            response_preview = response.text[:1000] if response.text else ""
            logger.warning(
                "Media transcription JSON download failed: json_url=%s down_url=%s status_code=%s response_preview=%s",
                json_url,
                down_url,
                response.status_code,
                response_preview,
            )
            unsigned_down_url = _unsigned_url_for_signature_mismatch(
                down_url, response_preview
            )
            if unsigned_down_url:
                logger.info(
                    "Retrying media transcription JSON download without presigned query: json_url=%s down_url=%s unsigned_down_url=%s",
                    json_url,
                    down_url,
                    unsigned_down_url,
                )
                response = await client.get(unsigned_down_url)
                if not response.is_error:
                    logger.info(
                        "Media transcription JSON unsigned retry succeeded: json_url=%s unsigned_down_url=%s",
                        json_url,
                        unsigned_down_url,
                    )
                    response_preview = ""
                else:
                    retry_preview = response.text[:1000] if response.text else ""
                    logger.warning(
                        "Media transcription JSON unsigned retry failed: json_url=%s unsigned_down_url=%s status_code=%s response_preview=%s",
                        json_url,
                        unsigned_down_url,
                        response.status_code,
                        retry_preview,
                    )
                    response_preview = retry_preview
            if response.is_error:
                raise MediaResourceError(
                    "media transcription JSON download failed: "
                    f"status_code={response.status_code} response_preview={response_preview}"
                )

    payload = _parse_transcription_result_response(response.text)
    if isinstance(payload, dict):
        text = extract_transcription_text(payload)
    else:
        text = str(payload or "").strip()
        if not text:
            raise TranscriptionPayloadError("Downloaded transcription result is empty")
    return text, json_url


def _parse_transcription_result_response(raw_text: str) -> Any:
    text = (raw_text or "").strip()
    if not text:
        raise TranscriptionPayloadError("Downloaded transcription result is empty")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def rewrite_media_download_url(url: str, config: MediaResourceConfig) -> str:
    target = config.download_origin.strip().rstrip("/")
    if not url or not target:
        return url
    return re.sub(r"^https?://[^/?#]+", target, url, count=1)


def _unsigned_url_for_signature_mismatch(url: str, response_preview: str) -> str:
    if "SignatureDoesNotMatch" not in response_preview:
        return ""
    if "X-Amz-" not in url:
        return ""
    unsigned_url = url.split("?", 1)[0]
    return unsigned_url if unsigned_url != url else ""


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

    logger.info(
        "Triggering media transcription: endpoint=%s payload=%s",
        config.endpoint,
        payload,
    )
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        response = await client.post(config.endpoint, json=payload)
    response_text_preview = response.text[:1000] if response.text else ""
    logger.info(
        "Media transcription trigger response: endpoint=%s status_code=%s response_preview=%s",
        config.endpoint,
        response.status_code,
        response_text_preview,
    )
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


async def upload_media_to_resource_service(
    config: MediaResourceConfig,
    file_path: Path,
    *,
    filename: str,
    content_type: str | None,
) -> str:
    try:
        import httpx
    except ImportError as exc:
        raise TranscriptionConfigError(
            "httpx is required for media resource upload"
        ) from exc

    headers = _media_resource_headers(config)
    file_md5 = _file_md5(file_path)
    file_size = file_path.stat().st_size
    file_type = _media_resource_file_type(filename)
    resolved_content_type = content_type or _media_resource_content_type(filename)

    async with httpx.AsyncClient(timeout=config.timeout) as client:
        check_data = await _media_resource_request_json(
            client,
            "GET",
            _media_resource_url(config, config.multipart_check_path),
            headers=headers,
            params={"md5": file_md5},
        )
        check_code = check_data.get("code")
        logger.info(
            "Media resource multipart check completed: filename=%s md5=%s code=%s result_keys=%s",
            filename,
            file_md5,
            check_code,
            _dict_keys(check_data.get("result")),
        )
        if check_code == 200:
            existing_url = _nested_str(check_data, "result", "url")
            if existing_url:
                if _media_resource_url_suffix_matches(filename, existing_url):
                    logger.info(
                        "Media resource upload reused existing object: filename=%s existing_url=%s",
                        filename,
                        existing_url,
                    )
                    return existing_url
                logger.warning(
                    "Media resource fast-check URL suffix does not match media file; reuploading: filename=%s existing_url=%s",
                    filename,
                    existing_url,
                )

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
            "contentType": resolved_content_type,
            "fileType": file_type,
            "chunkUploadedList": chunk_uploaded_list,
        }
        logger.info(
            "Initializing media resource upload: filename=%s file_type=%s content_type=%s file_size=%s chunk_num=%s",
            filename,
            file_type,
            resolved_content_type,
            file_size,
            chunk_num,
        )
        init_data = await _media_resource_request_json(
            client,
            "POST",
            _media_resource_url(config, config.multipart_init_path),
            headers=headers,
            json=init_payload,
        )
        upload_info = init_data.get("result")
        if not isinstance(upload_info, dict):
            raise MediaResourceError("media upload init response missing result")

        upload_urls = upload_info.get("urlList")
        if not isinstance(upload_urls, list) or len(upload_urls) != chunk_num:
            raise MediaResourceError("media upload init response has invalid urlList")
        logger.info(
            "Media resource upload init completed: filename=%s upload_id=%s object_url=%s upload_url_count=%s",
            filename,
            upload_info.get("uploadId"),
            upload_info.get("url"),
            len(upload_urls),
        )

        uploaded_set = set(chunk_uploaded_list)
        with file_path.open("rb") as file_obj:
            for index, upload_url in enumerate(upload_urls, start=1):
                chunk = file_obj.read(config.chunk_size)
                if index in uploaded_set:
                    continue
                response = await client.put(
                    str(upload_url),
                    content=chunk,
                    headers={"Content-Type": resolved_content_type},
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
        merge_data = await _media_resource_request_json(
            client,
            "POST",
            _media_resource_url(config, config.multipart_merge_path),
            headers=headers,
            json=merge_payload,
        )
        if merge_data.get("code") != 1:
            raise MediaResourceError(
                str(merge_data.get("msg") or "media upload merge failed")
            )
        merged_url = merge_data.get("url")
        if not isinstance(merged_url, str) or not merged_url.strip():
            raise MediaResourceError("media upload merge response missing url")
        logger.info(
            "Media resource upload merge completed: filename=%s merged_url=%s",
            filename,
            merged_url.strip(),
        )
        return merged_url.strip()


def _media_resource_headers(config: MediaResourceConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if config.token:
        headers["token"] = config.token
    if config.domain:
        headers["domain"] = config.domain
    return headers


def _media_resource_url(config: MediaResourceConfig, path: str) -> str:
    return f"{config.base_url}{path}"


def _env_path(name: str, default: str) -> str:
    value = os.getenv(name, "").strip() or default
    return value if value.startswith("/") else f"/{value}"


def _file_md5(file_path: Path) -> str:
    digest = md5()
    with file_path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _media_resource_file_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix in {"mp3", "wav", "m4a", "aac", "flac"}:
        return "audio"
    if suffix in {"mp4", "avi", "mov", "wmv", "flv", "mkv"}:
        return "video"
    return suffix


def _media_resource_content_type(filename: str) -> str:
    return MEDIA_CONTENT_TYPES.get(
        Path(filename).suffix.lower(), "application/octet-stream"
    )


def _media_resource_url_suffix_matches(filename: str, url: str) -> bool:
    expected_suffix = Path(filename).suffix.lower()
    if not expected_suffix:
        return True
    url_path = url.split("?", 1)[0].split("#", 1)[0]
    return Path(url_path).suffix.lower() == expected_suffix


def _dict_keys(value: Any) -> list[str]:
    return sorted(value.keys()) if isinstance(value, dict) else []


async def _media_resource_request_json(
    client: Any, method: str, url: str, **kwargs: Any
) -> dict:
    response = await client.request(method, url, **kwargs)
    response.raise_for_status()
    data = response.json() if response.text else {}
    if not isinstance(data, dict):
        raise MediaResourceError("media resource response is not a JSON object")
    return data


def _nested_str(data: dict[str, Any], *keys: str) -> str:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return ""
        value = value.get(key)
    return value.strip() if isinstance(value, str) else ""
