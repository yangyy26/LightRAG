"""
Callback notification helpers.

Used by the pipeline to notify external callers when long-running
operations (e.g. knowledge hierarchy generation) complete.
"""

import asyncio
from typing import Optional

import httpx

from .utils import logger

_CALLBACK_TIMEOUT = 10.0
_CALLBACK_MAX_RETRIES = 2
_CALLBACK_RETRY_DELAYS = (1.0, 3.0)


def _validate_callback_url(url: str) -> Optional[str]:
    """Return a normalized error message if *url* is invalid, else None."""
    if not isinstance(url, str) or not url.strip():
        return "callback_url is empty"
    stripped = url.strip()
    if not (stripped.startswith("http://") or stripped.startswith("https://")):
        return "callback_url must start with http:// or https://"
    return None


async def send_hierarchy_callback(callback_url: str, doc_id: str) -> bool:
    """Send a hierarchy-build notification to *callback_url* via HTTP GET.

    The *doc_id* is passed as a query parameter.  Retries on transient
    failures (up to ``_CALLBACK_MAX_RETRIES`` times with back-off) and
    never raises — errors are logged and ``False`` is returned.
    """
    last_error: Optional[str] = None
    for attempt in range(_CALLBACK_MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=_CALLBACK_TIMEOUT) as client:
                response = await client.get(
                    callback_url.strip(),
                    params={"doc_id": doc_id},
                )
                if 200 <= response.status_code < 300:
                    logger.info(
                        "Hierarchy callback succeeded: url=%s doc_id=%s "
                        "status_code=%d response=%s",
                        callback_url,
                        doc_id,
                        response.status_code,
                        response.text[:500],
                    )
                    return True
                last_error = (
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
                logger.warning(
                    "Hierarchy callback returned non-2xx: url=%s doc_id=%s "
                    "status_code=%d response=%s (attempt %d/%d)",
                    callback_url,
                    doc_id,
                    response.status_code,
                    response.text[:500],
                    attempt + 1,
                    _CALLBACK_MAX_RETRIES + 1,
                )
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            last_error = str(exc)
        except Exception:
            logger.exception(
                "Unexpected error sending hierarchy callback to %s",
                callback_url,
            )
            last_error = "unexpected"
            break  # do not retry on unexpected errors

        if attempt < _CALLBACK_MAX_RETRIES:
            delay = _CALLBACK_RETRY_DELAYS[attempt]
            logger.warning(
                "Hierarchy callback attempt %d/%d failed (%s), retrying in %.1fs",
                attempt + 1,
                _CALLBACK_MAX_RETRIES + 1,
                last_error,
                delay,
            )
            await asyncio.sleep(delay)

    logger.error(
        "Hierarchy callback to %s failed after %d attempts: %s",
        callback_url,
        _CALLBACK_MAX_RETRIES + 1,
        last_error,
    )
    return False
