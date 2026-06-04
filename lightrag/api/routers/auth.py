from typing import Optional

from fastapi import HTTPException, Request


def get_router_auth_dependency(api_key: Optional[str]):
    try:
        from lightrag.api.utils_api import get_combined_auth_dependency

        return get_combined_auth_dependency(api_key)
    except ModuleNotFoundError:
        if api_key is None:
            async def no_auth():
                return True

            return no_auth

        async def api_key_only_auth(request: Request):
            header_key = request.headers.get("X-API-Key")
            bearer = request.headers.get("Authorization", "")
            if header_key == api_key or bearer == f"Bearer {api_key}":
                return True
            raise HTTPException(status_code=401, detail="Invalid API Key")

        return api_key_only_auth
