import sys
from types import ModuleType

from lightrag.api.routers import auth as router_auth
from lightrag.api.routers import workspace_routes


def test_router_auth_delegates_to_combined_auth_without_api_key(monkeypatch):
    calls = []

    async def combined_dependency():
        return "combined"

    def fake_get_combined_auth_dependency(api_key=None):
        calls.append(api_key)
        return combined_dependency

    utils_api = ModuleType("lightrag.api.utils_api")
    utils_api.get_combined_auth_dependency = fake_get_combined_auth_dependency
    monkeypatch.setitem(sys.modules, "lightrag.api.utils_api", utils_api)

    dependency = router_auth.get_router_auth_dependency(api_key=None)

    assert dependency is combined_dependency
    assert calls == [None]


def test_router_auth_delegates_to_combined_auth_with_api_key(monkeypatch):
    calls = []

    async def combined_dependency():
        return "combined"

    def fake_get_combined_auth_dependency(api_key=None):
        calls.append(api_key)
        return combined_dependency

    utils_api = ModuleType("lightrag.api.utils_api")
    utils_api.get_combined_auth_dependency = fake_get_combined_auth_dependency
    monkeypatch.setitem(sys.modules, "lightrag.api.utils_api", utils_api)

    dependency = router_auth.get_router_auth_dependency(api_key="secret")

    assert dependency is combined_dependency
    assert calls == ["secret"]


def test_workspace_routes_use_shared_router_auth_without_api_key(monkeypatch):
    calls = []

    async def shared_dependency():
        return True

    def fake_shared_router_auth_dependency(api_key=None):
        calls.append(api_key)
        return shared_dependency

    monkeypatch.setattr(
        workspace_routes,
        "_shared_router_auth_dependency",
        fake_shared_router_auth_dependency,
    )

    class Registry:
        def list(self):
            return []

    class WorkspaceManager:
        registry = Registry()

    router = workspace_routes.create_workspace_routes(
        WorkspaceManager(),
        api_key=None,
    )

    assert calls == [None]
    assert router.routes
