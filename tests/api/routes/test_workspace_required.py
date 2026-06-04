from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from lightrag.api.routers.workspace_routes import create_workspace_routes
from lightrag.api.routers.query_routes import create_query_routes
from lightrag.api.routers.graph_routes import create_graph_routes
from lightrag.api.workspace import (
    WorkspaceContext,
    WorkspaceManager,
    WorkspaceRegistry,
    make_workspace_dependency,
)


class FakeStorage:
    async def drop(self):
        return {"status": "success"}


class FakeRAG:
    def __init__(self, workspace: str):
        self.workspace = workspace
        self.pipeline_status = {
            "busy": False,
            "scanning": False,
            "destructive_busy": False,
            "pending_enqueues": 0,
        }
        for attr in (
            "full_docs",
            "text_chunks",
            "llm_response_cache",
            "full_entities",
            "full_relations",
            "entity_chunks",
            "relation_chunks",
            "entities_vdb",
            "relationships_vdb",
            "chunks_vdb",
            "chunk_entity_relation_graph",
            "doc_status",
        ):
            setattr(self, attr, FakeStorage())

    async def initialize_storages(self):
        return None

    async def check_and_migrate_data(self):
        return None

    async def finalize_storages(self):
        return None

    async def aquery_llm(self, query, param):
        return {
            "llm_response": {"content": f"{self.workspace}:{query}"},
            "data": {"references": []},
        }

    async def get_graph_labels(self):
        return [self.workspace]


class FakeDocManager:
    def __init__(self, input_dir: str, workspace: str):
        self.input_dir = Path(input_dir) / workspace
        self.workspace = workspace
        self.input_dir.mkdir(parents=True, exist_ok=True)


def build_app(tmp_path: Path) -> FastAPI:
    registry = WorkspaceRegistry(tmp_path / "rag")
    manager = WorkspaceManager(
        registry=registry,
        working_dir=tmp_path / "rag",
        input_dir=tmp_path / "inputs",
        rag_factory=FakeRAG,
        document_manager_cls=FakeDocManager,
        default_workspace="default",
    )
    workspace_dep = make_workspace_dependency(manager)
    app = FastAPI()
    app.include_router(create_workspace_routes(manager, api_key=None))
    app.include_router(create_query_routes(workspace_dep, api_key=None))
    app.include_router(create_graph_routes(workspace_dep, api_key=None))

    @app.get("/business")
    async def business(context: WorkspaceContext = Depends(workspace_dep)):
        return {"workspace": context.workspace_id}

    return app


def test_workspace_routes_create_list_get_and_business_header(tmp_path: Path):
    client = TestClient(build_app(tmp_path))

    create = client.post("/workspaces", json={"id": "project_a", "name": "Project A"})
    assert create.status_code == 200
    assert create.json()["id"] == "project_a"

    listed = client.get("/workspaces")
    assert listed.status_code == 200
    assert [workspace["id"] for workspace in listed.json()["workspaces"]] == [
        "default",
        "project_a",
    ]

    get_one = client.get("/workspaces/project_a")
    assert get_one.status_code == 200
    assert get_one.json()["name"] == "Project A"

    business = client.get("/business", headers={"LIGHTRAG-WORKSPACE": "project_a"})
    assert business.status_code == 200
    assert business.json() == {"workspace": "project_a"}


def test_business_route_uses_default_workspace_without_header(tmp_path: Path):
    client = TestClient(build_app(tmp_path))

    response = client.get("/business")

    assert response.status_code == 200
    assert response.json() == {"workspace": "default"}


def test_business_route_rejects_unknown_workspace(tmp_path: Path):
    client = TestClient(build_app(tmp_path))

    response = client.get("/business", headers={"LIGHTRAG-WORKSPACE": "missing"})

    assert response.status_code == 404
    assert response.json()["detail"] == "Workspace not found"


def test_workspace_create_duplicate_returns_409(tmp_path: Path):
    client = TestClient(build_app(tmp_path))
    assert client.post("/workspaces", json={"id": "project_a"}).status_code == 200

    duplicate = client.post("/workspaces", json={"id": "project_a"})

    assert duplicate.status_code == 409


def test_delete_workspace_removes_registry_and_dirs(tmp_path: Path):
    client = TestClient(build_app(tmp_path))
    assert client.post("/workspaces", json={"id": "project_a"}).status_code == 200

    response = client.delete("/workspaces/project_a")

    assert response.status_code == 200
    assert response.json()["id"] == "project_a"
    assert not (tmp_path / "rag" / "project_a").exists()
    assert not (tmp_path / "inputs" / "project_a").exists()
    assert [workspace["id"] for workspace in client.get("/workspaces").json()["workspaces"]] == [
        "default",
    ]


def test_delete_default_workspace_returns_400(tmp_path: Path):
    client = TestClient(build_app(tmp_path))

    response = client.delete("/workspaces/default")

    assert response.status_code == 400
    assert response.json()["detail"] == "Cannot delete default workspace"


def test_query_route_uses_default_workspace_without_header(tmp_path: Path):
    client = TestClient(build_app(tmp_path))

    response = client.post("/query", json={"query": "hello world"})

    assert response.status_code == 200
    assert response.json()["response"] == "default:hello world"


def test_query_route_uses_workspace_context(tmp_path: Path):
    client = TestClient(build_app(tmp_path))
    assert client.post("/workspaces", json={"id": "project_a"}).status_code == 200

    response = client.post(
        "/query",
        json={"query": "hello world"},
        headers={"LIGHTRAG-WORKSPACE": "project_a"},
    )

    assert response.status_code == 200
    assert response.json()["response"] == "project_a:hello world"


def test_graph_route_uses_workspace_context(tmp_path: Path):
    client = TestClient(build_app(tmp_path))
    assert client.post("/workspaces", json={"id": "project_a"}).status_code == 200

    response = client.get(
        "/graph/label/list",
        headers={"LIGHTRAG-WORKSPACE": "project_a"},
    )

    assert response.status_code == 200
    assert response.json() == ["project_a"]
