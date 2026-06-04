from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from lightrag.api.workspace import (
    WorkspaceAlreadyExistsError,
    WorkspaceBusyError,
    WorkspaceCleanupError,
    WorkspaceDefaultDeletionError,
    WorkspaceInfo,
    WorkspaceManager,
    WorkspaceNotFoundError,
)
from .auth import get_router_auth_dependency as _shared_router_auth_dependency


class WorkspaceCreateRequest(BaseModel):
    id: str = Field(min_length=1)
    name: str | None = None


class WorkspaceListResponse(BaseModel):
    workspaces: list[WorkspaceInfo]


def _get_workspace_auth_dependency(api_key: Optional[str]):
    return _shared_router_auth_dependency(api_key)


def create_workspace_routes(
    workspace_manager: WorkspaceManager,
    api_key: Optional[str] = None,
):
    router = APIRouter(prefix="/workspaces", tags=["workspaces"])
    combined_auth = _get_workspace_auth_dependency(api_key)

    @router.get(
        "",
        response_model=WorkspaceListResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def list_workspaces():
        return WorkspaceListResponse(workspaces=workspace_manager.registry.list())

    @router.post(
        "",
        response_model=WorkspaceInfo,
        dependencies=[Depends(combined_auth)],
    )
    async def create_workspace(request: WorkspaceCreateRequest):
        try:
            workspace = workspace_manager.registry.create(request.id, name=request.name)
        except WorkspaceAlreadyExistsError:
            raise HTTPException(status_code=409, detail="Workspace already exists")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid workspace id")

        (workspace_manager.working_dir / workspace.id).mkdir(parents=True, exist_ok=True)
        (workspace_manager.input_dir / workspace.id).mkdir(parents=True, exist_ok=True)
        return workspace

    @router.get(
        "/{workspace_id}",
        response_model=WorkspaceInfo,
        dependencies=[Depends(combined_auth)],
    )
    async def get_workspace(workspace_id: str):
        try:
            return workspace_manager.registry.get(workspace_id)
        except WorkspaceNotFoundError:
            raise HTTPException(status_code=404, detail="Workspace not found")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid workspace id")

    @router.delete(
        "/{workspace_id}",
        response_model=WorkspaceInfo,
        dependencies=[Depends(combined_auth)],
    )
    async def delete_workspace(workspace_id: str):
        try:
            return await workspace_manager.delete_workspace(workspace_id)
        except WorkspaceDefaultDeletionError:
            raise HTTPException(
                status_code=400, detail="Cannot delete default workspace"
            )
        except WorkspaceBusyError:
            raise HTTPException(status_code=409, detail="Workspace is busy")
        except WorkspaceNotFoundError:
            raise HTTPException(status_code=404, detail="Workspace not found")
        except WorkspaceCleanupError:
            raise HTTPException(status_code=500, detail="Failed to delete workspace")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid workspace id")

    return router
