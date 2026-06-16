from __future__ import annotations

import asyncio
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

from lightrag.kg.shared_storage import get_namespace_data
from lightrag.utils import load_json, write_json

WORKSPACE_HEADER = "LIGHTRAG-WORKSPACE"
_WORKSPACE_ID_RE = re.compile(r"^[a-zA-Z0-9_]+$")


class WorkspaceAlreadyExistsError(ValueError):
    pass


class WorkspaceNotFoundError(ValueError):
    pass


class WorkspaceBusyError(RuntimeError):
    pass


class WorkspaceCleanupError(RuntimeError):
    pass


class WorkspaceDefaultDeletionError(ValueError):
    pass


class WorkspaceInfo(BaseModel):
    id: str = Field(description="Workspace identifier")
    name: str = Field(description="Display name")
    created_at: str = Field(description="UTC creation timestamp")
    updated_at: str = Field(description="UTC update timestamp")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_workspace_id(value: str) -> str:
    workspace_id = str(value or "").strip()
    if not workspace_id or not _WORKSPACE_ID_RE.fullmatch(workspace_id):
        raise ValueError("Invalid workspace id")
    return workspace_id


class WorkspaceRegistry:
    def __init__(self, working_dir: str | Path):
        self.working_dir = Path(working_dir)
        self.file_path = self.working_dir / "workspaces.json"
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[WorkspaceInfo]:
        data = self._read()
        return [WorkspaceInfo(**item) for item in data["workspaces"]]

    def exists(self, workspace_id: str) -> bool:
        workspace_id = validate_workspace_id(workspace_id)
        return any(item.id == workspace_id for item in self.list())

    def get(self, workspace_id: str) -> WorkspaceInfo:
        workspace_id = validate_workspace_id(workspace_id)
        for item in self.list():
            if item.id == workspace_id:
                return item
        raise WorkspaceNotFoundError(workspace_id)

    def create(self, workspace_id: str, name: str | None = None) -> WorkspaceInfo:
        workspace_id = validate_workspace_id(workspace_id)
        data = self._read()
        if any(item["id"] == workspace_id for item in data["workspaces"]):
            raise WorkspaceAlreadyExistsError(workspace_id)

        timestamp = utc_now_iso()
        item = {
            "id": workspace_id,
            "name": (name or workspace_id).strip() or workspace_id,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        data["workspaces"].append(item)
        data["workspaces"].sort(key=lambda row: row["id"])
        self._write(data)
        return WorkspaceInfo(**item)

    def delete(self, workspace_id: str) -> WorkspaceInfo:
        workspace_id = validate_workspace_id(workspace_id)
        data = self._read()
        kept = [item for item in data["workspaces"] if item["id"] != workspace_id]
        if len(kept) == len(data["workspaces"]):
            raise WorkspaceNotFoundError(workspace_id)

        deleted = next(item for item in data["workspaces"] if item["id"] == workspace_id)
        data["workspaces"] = kept
        self._write(data)
        return WorkspaceInfo(**deleted)

    def _read(self) -> dict[str, list[dict[str, str]]]:
        loaded = load_json(str(self.file_path)) if self.file_path.exists() else None
        if not loaded:
            return {"workspaces": []}
        workspaces = loaded.get("workspaces", [])
        if not isinstance(workspaces, list):
            return {"workspaces": []}
        return {"workspaces": workspaces}

    def _write(self, data: dict[str, list[dict[str, str]]]) -> None:
        write_json(data, str(self.file_path))


@dataclass(slots=True)
class WorkspaceContext:
    workspace_id: str
    rag: Any
    doc_manager: Any


def require_workspace_header(request: Request) -> str:
    raw_value = request.headers.get(WORKSPACE_HEADER)
    if raw_value is None or not raw_value.strip():
        raise HTTPException(
            status_code=400,
            detail=f"Missing {WORKSPACE_HEADER} header",
        )

    try:
        return validate_workspace_id(raw_value)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid workspace id")


def get_workspace_header(request: Request) -> str | None:
    raw_value = request.headers.get(WORKSPACE_HEADER)
    if raw_value is None or not raw_value.strip():
        return None

    try:
        return validate_workspace_id(raw_value)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid workspace id")


class WorkspaceManager:
    def __init__(
        self,
        registry: WorkspaceRegistry,
        working_dir: str | Path,
        input_dir: str | Path,
        rag_factory: Callable[[str], Any],
        document_manager_cls: type,
        default_workspace: str | None = None,
    ):
        self.registry = registry
        self.working_dir = Path(working_dir)
        self.input_dir = Path(input_dir)
        self.rag_factory = rag_factory
        self.document_manager_cls = document_manager_cls
        self.default_workspace = validate_workspace_id(default_workspace or "default")
        self._contexts: dict[str, WorkspaceContext] = {}
        self._context_locks: dict[str, asyncio.Lock] = {}
        self._ensure_workspace_registered(self.default_workspace)

    async def require_context(self, workspace_id: str | None) -> WorkspaceContext:
        try:
            workspace_id = validate_workspace_id(
                workspace_id or self.default_workspace
            )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid workspace id")

        if not self.registry.exists(workspace_id):
            raise HTTPException(status_code=404, detail="Workspace not found")

        return await self.get_context(workspace_id)

    async def get_context(self, workspace_id: str) -> WorkspaceContext:
        workspace_id = validate_workspace_id(workspace_id)
        cached = self._contexts.get(workspace_id)
        if cached is not None:
            return cached

        lock = self._context_locks.setdefault(workspace_id, asyncio.Lock())
        async with lock:
            cached = self._contexts.get(workspace_id)
            if cached is not None:
                return cached

            rag = self.rag_factory(workspace_id)
            await rag.initialize_storages()
            await rag.check_and_migrate_data()
            doc_manager = self.document_manager_cls(
                str(self.input_dir), workspace=workspace_id
            )
            context = WorkspaceContext(
                workspace_id=workspace_id,
                rag=rag,
                doc_manager=doc_manager,
            )
            self._contexts[workspace_id] = context
            return context

    async def delete_workspace(self, workspace_id: str) -> WorkspaceInfo:
        workspace_id = validate_workspace_id(workspace_id)
        if workspace_id == self.default_workspace:
            raise WorkspaceDefaultDeletionError(workspace_id)
        self.registry.get(workspace_id)
        context = await self.get_context(workspace_id)
        pipeline_status = await self._get_pipeline_status(context.rag)
        if self._is_busy(pipeline_status):
            raise WorkspaceBusyError(workspace_id)

        await self._drop_workspace_storages(context.rag)
        await context.rag.finalize_storages()
        self._contexts.pop(workspace_id, None)
        self._remove_workspace_dirs(workspace_id)
        return self.registry.delete(workspace_id)

    async def shutdown(self) -> None:
        contexts = list(self._contexts.values())
        self._contexts.clear()
        for context in contexts:
            await context.rag.finalize_storages()
        self._context_locks.clear()

    async def _get_pipeline_status(self, rag: Any) -> dict[str, Any]:
        if hasattr(rag, "pipeline_status"):
            return dict(rag.pipeline_status)
        return await get_namespace_data("pipeline_status", workspace=rag.workspace)

    def _is_busy(self, status: dict[str, Any]) -> bool:
        return bool(
            status.get("busy")
            or status.get("scanning")
            or status.get("destructive_busy")
            or status.get("pending_enqueues", 0) > 0
        )

    async def _drop_workspace_storages(self, rag: Any) -> None:
        storages = [
            rag.full_docs,
            rag.text_chunks,
            rag.llm_response_cache,
            rag.full_entities,
            rag.full_relations,
            rag.entity_chunks,
            rag.relation_chunks,
            rag.entities_vdb,
            rag.relationships_vdb,
            rag.chunks_vdb,
            rag.chunk_entity_relation_graph,
            rag.doc_status,
        ]
        for storage in storages:
            await storage.drop()

    def _remove_workspace_dirs(self, workspace_id: str) -> None:
        for root in (self.working_dir, self.input_dir):
            root.mkdir(parents=True, exist_ok=True)
            root_resolved = root.resolve()
            target = (root / workspace_id).resolve()
            if target == root_resolved or root_resolved not in target.parents:
                raise WorkspaceCleanupError(str(target))
            if target.exists():
                shutil.rmtree(target)

    def _ensure_workspace_registered(self, workspace_id: str) -> None:
        if not self.registry.exists(workspace_id):
            self.registry.create(workspace_id)


def make_workspace_dependency(manager: WorkspaceManager):
    async def dependency(request: Request) -> WorkspaceContext:
        workspace_id = get_workspace_header(request)
        return await manager.require_context(workspace_id)

    dependency._workspace_manager = manager  # type: ignore[attr-defined]
    return dependency
