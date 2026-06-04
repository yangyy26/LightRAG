import { FormEvent, useEffect, useMemo, useState } from 'react'
import { FolderKanban, Plus, RefreshCw } from 'lucide-react'
import { toast } from 'sonner'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/Dialog'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/Select'
import { createWorkspace, listWorkspaces, WorkspaceInfo } from '@/api/lightrag'
import { useBackendState } from '@/stores/state'
import { useGraphStore } from '@/stores/graph'
import { useSettingsStore } from '@/stores/settings'

const resetWorkspaceScopedState = () => {
  const graphStore = useGraphStore.getState()
  graphStore.reset()
  graphStore.setGraphDataFetchAttempted(false)
  graphStore.setLabelsFetchAttempted(false)
  graphStore.setIsFetching(false)
  useBackendState.getState().clear()
  useSettingsStore.getState().setRetrievalHistory([])
}

export default function WorkspaceSelector() {
  const currentWorkspaceId = useSettingsStore.use.currentWorkspaceId()
  const setCurrentWorkspaceId = useSettingsStore.use.setCurrentWorkspaceId()
  const [workspaces, setWorkspaces] = useState<WorkspaceInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [newWorkspaceId, setNewWorkspaceId] = useState('')
  const [newWorkspaceName, setNewWorkspaceName] = useState('')
  const [creating, setCreating] = useState(false)

  const selectedWorkspace = useMemo(
    () => workspaces.find((workspace) => workspace.id === currentWorkspaceId),
    [currentWorkspaceId, workspaces]
  )

  const loadWorkspaces = async () => {
    setLoading(true)
    try {
      const items = await listWorkspaces()
      const workspaceItems = Array.isArray(items) ? items : []
      setWorkspaces(workspaceItems)
      if (!currentWorkspaceId && workspaceItems.length > 0) {
        setCurrentWorkspaceId(workspaceItems[0].id)
        resetWorkspaceScopedState()
      }
    } catch (error) {
      toast.error(`Failed to load workspaces: ${String(error)}`)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadWorkspaces()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleWorkspaceChange = (workspaceId: string) => {
    if (workspaceId === currentWorkspaceId) return
    setCurrentWorkspaceId(workspaceId)
    resetWorkspaceScopedState()
  }

  const handleCreateWorkspace = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const id = newWorkspaceId.trim()
    if (!id) return

    setCreating(true)
    try {
      const workspace = await createWorkspace({
        id,
        name: newWorkspaceName.trim() || null
      })
      setWorkspaces((items) => [...items.filter((item) => item.id !== workspace.id), workspace])
      setCurrentWorkspaceId(workspace.id)
      resetWorkspaceScopedState()
      setNewWorkspaceId('')
      setNewWorkspaceName('')
      setDialogOpen(false)
    } catch (error) {
      toast.error(`Failed to create workspace: ${String(error)}`)
    } finally {
      setCreating(false)
    }
  }

  return (
    <div className="flex h-8 min-w-0 items-center gap-1">
      <FolderKanban className="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
      <Select
        value={currentWorkspaceId ?? ''}
        onValueChange={handleWorkspaceChange}
        disabled={loading || workspaces.length === 0}
      >
        <SelectTrigger className="h-8 w-[180px]">
          <SelectValue placeholder={loading ? 'Loading...' : 'Workspace'} />
        </SelectTrigger>
        <SelectContent>
          {workspaces.map((workspace) => (
            <SelectItem key={workspace.id} value={workspace.id}>
              {workspace.name || workspace.id}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Button
        variant="ghost"
        size="icon"
        tooltip="Refresh workspaces"
        side="bottom"
        onClick={loadWorkspaces}
        disabled={loading}
      >
        <RefreshCw className={loading ? 'animate-spin' : ''} />
      </Button>
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogTrigger asChild>
          <Button variant="ghost" size="icon" tooltip="Create workspace" side="bottom">
            <Plus />
          </Button>
        </DialogTrigger>
        <DialogContent className="max-w-sm">
          <form onSubmit={handleCreateWorkspace} className="space-y-4">
            <DialogHeader>
              <DialogTitle>Create Workspace</DialogTitle>
              <DialogDescription>
                Business APIs use this workspace after it is selected.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-3">
              <Input
                value={newWorkspaceId}
                onChange={(event) => setNewWorkspaceId(event.target.value)}
                placeholder="workspace-id"
                autoFocus
              />
              <Input
                value={newWorkspaceName}
                onChange={(event) => setNewWorkspaceName(event.target.value)}
                placeholder="Display name"
              />
            </div>
            <DialogFooter>
              <Button type="submit" disabled={creating || !newWorkspaceId.trim()}>
                {creating ? 'Creating...' : 'Create'}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
      {selectedWorkspace && (
        <span className="hidden max-w-[120px] truncate text-xs text-muted-foreground xl:inline">
          {selectedWorkspace.id}
        </span>
      )}
    </div>
  )
}
