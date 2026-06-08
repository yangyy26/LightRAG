import { afterEach, beforeAll, describe, expect, test } from 'bun:test'

type DocumentsRequest = {
  status_filter?: 'pending' | 'processing' | 'preprocessed' | 'processed' | 'failed' | null
  page: number
  page_size: number
  sort_field: 'created_at' | 'updated_at' | 'id' | 'file_path'
  sort_direction: 'asc' | 'desc'
}

type LightragApiModule = typeof import('./lightrag')

const storageMock = () => {
  const data = new Map<string, string>()

  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => {
      data.set(key, value)
    },
    removeItem: (key: string) => {
      data.delete(key)
    },
    clear: () => {
      data.clear()
    }
  }
}

let apiModule: LightragApiModule

beforeAll(async () => {
  Object.defineProperty(globalThis, 'localStorage', {
    value: storageMock(),
    configurable: true
  })
  Object.defineProperty(globalThis, 'sessionStorage', {
    value: storageMock(),
    configurable: true
  })

  apiModule = await import('./lightrag')
})

afterEach(() => {
  apiModule.__resetPaginatedDocumentRequestsForTests()
  apiModule.__resetWorkspaceListGetForTests()
  apiModule.__resetHierarchyGetForTests()
  apiModule.__resetGraphsGetForTests()
})

describe('workspace headers', () => {
  afterEach(async () => {
    const { useSettingsStore } = await import('@/stores/settings')
    useSettingsStore.getState().setCurrentWorkspaceId(null)
    localStorage.clear()
  })

  test('requires workspace for business API paths only', () => {
    expect(apiModule.requiresWorkspaceHeader('/query')).toBe(true)
    expect(apiModule.requiresWorkspaceHeader('/documents')).toBe(true)
    expect(apiModule.requiresWorkspaceHeader('/graph/label/list')).toBe(true)

    expect(apiModule.requiresWorkspaceHeader('/workspaces')).toBe(false)
    expect(apiModule.requiresWorkspaceHeader('/workspaces/project-a')).toBe(false)
    expect(apiModule.requiresWorkspaceHeader('/health')).toBe(false)
    expect(apiModule.requiresWorkspaceHeader('/auth-status')).toBe(false)
    expect(apiModule.requiresWorkspaceHeader('/login')).toBe(false)
  })

  test('builds LIGHTRAG-WORKSPACE header from selected workspace', async () => {
    const { useSettingsStore } = await import('@/stores/settings')
    expect(apiModule.getWorkspaceHeaders('/query')).toEqual({})

    useSettingsStore.getState().setCurrentWorkspaceId('project-a')

    expect(apiModule.getWorkspaceHeaders('/query')).toEqual({
      [apiModule.workspaceHeader]: 'project-a'
    })
    expect(apiModule.getWorkspaceHeaders('/workspaces')).toEqual({})
  })
})

describe('workspace list', () => {
  test('normalizes missing workspace payload to an empty list', async () => {
    apiModule.__setWorkspaceListGetForTests(async () => ({}))

    await expect(apiModule.listWorkspaces()).resolves.toEqual([])
  })
})

describe('queryGraphHierarchy', () => {
  test('converts hierarchy tree payload to graph nodes and edges', async () => {
    let requestedUrl = ''
    apiModule.__setHierarchyGetForTests(async (url) => {
      requestedUrl = url
      return {
        data: {
          entity_id: 'resource:doc-a',
          entity_name: 'trees.pdf',
          root_id: 'resource:doc-a',
          hierarchy_kind: 'root',
          children: [
            {
              entity_id: 'Tree',
              entity_name: 'Tree',
              root_id: 'resource:doc-a',
              parent_id: 'resource:doc-a',
              children: [
                {
                  entity_id: 'Binary Tree',
                  entity_name: 'Binary Tree',
                  root_id: 'resource:doc-a',
                  parent_id: 'Tree',
                  children: []
                }
              ]
            }
          ]
        }
      }
    })

    const graph = await apiModule.queryGraphHierarchy('resource:doc-a', 5)

    expect(requestedUrl).toBe('/graph/hierarchy?root_id=resource%3Adoc-a&max_depth=5')
    expect(graph.nodes.map((node) => node.id)).toEqual([
      'resource:doc-a',
      'Tree',
      'Binary Tree'
    ])
    expect(graph.edges.map((edge) => [edge.source, edge.target])).toEqual([
      ['resource:doc-a', 'Tree'],
      ['Tree', 'Binary Tree']
    ])
    expect(graph.edges[0].properties.edge_type).toBe('hierarchy')
  })

  test('falls back to normal graph when hierarchy is not found', async () => {
    let hierarchyRequested = ''
    let fallbackRequest: [string, number, number] | null = null
    apiModule.__setHierarchyGetForTests(async (url) => {
      hierarchyRequested = url
      const error = new Error('Hierarchy root not found') as Error & {
        response?: { status: number }
      }
      error.response = { status: 404 }
      throw error
    })
    apiModule.__setGraphsGetForTests(async (label, maxDepth, maxNodes) => {
      fallbackRequest = [label, maxDepth, maxNodes]
      return {
        nodes: [],
        edges: []
      }
    })

    const graph = await apiModule.queryResourceGraph('Not A Resource', 3, 1000)

    expect(hierarchyRequested).toBe('/graph/hierarchy?root_id=Not%20A%20Resource&max_depth=3')
    expect(fallbackRequest).toEqual(['Not A Resource', 3, 1000])
    expect(graph.nodes).toBeDefined()
    expect(graph.edges).toBeDefined()
  })
})

describe('getDocumentsPaginated', () => {
  test('issues a fresh request after aborting a timed-out in-flight request', async () => {
    const request: DocumentsRequest = {
      status_filter: null,
      page: 1,
      page_size: 20,
      sort_field: 'updated_at',
      sort_direction: 'desc'
    }

    let callCount = 0
    const resolvers: Array<(value: any) => void> = []

    apiModule.__setPaginatedDocumentsPostForTests((_request, controller) => {
      callCount += 1

      return new Promise((resolve, reject) => {
        resolvers.push(resolve)
        controller.signal.addEventListener(
          'abort',
          () => reject(new DOMException('Aborted', 'AbortError')),
          { once: true }
        )
      })
    })

    const firstRequest = apiModule.getDocumentsPaginated(request)
    const secondRequest = apiModule.getDocumentsPaginated(request)

    expect(callCount).toBe(1)

    apiModule.abortDocumentsPaginated(request)
    const [firstResult, secondResult] = await Promise.allSettled([
      firstRequest,
      secondRequest
    ])
    expect(firstResult.status).toBe('rejected')
    expect(secondResult.status).toBe('rejected')

    const thirdRequest = apiModule.getDocumentsPaginated(request)
    expect(callCount).toBe(2)

    resolvers[1]({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })

    await expect(thirdRequest).resolves.toEqual({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })
  })

  test('times out hanging requests and allows a fresh retry', async () => {
    const request: DocumentsRequest = {
      status_filter: null,
      page: 1,
      page_size: 20,
      sort_field: 'updated_at',
      sort_direction: 'desc'
    }

    let callCount = 0
    const resolvers: Array<(value: any) => void> = []

    apiModule.__setPaginatedDocumentsPostForTests((_request, controller) => {
      callCount += 1

      return new Promise((resolve, reject) => {
        resolvers.push(resolve)
        controller.signal.addEventListener(
          'abort',
          () => reject(new DOMException('Aborted', 'AbortError')),
          { once: true }
        )
      })
    })

    await expect(
      apiModule.getDocumentsPaginatedWithTimeout(request, 1)
    ).rejects.toThrow('Document fetch timeout')

    expect(callCount).toBe(1)

    const retryRequest = apiModule.getDocumentsPaginated(request)
    expect(callCount).toBe(2)

    resolvers[1]({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })

    await expect(retryRequest).resolves.toEqual({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })
  })

  test('does not abort a shared request when only one timeout subscriber expires', async () => {
    const request: DocumentsRequest = {
      status_filter: null,
      page: 1,
      page_size: 20,
      sort_field: 'updated_at',
      sort_direction: 'desc'
    }

    let callCount = 0
    let resolveSharedRequest: ((value: any) => void) | undefined
    let abortCount = 0

    apiModule.__setPaginatedDocumentsPostForTests((_request, controller) => {
      callCount += 1

      return new Promise((resolve, reject) => {
        resolveSharedRequest = resolve
        controller.signal.addEventListener(
          'abort',
          () => {
            abortCount += 1
            reject(new DOMException('Aborted', 'AbortError'))
          },
          { once: true }
        )
      })
    })

    const shortTimeoutRequest = apiModule.getDocumentsPaginatedWithTimeout(request, 1)
    const longTimeoutRequest = apiModule.getDocumentsPaginatedWithTimeout(request, 100)

    await expect(shortTimeoutRequest).rejects.toThrow('Document fetch timeout')

    expect(callCount).toBe(1)
    expect(abortCount).toBe(0)

    resolveSharedRequest?.({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })

    await expect(longTimeoutRequest).resolves.toEqual({
      documents: [],
      pagination: {
        page: 1,
        page_size: 20,
        total_count: 0,
        total_pages: 0,
        has_next: false,
        has_prev: false
      },
      status_counts: { all: 0 }
    })
  })
})
