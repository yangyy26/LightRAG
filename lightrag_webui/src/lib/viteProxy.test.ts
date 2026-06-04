/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'
import { buildDevProxyConfig, defaultApiEndpoints } from './viteProxy'

describe('buildDevProxyConfig', () => {
  test('proxies default backend endpoints when env is omitted', () => {
    const proxy = buildDevProxyConfig({})

    expect(Object.keys(proxy)).toContain('/workspaces')
    expect(Object.keys(proxy)).toContain('/documents')
    expect(Object.keys(proxy)).toContain('/query')
    expect(proxy['/workspaces'].target).toBe('http://localhost:9621')
  })

  test('allows explicit proxy disable', () => {
    expect(buildDevProxyConfig({ VITE_API_PROXY: 'false' })).toEqual({})
  })

  test('prefixes proxy keys for multi-site simulation', () => {
    const proxy = buildDevProxyConfig({ VITE_DEV_API_PREFIX: '/site01' })

    expect(Object.keys(proxy)).toContain('/site01/workspaces')
    expect(proxy['/site01/workspaces'].target).toBe('http://localhost:9621')
  })
})

describe('defaultApiEndpoints', () => {
  test('includes workspace routes', () => {
    expect(defaultApiEndpoints).toContain('/workspaces')
  })
})
