import type { ProxyOptions } from 'vite'
import { normalizeApiPrefix } from './pathPrefix'

export const defaultApiEndpoints = [
  '/api',
  '/documents',
  '/graphs',
  '/graph',
  '/health',
  '/query',
  '/workspaces',
  '/docs',
  '/redoc',
  '/openapi.json',
  '/login',
  '/auth-status',
  '/static'
]

const parseApiEndpoints = (value: string | undefined): string[] => {
  if (!value?.trim()) return defaultApiEndpoints
  return value
    .split(',')
    .map((endpoint) => endpoint.trim())
    .filter(Boolean)
}

export const buildDevProxyConfig = (
  env: Record<string, string | undefined>
): Record<string, ProxyOptions> => {
  if (env.VITE_API_PROXY === 'false') return {}

  const devApiPrefix = normalizeApiPrefix(env.VITE_DEV_API_PREFIX)
  const target = env.VITE_BACKEND_URL || 'http://localhost:9621'

  return Object.fromEntries(
    parseApiEndpoints(env.VITE_API_ENDPOINTS).map((endpoint) => [
      devApiPrefix + endpoint,
      {
        target,
        changeOrigin: true
      }
    ])
  )
}
