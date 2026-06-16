import { describe, expect, test } from 'bun:test'
import { supportedFileTypes } from './constants'

function acceptsExtension(extension: string): boolean {
  return Object.values(supportedFileTypes).some((extensions) => extensions.includes(extension))
}

describe('supportedFileTypes', () => {
  test('accepts doc audio and video extensions', () => {
    for (const extension of [
      '.doc',
      '.mp4',
      '.avi',
      '.mov',
      '.wmv',
      '.flv',
      '.mkv',
      '.mp3',
      '.wav',
      '.m4a',
      '.aac',
      '.flac'
    ]) {
      expect(acceptsExtension(extension)).toBe(true)
    }
  })
})
