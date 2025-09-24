import { describe, it, expect } from 'vitest'
import {
  getFileIcon,
  getFileTypeFromName,
  formatFileSize,
  generateId,
  createElementFromHTML
} from './utils'

describe('getFileIcon', () => {
  it('returns correct icon for Python files', () => {
    expect(getFileIcon('test.py')).toBe('🐍')
  })

  it('returns correct icon for TypeScript files', () => {
    expect(getFileIcon('test.ts')).toBe('🔷')
  })

  it('returns default icon for unknown extensions', () => {
    expect(getFileIcon('test.unknown')).toBe('📄')
  })

  it('handles files without extension', () => {
    expect(getFileIcon('README')).toBe('📄')
  })
})

describe('getFileTypeFromName', () => {
  it('returns "code" for Python files', () => {
    expect(getFileTypeFromName('test.py')).toBe('code')
  })

  it('returns "markdown" for markdown files', () => {
    expect(getFileTypeFromName('test.md')).toBe('markdown')
  })

  it('returns "text" for unknown extensions', () => {
    expect(getFileTypeFromName('test.unknown')).toBe('text')
  })
})

describe('formatFileSize', () => {
  it('formats bytes correctly', () => {
    expect(formatFileSize(0)).toBe('0 Bytes')
    expect(formatFileSize(1023)).toBe('1023 Bytes')
    expect(formatFileSize(1024)).toBe('1 KB')
    expect(formatFileSize(1536)).toBe('1.5 KB')
    expect(formatFileSize(1048576)).toBe('1 MB')
  })
})

describe('generateId', () => {
  it('generates a string id', () => {
    const id = generateId()
    expect(typeof id).toBe('string')
    expect(id.length).toBeGreaterThan(0)
  })

  it('generates unique ids', () => {
    const id1 = generateId()
    const id2 = generateId()
    expect(id1).not.toBe(id2)
  })
})


describe('createElementFromHTML', () => {
  it('creates element from HTML string', () => {
    const element = createElementFromHTML('<div class="test">Hello</div>')
    expect(element.tagName).toBe('DIV')
    expect(element.className).toBe('test')
    expect(element.textContent).toBe('Hello')
  })
})
