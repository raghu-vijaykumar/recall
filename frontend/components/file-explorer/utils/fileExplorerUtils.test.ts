import { describe, it, expect } from 'vitest'
import { filterTreeBySearch } from './fileExplorerUtils'
import { FolderTreeNode } from '../../../src/core/types'

describe('filterTreeBySearch', () => {
  const sampleTree: FolderTreeNode[] = [
    {
      name: 'src',
      path: 'src',
      type: 'directory',
      children: [
        { name: 'App.tsx', path: 'src/App.tsx', type: 'file' },
        { name: 'index.ts', path: 'src/index.ts', type: 'file' },
        {
          name: 'components',
          path: 'src/components',
          type: 'directory',
          children: [
            { name: 'Button.tsx', path: 'src/components/Button.tsx', type: 'file' },
            { name: 'Header.ts', path: 'src/components/Header.ts', type: 'file' }
          ]
        }
      ]
    },
    { name: 'README.md', path: 'README.md', type: 'file' },
    { name: 'package.json', path: 'package.json', type: 'file' }
  ]

  it('returns all nodes when query is empty', () => {
    const result = filterTreeBySearch(sampleTree, '')
    expect(result).toEqual(sampleTree)
  })

  it('returns all nodes when query is whitespace', () => {
    const result = filterTreeBySearch(sampleTree, '   ')
    expect(result).toEqual(sampleTree)
  })

  it('filters files that match the query', () => {
    const result = filterTreeBySearch(sampleTree, 'tsx')
    expect(result).toEqual([
      {
        name: 'src',
        path: 'src',
        type: 'directory',
        children: [
          { name: 'App.tsx', path: 'src/App.tsx', type: 'file' },
          {
            name: 'components',
            path: 'src/components',
            type: 'directory',
            children: [
              { name: 'Button.tsx', path: 'src/components/Button.tsx', type: 'file' }
            ]
          }
        ]
      }
    ])
  })

  it('filters directories that match the query', () => {
    const result = filterTreeBySearch(sampleTree, 'components')
    expect(result).toEqual([
      {
        name: 'src',
        path: 'src',
        type: 'directory',
        children: [
          {
            name: 'components',
            path: 'src/components',
            type: 'directory',
            children: []
          }
        ]
      }
    ])
  })

  it('returns empty array when no matches', () => {
    const result = filterTreeBySearch(sampleTree, 'nonexistent')
    expect(result).toEqual([])
  })

  it('is case insensitive', () => {
    const result = filterTreeBySearch(sampleTree, 'README')
    expect(result).toEqual([{ name: 'README.md', path: 'README.md', type: 'file' }])
  })

  it('filters nested directories correctly', () => {
    const result = filterTreeBySearch(sampleTree, 'header')
    expect(result).toEqual([
      {
        name: 'src',
        path: 'src',
        type: 'directory',
        children: [
          {
            name: 'components',
            path: 'src/components',
            type: 'directory',
            children: [
              { name: 'Header.ts', path: 'src/components/Header.ts', type: 'file' }
            ]
          }
        ]
      }
    ])
  })

  it('includes directory if it matches the query', () => {
    const result = filterTreeBySearch(sampleTree, 'src')
    expect(result).toEqual([
      {
        name: 'src',
        path: 'src',
        type: 'directory',
        children: []
      }
    ])
  })
})
