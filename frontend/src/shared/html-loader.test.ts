import { describe, it, expect, vi, beforeEach } from 'vitest'
import { HtmlLoader } from './html-loader'

// Mock fetch globally
global.fetch = vi.fn()

describe('HtmlLoader', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Clear cache
    HtmlLoader['componentCache'].clear()
  })

  describe('loadComponent', () => {
    it('loads component from fetch and caches it', async () => {
      const mockHtml = '<div>Test Component</div>'
      ;(global.fetch as any).mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(mockHtml)
      })

      const result = await HtmlLoader.loadComponent('test')

      expect(global.fetch).toHaveBeenCalledWith('./components/test/test.html')
      expect(result).toBe(mockHtml)
    })

    it('returns cached component on second call', async () => {
      const mockHtml = '<div>Cached Component</div>'
      ;(global.fetch as any).mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(mockHtml)
      })

      await HtmlLoader.loadComponent('cached')
      const result = await HtmlLoader.loadComponent('cached')

      expect(global.fetch).toHaveBeenCalledTimes(1)
      expect(result).toBe(mockHtml)
    })

    it('returns error message on fetch failure', async () => {
      ;(global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 404
      })

      const result = await HtmlLoader.loadComponent('missing')

      expect(result).toBe('<div class="error">Failed to load missing component</div>')
    })

    it('returns error message on fetch error', async () => {
      ;(global.fetch as any).mockRejectedValueOnce(new Error('Network error'))

      const result = await HtmlLoader.loadComponent('error')

      expect(result).toBe('<div class="error">Failed to load error component</div>')
    })
  })

  describe('loadAllComponents', () => {
    it('loads all components and inserts into DOM', async () => {
      const mockHtmls = ['<div>Workspaces</div>', '<div>File Explorer</div>', '<div>Quiz</div>', '<div>Progress</div>']
      mockHtmls.forEach(html => {
        ;(global.fetch as any).mockResolvedValueOnce({
          ok: true,
          text: () => Promise.resolve(html)
        })
      })

      // Mock document methods
      const mockContainers = mockHtmls.map(() => ({ innerHTML: '' } as HTMLElement))
      document.getElementById = vi.fn((id) => {
        const index = ['workspaces-container', 'file-explorer-container', 'quiz-container', 'progress-container'].indexOf(id)
        return index >= 0 ? mockContainers[index] : null
      })

      await HtmlLoader.loadAllComponents()

      expect(global.fetch).toHaveBeenCalledTimes(4)
      expect(mockContainers[0].innerHTML).toBe('<div>Workspaces</div>')
      expect(mockContainers[1].innerHTML).toBe('<div>File Explorer</div>')
      expect(mockContainers[2].innerHTML).toBe('<div>Quiz</div>')
      expect(mockContainers[3].innerHTML).toBe('<div>Progress</div>')
    })

    it('handles errors in loading components', async () => {
      ;(global.fetch as any).mockRejectedValue(new Error('Error'))

      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})

      await HtmlLoader.loadAllComponents()

      expect(consoleSpy).toHaveBeenCalledTimes(4)
      expect(consoleSpy).toHaveBeenCalledWith('Error loading component workspaces:', expect.any(Error))
      consoleSpy.mockRestore()
    })
  })
})
