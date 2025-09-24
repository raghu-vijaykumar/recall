import { describe, it, expect, vi, beforeEach } from 'vitest'
import { ApiService, API_BASE } from './api'

// Mock fetch globally
global.fetch = vi.fn()

describe('ApiService', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('API_BASE', () => {
    it('has correct base URL', () => {
      expect(API_BASE).toBe('http://127.0.0.1:8000/api')
    })
  })

  describe('get', () => {
    it('makes GET request and returns JSON', async () => {
      const mockData = { id: 1, name: 'test' }
      ;(global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockData)
      })

      const result = await ApiService.get('/test')

      expect(global.fetch).toHaveBeenCalledWith(`${API_BASE}/test`)
      expect(result).toEqual(mockData)
    })

    it('throws error on failed response', async () => {
      ;(global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found'
      })

      await expect(ApiService.get('/missing')).rejects.toThrow('API Error: 404 Not Found')
    })
  })

  describe('post', () => {
    it('makes POST request with data and returns JSON', async () => {
      const mockData = { id: 2, name: 'created' }
      const postData = { name: 'new item' }
      ;(global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockData)
      })

      const result = await ApiService.post('/create', postData)

      expect(global.fetch).toHaveBeenCalledWith(`${API_BASE}/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(postData)
      })
      expect(result).toEqual(mockData)
    })

    it('throws error on failed response', async () => {
      ;(global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error'
      })

      await expect(ApiService.post('/create', {})).rejects.toThrow('API Error: 500 Internal Server Error')
    })
  })

  describe('put', () => {
    it('makes PUT request with data and returns JSON', async () => {
      const mockData = { id: 1, name: 'updated' }
      const putData = { name: 'updated item' }
      ;(global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockData)
      })

      const result = await ApiService.put('/update/1', putData)

      expect(global.fetch).toHaveBeenCalledWith(`${API_BASE}/update/1`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(putData)
      })
      expect(result).toEqual(mockData)
    })

    it('throws error on failed response', async () => {
      ;(global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 403,
        statusText: 'Forbidden'
      })

      await expect(ApiService.put('/update/1', {})).rejects.toThrow('API Error: 403 Forbidden')
    })
  })

  describe('delete', () => {
    it('makes DELETE request and returns JSON', async () => {
      const mockData = { deleted: true }
      ;(global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockData)
      })

      const result = await ApiService.delete('/delete/1')

      expect(global.fetch).toHaveBeenCalledWith(`${API_BASE}/delete/1`, {
        method: 'DELETE'
      })
      expect(result).toEqual(mockData)
    })

    it('throws error on failed response', async () => {
      ;(global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found'
      })

      await expect(ApiService.delete('/delete/1')).rejects.toThrow('API Error: 404 Not Found')
    })
  })
})
