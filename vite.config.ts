import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  root: 'frontend',
  build: {
    outDir: '../dist/frontend',
    emptyOutDir: true,
  },
  base: './',
  server: {
    port: 3000,
  },
})
