import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base must match your GitHub repo name for GitHub Pages to resolve assets.
export default defineConfig({
  plugins: [react()],
  base: '/NodeSense/',
  server: {
    proxy: {
      // Local dev proxy so fetch('/api/...') hits the FastAPI server
      '/api': {
        target: 'http://localhost:7860',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
