import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev mode: /api is proxied to the FastAPI server (uv run uvicorn ... --port 8080).
// In Docker, nginx does the same proxying, so the app always talks to a same-origin /api.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: process.env.VITE_API_PROXY ?? 'http://127.0.0.1:8080', changeOrigin: true },
    },
  },
})
