import process from 'node:process'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Port and backend target are configurable via env so the dev server and the
// FastAPI backend can be moved off the default ports together, e.g.
//   VITE_PORT=4321 VITE_PROXY_TARGET=http://localhost:8010 npm run dev
const port = Number(process.env.VITE_PORT) || 5173
const proxyTarget = process.env.VITE_PROXY_TARGET || 'http://localhost:8000'

export default defineConfig({
  base: '/',
  plugins: [vue()],
  server: {
    port,
    proxy: {
      // any request to /image/* will go to your FastAPI server
      '/image': {
        target: proxyTarget,
        changeOrigin: true,
        secure: false,
      },
    },
  },
})
