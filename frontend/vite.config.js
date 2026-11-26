import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import process from 'node:process'
import { forwardApiResponse } from './apiProxy.js'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: process.env.WAYCON_API_PROXY || 'http://127.0.0.1:8002',
        changeOrigin: true,
        selfHandleResponse: true,
        configure(proxy) {
          proxy.on('proxyRes', (upstream, request, response) => forwardApiResponse(upstream, response))
        },
      },
    },
  },
})
