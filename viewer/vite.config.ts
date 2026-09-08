import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
// base: '/r9700-llm-bench/' - GitHub Pages project sites serve from
// https://<user>.github.io/<repo>/, not the domain root, so every asset
// URL needs that prefix. Override at build time for local previews if
// hosting elsewhere: `vite build --base=/`.
export default defineConfig({
  plugins: [react()],
  base: '/r9700-llm-bench/',
})
