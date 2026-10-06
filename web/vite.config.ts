import { resolve } from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const nm = (p: string) => resolve(__dirname, "node_modules", p);
const api = "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: [
      { find: "@static", replacement: resolve(__dirname, "../src/isharati/app/static") },
      { find: /^three\/addons\/(.*)$/, replacement: nm("three/examples/jsm/$1") },
      { find: /^three$/, replacement: nm("three/build/three.module.js") },
      { find: /^@pixiv\/three-vrm$/, replacement: nm("@pixiv/three-vrm/lib/three-vrm.module.js") },
    ],
  },
  server: {
    fs: { allow: [".."] },
    proxy: { "/api": api, "/pose": api, "/video": api, "/static": api },
  },
  build: {
    rollupOptions: { input: { main: resolve(__dirname, "index.html"), record: resolve(__dirname, "record.html") } },
  },
  test: { environment: "jsdom", setupFiles: ["./src/test-setup.ts"], globals: true },
});
