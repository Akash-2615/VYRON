import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, forward API + image requests to the FastAPI service.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
      "/media": "http://localhost:8000",
    },
  },
});
