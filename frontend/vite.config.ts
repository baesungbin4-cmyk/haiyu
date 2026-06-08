import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  build: {
    chunkSizeWarningLimit: 550,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules/echarts")) {
            return "echarts";
          }
          if (id.includes("node_modules/react")) {
            return "react-vendor";
          }
          return undefined;
        },
      },
    },
  },
  plugins: [react()],
});
