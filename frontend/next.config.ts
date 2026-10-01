import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emits .next/standalone with only the files needed to run, so the Docker
  // image ships a few MB of server code instead of the whole node_modules tree.
  output: "standalone",
  // The backend is reached through app/api/query/route.ts, which reads
  // BACKEND_URL at request time. A rewrite here would bake the URL in at build
  // time, which is wrong inside a container that only learns it from the ConfigMap.
};

export default nextConfig;
