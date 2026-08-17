import type { NextConfig } from "next";

const centralServerUrl = process.env.CENTRAL_SERVER_URL || 'http://localhost:8001';
const edgeServerUrl = process.env.EDGE_SERVER_URL || 'http://localhost:8080';

const nextConfig: NextConfig = {
  output: 'standalone',
  images: {
    remotePatterns: [
      { protocol: 'http', hostname: 'localhost', port: '9002' },
      { protocol: 'http', hostname: 'localhost', port: '9001' },
      { protocol: 'http', hostname: 'minio', port: '9000' },
      { protocol: 'http', hostname: 'minio', port: '9001' },
      { protocol: 'https', hostname: '**' },
      { protocol: 'http', hostname: '**' },
    ],
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${centralServerUrl}/api/:path*`,
      },
      {
        source: '/api/v1/:path*',
        destination: `${centralServerUrl}/api/v1/:path*`,
      },
      {
        source: '/edge-api/:path*',
        destination: `${edgeServerUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
