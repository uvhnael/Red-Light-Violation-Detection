# Next.js Web Dashboard Dockerfile
FROM node:20-alpine AS base

# Install dependencies only when needed
FROM base AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_TELEMETRY_DISABLED 1
# Rewrites destinations are baked at build time (next.config.ts reads these).
# Point them at the Docker-network service names so the server-side proxy works.
ARG CENTRAL_SERVER_URL=http://central-server:8000
ARG EDGE_SERVER_URL=http://edge-pipeline:8080
ENV CENTRAL_SERVER_URL=${CENTRAL_SERVER_URL}
ENV EDGE_SERVER_URL=${EDGE_SERVER_URL}
RUN npm run build

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production
ENV NEXT_TELEMETRY_DISABLED 1
ENV TZ Asia/Ho_Chi_Minh

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/package.json ./package.json

USER nextjs

EXPOSE 3000

ENV PORT 3000
ENV HOSTNAME "0.0.0.0"

CMD ["node", "server.js"]
