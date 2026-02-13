# Dev stage: includes devDependencies for watch mode
FROM node:20-alpine AS dev

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
EXPOSE 3001
CMD ["npx", "nest", "start", "--watch"]

# Production build
FROM dev AS build
RUN npm run build

FROM node:20-alpine AS production
WORKDIR /app
COPY --from=build /app/dist ./dist
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/package.json ./
EXPOSE 3001
CMD ["node", "dist/main.js"]
