# Load Test Results

Last run: December 25, 2025

## Environment Setup

### Required Configuration (.env)

```bash
# Database (local postgres or docker container)
DATABASE_URL=postgresql://ringrift:password@localhost:5432/ringrift

# Redis (local or docker container)
REDIS_URL=redis://localhost:6379

# JWT Authentication
JWT_SECRET=dev-jwt-secret-for-local-testing-only-32chars
JWT_REFRESH_SECRET=dev-refresh-secret-for-local-testing-32chars
JWT_EXPIRES_IN=15m
JWT_REFRESH_EXPIRES_IN=7d

# Load Testing Configuration
LOADTEST_USER_POOL_SIZE=50
LOADTEST_USER_POOL_PASSWORD=TestPassword123!

# Rate Limits for Load Testing (relaxed for k6)
RATE_LIMIT_API_POINTS=100000
RATE_LIMIT_API_AUTH_POINTS=100000
RATE_LIMIT_AUTH_POINTS=100000
RATE_LIMIT_AUTH_LOGIN_POINTS=100000
RATE_LIMIT_GAME_POINTS=100000
RATE_LIMIT_GAME_MOVES_POINTS=100000
RATE_LIMIT_GAME_CREATE_USER_POINTS=100000
RATE_LIMIT_GAME_CREATE_IP_POINTS=100000
RATE_LIMIT_WEBSOCKET_POINTS=100000
```

### Pre-requisites

1. **Database Setup**

   ```bash
   # Start postgres (docker or local)
   docker compose up -d postgres

   # Run migrations
   npx prisma migrate deploy
   ```

2. **Redis Setup**

   ```bash
   docker compose up -d redis
   ```

3. **Seed Load Test Users**

   ```bash
   # Seeds 400 users with default password LoadTestK6Pass123
   node scripts/seed-loadtest-users.js

   # Or update existing users' passwords:
   DATABASE_URL='postgresql://...' node -e "
   const { PrismaClient } = require('@prisma/client');
   const bcrypt = require('bcryptjs');
   const prisma = new PrismaClient();
   async function main() {
     const hash = await bcrypt.hash('LoadTestK6Pass123', 12);
     const result = await prisma.user.updateMany({
       where: { email: { contains: 'loadtest' } },
       data: { passwordHash: hash }
     });
     console.log('Updated', result.count, 'users');
   }
   main().finally(() => prisma.\$disconnect());
   "
   ```

4. **Start Server**

   ```bash
   TS_NODE_PROJECT=tsconfig.server.json npx ts-node -T src/server/index.ts
   ```

5. **Flush Redis (before each test)**
   ```bash
   docker exec ringrift-redis-1 redis-cli FLUSHALL
   ```

## Running Load Tests

```bash
# Game creation scenario (4 minutes, 50 VUs)
BASE_URL=http://localhost:3000 npx k6 run \
  -e LOADTEST_USER_POOL_SIZE=400 \
  tests/load/scenarios/game-creation.js
```

## Test Results: Game Creation Scenario

### Summary (Dec 25, 2025)

| Metric            | Value         | Status   |
| ----------------- | ------------- | -------- |
| Duration          | 4 minutes     | -        |
| VUs (max)         | 50            | -        |
| Total Iterations  | 9,718         | -        |
| Games Created     | 1,199 (12.3%) | Note 1   |
| Contract Failures | 0             | PASS     |
| True Errors       | 0             | PASS     |
| Rate Limit Hits   | 8,538         | Expected |

**Note 1**: After ~650 iterations (~1.5 minutes), the server's adaptive rate limiter
kicks in to protect against DoS. This is expected and correct behavior.

### Performance Metrics

| Metric                | Avg   | Median | P90   | P95   | Max   |
| --------------------- | ----- | ------ | ----- | ----- | ----- |
| Game Creation Latency | 37ms  | 2ms    | 50ms  | 230ms | 1.27s |
| Get Game Latency      | 262ms | 127ms  | 716ms | 787ms | 1.15s |
| HTTP Request Duration | 77ms  | 2.6ms  | 183ms | 529ms | 6.7s  |

### Thresholds

| Threshold                      | Result | Target |
| ------------------------------ | ------ | ------ |
| contract_failures_total        | 0      | 0      |
| true_errors_total              | 0      | 0      |
| id_lifecycle_mismatches_total  | 0      | 0      |
| game_creation_latency_ms (p95) | 230ms  | <500ms |
| Login successful               | 100%   | 100%   |
| Access token present           | 100%   | 100%   |

### Key Observations

1. **Authentication works correctly** - All logins succeed, tokens are valid
2. **Game creation is fast** - 37ms average, 230ms p95 when not rate-limited
3. **No data integrity issues** - Zero contract failures, zero lifecycle mismatches
4. **Rate limiting is working** - Server protects itself under sustained load
5. **Throughput** - ~45 requests/second, ~40 iterations/second

### Rate Limiting Behavior

The server has multiple rate limiting layers:

- Per-user API rate limits (configurable via env)
- Per-IP game creation limits (configurable via env)
- Adaptive rate limiting (protects against sustained high load)

Under the 50 VU load test, the adaptive rate limiter activates after ~650 iterations
(~1.5 minutes) which is expected protective behavior.

## Password Configuration Notes

**Important**: The k6 load test helpers use different env var names than the seed script:

| Component   | Env Var                       | Default             |
| ----------- | ----------------------------- | ------------------- |
| k6 helpers  | `LOADTEST_USER_POOL_PASSWORD` | `LoadTestK6Pass123` |
| Seed script | `LOADTEST_USER_PASSWORD`      | `TestPassword123!`  |

Ensure passwords match by either:

1. Seeding users with `LOADTEST_USER_PASSWORD=LoadTestK6Pass123`
2. Or updating users after seeding (see Pre-requisites section)

## Troubleshooting

### 401 AUTH_INVALID_CREDENTIALS

- Password mismatch between k6 and seeded users
- Solution: Update user passwords or reseed with correct password

### 429 Rate Limiting Errors Early in Test

- Rate limits from previous runs persisted in Redis
- Solution: `docker exec ringrift-redis-1 redis-cli FLUSHALL`

### Prisma Migrations Not Applied

- Tables don't exist in database
- Solution: `npx prisma migrate deploy`

### Server Not Picking Up Env Vars

- Server was started before .env was updated
- Solution: Restart server after .env changes
