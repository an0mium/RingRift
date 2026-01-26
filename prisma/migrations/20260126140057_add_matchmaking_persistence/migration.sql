-- CreateEnum
CREATE TYPE "MatchmakingQueueStatus" AS ENUM ('searching', 'matching', 'matched', 'cancelled', 'expired');

-- CreateEnum
CREATE TYPE "MatchmakingOutcome" AS ENUM ('matched', 'cancelled_user', 'cancelled_disconnect', 'expired');

-- CreateTable
CREATE TABLE "matchmaking_queue" (
    "id" TEXT NOT NULL,
    "ticketId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "boardType" "BoardType" NOT NULL,
    "ratingRangeMin" INTEGER NOT NULL,
    "ratingRangeMax" INTEGER NOT NULL,
    "timeControlMin" INTEGER NOT NULL,
    "timeControlMax" INTEGER NOT NULL,
    "status" "MatchmakingQueueStatus" NOT NULL DEFAULT 'searching',
    "rating" INTEGER NOT NULL,
    "joinedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "matchedAt" TIMESTAMP(3),
    "cancelledAt" TIMESTAMP(3),
    "gameId" TEXT,
    "matchedWithId" TEXT,
    "serverId" TEXT,
    "lastHeartbeat" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "matchmaking_queue_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "matchmaking_metrics" (
    "id" TEXT NOT NULL,
    "ticketId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "boardType" "BoardType" NOT NULL,
    "rating" INTEGER NOT NULL,
    "joinedAt" TIMESTAMP(3) NOT NULL,
    "matchedAt" TIMESTAMP(3),
    "waitTimeMs" INTEGER,
    "ratingDiff" INTEGER,
    "matchQualityScore" DOUBLE PRECISION,
    "outcome" "MatchmakingOutcome" NOT NULL,
    "queueSizeAtJoin" INTEGER,
    "queueSizeAtMatch" INTEGER,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "matchmaking_metrics_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "matchmaking_queue_ticketId_key" ON "matchmaking_queue"("ticketId");

-- CreateIndex
CREATE INDEX "matchmaking_queue_status_boardType_idx" ON "matchmaking_queue"("status", "boardType");

-- CreateIndex
CREATE INDEX "matchmaking_queue_status_rating_idx" ON "matchmaking_queue"("status", "rating");

-- CreateIndex
CREATE INDEX "matchmaking_queue_joinedAt_idx" ON "matchmaking_queue"("joinedAt");

-- CreateIndex
CREATE UNIQUE INDEX "matchmaking_queue_userId_status_key" ON "matchmaking_queue"("userId", "status");

-- CreateIndex
CREATE INDEX "matchmaking_metrics_userId_createdAt_idx" ON "matchmaking_metrics"("userId", "createdAt");

-- CreateIndex
CREATE INDEX "matchmaking_metrics_boardType_createdAt_idx" ON "matchmaking_metrics"("boardType", "createdAt");

-- CreateIndex
CREATE INDEX "matchmaking_metrics_outcome_createdAt_idx" ON "matchmaking_metrics"("outcome", "createdAt");

-- AddForeignKey
ALTER TABLE "matchmaking_queue" ADD CONSTRAINT "matchmaking_queue_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
