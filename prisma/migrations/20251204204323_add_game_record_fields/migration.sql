-- CreateEnum
CREATE TYPE "RematchRequestStatus" AS ENUM ('pending', 'accepted', 'declined', 'expired');

-- AlterTable
ALTER TABLE "games" ADD COLUMN     "finalScore" JSONB,
ADD COLUMN     "outcome" TEXT,
ADD COLUMN     "recordMetadata" JSONB;

-- CreateTable
CREATE TABLE "chat_messages" (
    "id" TEXT NOT NULL,
    "gameId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "message" VARCHAR(500) NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "chat_messages_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "rematch_requests" (
    "id" TEXT NOT NULL,
    "gameId" TEXT NOT NULL,
    "requesterId" TEXT NOT NULL,
    "status" "RematchRequestStatus" NOT NULL DEFAULT 'pending',
    "expiresAt" TIMESTAMP(3) NOT NULL,
    "respondedAt" TIMESTAMP(3),
    "newGameId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "rematch_requests_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "chat_messages_gameId_createdAt_idx" ON "chat_messages"("gameId", "createdAt");

-- CreateIndex
CREATE INDEX "rematch_requests_gameId_status_idx" ON "rematch_requests"("gameId", "status");

-- CreateIndex
CREATE UNIQUE INDEX "rematch_requests_gameId_status_key" ON "rematch_requests"("gameId", "status");

-- AddForeignKey
ALTER TABLE "chat_messages" ADD CONSTRAINT "chat_messages_gameId_fkey" FOREIGN KEY ("gameId") REFERENCES "games"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "chat_messages" ADD CONSTRAINT "chat_messages_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "rematch_requests" ADD CONSTRAINT "rematch_requests_gameId_fkey" FOREIGN KEY ("gameId") REFERENCES "games"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "rematch_requests" ADD CONSTRAINT "rematch_requests_requesterId_fkey" FOREIGN KEY ("requesterId") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
