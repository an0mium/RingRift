-- AlterEnum
ALTER TYPE "BoardType" ADD VALUE 'hex8';

-- AlterEnum
ALTER TYPE "MoveType" ADD VALUE 'recovery_slide';

-- AlterTable
ALTER TABLE "refresh_tokens" ADD COLUMN     "rememberMe" BOOLEAN NOT NULL DEFAULT false;

-- CreateTable
CREATE TABLE "rating_history" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "gameId" TEXT,
    "oldRating" INTEGER NOT NULL,
    "newRating" INTEGER NOT NULL,
    "change" INTEGER NOT NULL,
    "timestamp" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "rating_history_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "rating_history_userId_timestamp_idx" ON "rating_history"("userId", "timestamp");

-- CreateIndex
CREATE INDEX "rating_history_gameId_idx" ON "rating_history"("gameId");

-- AddForeignKey
ALTER TABLE "rating_history" ADD CONSTRAINT "rating_history_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
