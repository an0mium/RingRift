-- AlterEnum
-- This migration adds more than one value to an enum.
-- With PostgreSQL versions 11 and earlier, this is not possible
-- in a single migration. This can be worked around by creating
-- multiple migrations, each migration adding only one value to
-- the enum.


ALTER TYPE "MoveType" ADD VALUE 'skip_placement';
ALTER TYPE "MoveType" ADD VALUE 'continue_capture_segment';
ALTER TYPE "MoveType" ADD VALUE 'process_line';
ALTER TYPE "MoveType" ADD VALUE 'choose_line_reward';
ALTER TYPE "MoveType" ADD VALUE 'process_territory_region';
ALTER TYPE "MoveType" ADD VALUE 'eliminate_rings_from_stack';

-- AlterTable
ALTER TABLE "games" ADD COLUMN     "finalState" JSONB;

-- AlterTable
ALTER TABLE "moves" ADD COLUMN     "moveData" JSONB;

-- AlterTable
ALTER TABLE "users" ADD COLUMN     "tokenVersion" INTEGER NOT NULL DEFAULT 0;

-- CreateIndex
CREATE INDEX "moves_gameId_timestamp_idx" ON "moves"("gameId", "timestamp");

-- CreateIndex
CREATE INDEX "users_deletedAt_idx" ON "users"("deletedAt");
