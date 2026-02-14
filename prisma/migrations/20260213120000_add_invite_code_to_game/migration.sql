-- AlterTable
ALTER TABLE "games" ADD COLUMN "inviteCode" TEXT;

-- CreateIndex
CREATE UNIQUE INDEX "games_inviteCode_key" ON "games"("inviteCode");
