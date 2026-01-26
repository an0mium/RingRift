-- CreateIndex
CREATE INDEX "games_status_idx" ON "games"("status");

-- CreateIndex
CREATE INDEX "games_createdAt_idx" ON "games"("createdAt");

-- CreateIndex
CREATE INDEX "games_status_createdAt_idx" ON "games"("status", "createdAt");
