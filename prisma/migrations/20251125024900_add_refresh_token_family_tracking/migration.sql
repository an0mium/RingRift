-- AlterTable: Add token family tracking fields for refresh token rotation security
-- familyId: Groups all tokens in a rotation chain so they can be revoked together
-- revokedAt: Marks revoked tokens so we can detect reuse attacks

ALTER TABLE "refresh_tokens" ADD COLUMN "familyId" TEXT;
ALTER TABLE "refresh_tokens" ADD COLUMN "revokedAt" TIMESTAMP(3);

-- CreateIndex: Index on familyId for efficient family lookups during revocation
CREATE INDEX "refresh_tokens_familyId_idx" ON "refresh_tokens"("familyId");