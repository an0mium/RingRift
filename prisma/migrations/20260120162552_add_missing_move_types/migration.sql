-- AlterEnum
-- This migration adds more than one value to an enum.
-- With PostgreSQL versions 11 and earlier, this is not possible
-- in a single migration. This can be worked around by creating
-- multiple migrations, each migration adding only one value to
-- the enum.


ALTER TYPE "MoveType" ADD VALUE 'no_placement_action';
ALTER TYPE "MoveType" ADD VALUE 'no_movement_action';
ALTER TYPE "MoveType" ADD VALUE 'skip_capture';
ALTER TYPE "MoveType" ADD VALUE 'choose_line_option';
ALTER TYPE "MoveType" ADD VALUE 'no_line_action';
ALTER TYPE "MoveType" ADD VALUE 'choose_territory_option';
ALTER TYPE "MoveType" ADD VALUE 'skip_territory_processing';
ALTER TYPE "MoveType" ADD VALUE 'no_territory_action';
ALTER TYPE "MoveType" ADD VALUE 'skip_recovery';
ALTER TYPE "MoveType" ADD VALUE 'forced_elimination';
ALTER TYPE "MoveType" ADD VALUE 'swap_sides';
ALTER TYPE "MoveType" ADD VALUE 'resign';
ALTER TYPE "MoveType" ADD VALUE 'timeout';
