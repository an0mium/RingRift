#!/usr/bin/env python
"""Debug script for hex8 intermittent hanging bug.

This script runs hex8 selfplay games with detailed logging to identify
where the hang occurs during game execution.
"""

import logging
import os
import signal
import sys
import time
from pathlib import Path

# Add app/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable detailed engine logging
os.environ["RINGRIFT_DEBUG_ENGINE"] = "1"
os.environ["RINGRIFT_STRICT_NO_MOVE_INVARIANT"] = "1"

from app.ai.hybrid_gpu import create_hybrid_evaluator
from app.game_engine import GameEngine
from app.models import BoardType
from app.training.initial_state import create_initial_state

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/hex8_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    """Exception raised when a game times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Game timed out")


def run_single_game_with_timeout(game_idx: int, timeout_seconds: int = 30):
    """Run a single game with timeout and detailed logging.

    Args:
        game_idx: Game number for logging
        timeout_seconds: Timeout in seconds

    Returns:
        True if game completed, False if timeout
    """
    logger.info(f"=" * 60)
    logger.info(f"Starting game {game_idx}")
    logger.info(f"=" * 60)

    # Set up signal handler for timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        # Create initial state
        game_state = create_initial_state(
            board_type=BoardType.HEX8,
            num_players=2,
        )

        # Create evaluator
        evaluator = create_hybrid_evaluator(
            board_type="hex8",
            num_players=2,
            prefer_gpu=False,  # Use CPU to avoid GPU complications
        )

        move_count = 0
        max_moves = 500

        logger.info(f"Game {game_idx}: Initial state created")

        while game_state.game_status == "active" and move_count < max_moves:
            current_player = game_state.current_player
            phase = game_state.current_phase

            logger.debug(
                f"Game {game_idx} Move {move_count}: "
                f"Player {current_player}, Phase {phase}"
            )

            # Get valid moves with timeout tracking
            move_start = time.time()
            logger.debug(f"Game {game_idx}: Calling get_valid_moves...")
            valid_moves = GameEngine.get_valid_moves(game_state, current_player)
            move_time = time.time() - move_start

            if move_time > 1.0:
                logger.warning(
                    f"Game {game_idx}: get_valid_moves took {move_time:.2f}s "
                    f"(Phase: {phase}, Player: {current_player})"
                )

            logger.debug(
                f"Game {game_idx}: Found {len(valid_moves)} valid moves "
                f"in {move_time:.3f}s"
            )

            if not valid_moves:
                # Check for phase requirements
                logger.debug(f"Game {game_idx}: No valid moves, checking phase requirement...")
                req_start = time.time()
                requirement = GameEngine.get_phase_requirement(game_state, current_player)
                req_time = time.time() - req_start

                if req_time > 1.0:
                    logger.warning(
                        f"Game {game_idx}: get_phase_requirement took {req_time:.2f}s"
                    )

                if requirement is not None:
                    logger.debug(
                        f"Game {game_idx}: Phase requirement: {requirement.type}"
                    )
                    # Synthesize bookkeeping move
                    synth_start = time.time()
                    best_move = GameEngine.synthesize_bookkeeping_move(
                        requirement, game_state
                    )
                    synth_time = time.time() - synth_start

                    if synth_time > 1.0:
                        logger.warning(
                            f"Game {game_idx}: synthesize_bookkeeping_move took {synth_time:.2f}s"
                        )
                else:
                    logger.error(
                        f"Game {game_idx}: No valid moves and no phase requirement! "
                        f"Phase: {phase}, Player: {current_player}"
                    )
                    GameEngine._check_victory(game_state)
                    break
            else:
                # Select move based on heuristic
                eval_start = time.time()
                move_scores = evaluator.evaluate_moves(
                    game_state,
                    valid_moves,
                    current_player,
                    GameEngine,
                )
                eval_time = time.time() - eval_start

                if eval_time > 1.0:
                    logger.warning(
                        f"Game {game_idx}: evaluate_moves took {eval_time:.2f}s "
                        f"for {len(valid_moves)} moves"
                    )

                if move_scores:
                    best_score = max(s for _, s in move_scores)
                    best_moves = [m for m, s in move_scores if s == best_score]
                    best_move = best_moves[0]
                else:
                    best_move = valid_moves[0]

            # Apply move
            apply_start = time.time()
            logger.debug(f"Game {game_idx}: Applying move {best_move.type}...")
            game_state = GameEngine.apply_move(game_state, best_move)
            apply_time = time.time() - apply_start

            if apply_time > 1.0:
                logger.warning(
                    f"Game {game_idx}: apply_move took {apply_time:.2f}s "
                    f"(Move: {best_move.type})"
                )

            move_count += 1

            # Log progress every 10 moves
            if move_count % 10 == 0:
                logger.info(f"Game {game_idx}: Completed {move_count} moves")

        # Cancel alarm
        signal.alarm(0)

        # Log game completion
        winner = game_state.winner or 0
        logger.info(
            f"Game {game_idx} COMPLETED: "
            f"{move_count} moves, Winner: {winner}, Status: {game_state.game_status}"
        )

        # Clear move cache
        GameEngine.clear_cache()

        return True

    except TimeoutException:
        signal.alarm(0)  # Cancel alarm
        logger.error(
            f"Game {game_idx} TIMEOUT after {timeout_seconds}s at move {move_count}! "
            f"Phase: {game_state.current_phase}, Player: {game_state.current_player}"
        )

        # Log detailed state info
        logger.error(f"Board stacks: {len(game_state.board.stacks)}")
        logger.error(f"Board markers: {len(game_state.board.markers)}")
        logger.error(f"Collapsed spaces: {len(game_state.board.collapsed_spaces)}")

        # Try to get move state when it hung
        try:
            valid_moves = GameEngine.get_valid_moves(
                game_state, game_state.current_player
            )
            logger.error(f"Valid moves at timeout: {len(valid_moves)}")
        except Exception as e:
            logger.error(f"Could not get valid moves: {e}")

        GameEngine.clear_cache()
        return False

    except Exception as e:
        signal.alarm(0)  # Cancel alarm
        logger.exception(f"Game {game_idx} ERROR: {e}")
        GameEngine.clear_cache()
        return False


def main():
    """Run multiple hex8 games to reproduce the hanging bug."""
    num_games = 10
    timeout_seconds = 30

    logger.info("=" * 60)
    logger.info("HEX8 HANGING BUG DEBUG SESSION")
    logger.info("=" * 60)
    logger.info(f"Running {num_games} games with {timeout_seconds}s timeout each")
    logger.info("")

    completed = 0
    timeout_count = 0
    error_count = 0

    start_time = time.time()

    for i in range(num_games):
        result = run_single_game_with_timeout(i, timeout_seconds)

        if result:
            completed += 1
        else:
            timeout_count += 1

        logger.info("")

    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info("DEBUG SESSION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total games: {num_games}")
    logger.info(f"Completed: {completed}")
    logger.info(f"Timeouts: {timeout_count}")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Average time per game: {elapsed / num_games:.1f}s")
    logger.info("")
    logger.info(f"Log file: /tmp/hex8_debug.log")

    if timeout_count > 0:
        logger.error(f"HANGING BUG REPRODUCED: {timeout_count}/{num_games} games timed out")
        sys.exit(1)
    else:
        logger.info("No hangs detected in this run")
        sys.exit(0)


if __name__ == "__main__":
    main()
