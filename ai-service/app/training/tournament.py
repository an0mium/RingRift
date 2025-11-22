"""
Tournament system for evaluating AI models
"""

import sys
import os
import logging
import torch
from typing import Dict, Optional
from datetime import datetime

from app.ai.descent_ai import DescentAI
from app.game_engine import GameEngine
from app.models import (
    GameState, BoardType, BoardState, GamePhase, GameStatus, TimeControl,
    Player, AIConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Tournament:
    def __init__(
        self,
        model_path_a: str,
        model_path_b: str,
        num_games: int = 20,
        k_elo: int = 32,
    ):
        self.model_path_a = model_path_a
        self.model_path_b = model_path_b
        self.num_games = num_games
        self.results = {"A": 0, "B": 0, "Draw": 0}
        # Simple Elo-like rating system for candidate (A) vs best (B).
        self.k_elo = k_elo
        self.ratings = {"A": 1500.0, "B": 1500.0}
        
    def _create_ai(self, player_number: int, model_path: str) -> DescentAI:
        """Create an AI instance with specific model weights.
        
        The checkpoint basename (without .pth) is treated as the nn_model_id so
        that NeuralNetAI can load it via AIConfig.nn_model_id. We keep the
        manual load as a fallback in case older DescentAI/NeuralNetAI versions
        ignore nn_model_id.
        """
        model_id = os.path.splitext(os.path.basename(model_path))[0]
        config = AIConfig(
            difficulty=10,
            randomness=0.1,
            think_time=500,
            rngSeed=None,
            nn_model_id=model_id,
        )
        ai = DescentAI(player_number, config)

        # Fallback/manual load for robustness with legacy implementations.
        if ai.neural_net and os.path.exists(model_path):
            try:
                ai.neural_net.model.load_state_dict(
                    torch.load(model_path, weights_only=True)
                )
                ai.neural_net.model.eval()
            except Exception as e:
                logger.error(f"Failed to load model {model_path}: {e}")
        
        return ai

    def run(self) -> Dict[str, int]:
        """Run the tournament"""
        logger.info(
            "Starting tournament: %s vs %s",
            self.model_path_a,
            self.model_path_b,
        )
        
        for i in range(self.num_games):
            # Alternate colors
            if i % 2 == 0:
                p1_model = self.model_path_a
                p2_model = self.model_path_b
                p1_label = "A"
                p2_label = "B"
            else:
                p1_model = self.model_path_b
                p2_model = self.model_path_a
                p1_label = "B"
                p2_label = "A"
                
            ai1 = self._create_ai(1, p1_model)
            ai2 = self._create_ai(2, p2_model)
            
            winner = self._play_game(ai1, ai2)
            
            if winner == 1:
                self.results[p1_label] += 1
                self._update_elo(p1_label)
            elif winner == 2:
                self.results[p2_label] += 1
                self._update_elo(p2_label)
            else:
                self.results["Draw"] += 1
                self._update_elo(None)
                
            if winner == 1:
                winner_label_str = p1_label
            elif winner == 2:
                winner_label_str = p2_label
            else:
                winner_label_str = "Draw"
            logger.info(
                "Game %d/%d: Winner %s (%s)",
                i + 1,
                self.num_games,
                winner,
                winner_label_str,
            )
            
        logger.info("Tournament finished. Results: %s", self.results)
        logger.info(
            "Final Elo ratings: A=%.1f, B=%.1f",
            self.ratings["A"],
            self.ratings["B"],
        )
        return self.results

    def _play_game(self, ai1: DescentAI, ai2: DescentAI) -> Optional[int]:
        """Play a single game"""
        # Initialize game state
        state = self._create_initial_state()
        move_count = 0
        
        while state.game_status == GameStatus.ACTIVE and move_count < 200:
            current_player = state.current_player
            ai = ai1 if current_player == 1 else ai2
            
            move = ai.select_move(state)
            
            if not move:
                # No moves available, current player loses
                state.winner = 2 if current_player == 1 else 1
                state.game_status = GameStatus.FINISHED
                break
                
            state = GameEngine.apply_move(state, move)
            move_count += 1
            
        return state.winner

    def _update_elo(self, winner_label: Optional[str]) -> None:
        """Update Elo-like ratings for candidate (A) and best (B)."""
        ra = self.ratings["A"]
        rb = self.ratings["B"]
        # Expected scores
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        eb = 1.0 - ea

        if winner_label == "A":
            sa, sb = 1.0, 0.0
        elif winner_label == "B":
            sa, sb = 0.0, 1.0
        else:
            # Draw
            sa = sb = 0.5

        self.ratings["A"] = ra + self.k_elo * (sa - ea)
        self.ratings["B"] = rb + self.k_elo * (sb - eb)

    def _create_initial_state(self) -> GameState:
        """Create initial game state"""
        # Simplified version of generate_data.create_initial_state
        size = 8
        rings = 18
        return GameState(
            id="tournament",
            boardType=BoardType.SQUARE8,
            rngSeed=None,
            board=BoardState(
                type=BoardType.SQUARE8,
                size=size,
                stacks={},
                markers={},
                collapsedSpaces={},
                eliminatedRings={},
            ),
            players=[
                Player(
                    id="p1",
                    username="AI 1",
                    type="ai",
                    playerNumber=1,
                    isReady=True,
                    timeRemaining=600,
                    ringsInHand=rings,
                    eliminatedRings=0,
                    territorySpaces=0,
                    aiDifficulty=10,
                ),
                Player(
                    id="p2",
                    username="AI 2",
                    type="ai",
                    playerNumber=2,
                    isReady=True,
                    timeRemaining=600,
                    ringsInHand=rings,
                    eliminatedRings=0,
                    territorySpaces=0,
                    aiDifficulty=10,
                ),
            ],
            currentPhase=GamePhase.RING_PLACEMENT,
            currentPlayer=1,
            moveHistory=[],
            timeControl=TimeControl(
                initialTime=600,
                increment=0,
                type="blitz",
            ),
            gameStatus=GameStatus.ACTIVE,
            createdAt=datetime.now(),
            lastMoveAt=datetime.now(),
            isRated=False,
            maxPlayers=2,
            totalRingsInPlay=rings * 2,
            totalRingsEliminated=0,
            victoryThreshold=19,
            territoryVictoryThreshold=33,
            chainCaptureState=None,
            mustMoveFromStackKey=None,
            zobristHash=None,
        )


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        t = Tournament(sys.argv[1], sys.argv[2])
        t.run()
    else:
        print("Usage: python tournament.py <model_a_path> <model_b_path>")