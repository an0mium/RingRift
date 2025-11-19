import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.training.generate_data import run_self_play_game
from app.ai.heuristic_ai import HeuristicAI
from app.ai.minimax_ai import MinimaxAI
from app.models import AIConfig

def test_self_play():
    print("Starting self-play test...")
    ai1 = HeuristicAI(1, AIConfig(difficulty=5))
    ai2 = MinimaxAI(2, AIConfig(difficulty=3))
    
    history, winner = run_self_play_game(ai1, ai2)
    
    print(f"Game finished. Winner: {winner}")
    print(f"History length: {len(history)}")
    
    # Check if it failed due to no moves
    if len(history) > 0:
        last_state_features, last_player = history[-1]
        print(f"Last player to move: {last_player}")

if __name__ == "__main__":
    test_self_play()