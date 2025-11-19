import sys
import os
import json
from typing import List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.game_engine import GameEngine
from app.models import GameState, Move

def main():
    if len(sys.argv) < 2:
        print("Usage: python parity_check.py <json_input>")
        sys.exit(1)

    try:
        input_data = json.loads(sys.argv[1])
        game_state_data = input_data.get("gameState")
        player_number = input_data.get("playerNumber")
        
        # Convert JSON to Pydantic model
        # We need to handle datetime fields if any, but GameState uses datetime
        # Pydantic can parse ISO strings
        game_state = GameState(**game_state_data)
        
        valid_moves = GameEngine.get_valid_moves(game_state, player_number)
        
        # Convert moves to dicts for JSON output
        moves_data = [move.dict(by_alias=True) for move in valid_moves]
        
        # Handle datetime serialization
        def json_serial(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")

        print(json.dumps(moves_data, default=json_serial))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()