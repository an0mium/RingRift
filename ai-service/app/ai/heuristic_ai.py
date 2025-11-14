"""
Heuristic AI implementation for RingRift
Uses strategic heuristics to evaluate and select moves
"""

from typing import Optional, List, Dict, Any
import random
from datetime import datetime
import uuid

from .base import BaseAI
from ..models import GameState, Move, AIConfig, Position, RingStack


class HeuristicAI(BaseAI):
    """AI that uses heuristics to select strategic moves"""
    
    # Evaluation weights for different factors
    WEIGHT_STACK_CONTROL = 10.0
    WEIGHT_STACK_HEIGHT = 5.0
    WEIGHT_TERRITORY = 8.0
    WEIGHT_RINGS_IN_HAND = 3.0
    WEIGHT_CENTER_CONTROL = 4.0
    WEIGHT_ADJACENCY = 2.0
    WEIGHT_OPPONENT_THREAT = 6.0
    
    def select_move(self, game_state: GameState) -> Optional[Move]:
        """
        Select the best move using heuristic evaluation
        
        Args:
            game_state: Current game state
            
        Returns:
            Best heuristic move or None if no valid moves
        """
        # Simulate thinking for natural behavior
        self.simulate_thinking(min_ms=500, max_ms=1500)
        
        # Get all valid moves
        valid_moves = self._get_valid_moves_for_phase(game_state)
        
        if not valid_moves:
            return None
        
        # Check if should pick random move based on randomness setting
        if self.should_pick_random_move():
            selected = random.choice(valid_moves)
        else:
            # Evaluate each move and pick the best one
            best_move = None
            best_score = float('-inf')
            
            for move_dict in valid_moves:
                score = self._evaluate_move(move_dict, game_state)
                if score > best_score:
                    best_score = score
                    best_move = move_dict
            
            selected = best_move if best_move else random.choice(valid_moves)
        
        # Convert to Move object
        move = self._create_move_object(selected, game_state)
        
        self.move_count += 1
        return move
    
    def evaluate_position(self, game_state: GameState) -> float:
        """
        Evaluate the current position using heuristics
        
        Args:
            game_state: Current game state
            
        Returns:
            Evaluation score (positive = good for this AI)
        """
        score = 0.0
        
        # Stack control evaluation
        score += self._evaluate_stack_control(game_state)
        
        # Territory evaluation
        score += self._evaluate_territory(game_state)
        
        # Rings in hand evaluation
        score += self._evaluate_rings_in_hand(game_state)
        
        # Center control evaluation
        score += self._evaluate_center_control(game_state)
        
        # Opponent threat evaluation
        score += self._evaluate_opponent_threats(game_state)
        
        return score
    
    def get_evaluation_breakdown(self, game_state: GameState) -> Dict[str, float]:
        """
        Get detailed breakdown of position evaluation
        
        Args:
            game_state: Current game state
            
        Returns:
            Dictionary with evaluation components
        """
        return {
            "total": self.evaluate_position(game_state),
            "stack_control": self._evaluate_stack_control(game_state),
            "territory": self._evaluate_territory(game_state),
            "rings_in_hand": self._evaluate_rings_in_hand(game_state),
            "center_control": self._evaluate_center_control(game_state),
            "opponent_threats": self._evaluate_opponent_threats(game_state)
        }
    
    def _evaluate_stack_control(self, game_state: GameState) -> float:
        """Evaluate stack control"""
        score = 0.0
        my_stacks = 0
        opponent_stacks = 0
        my_height = 0
        opponent_height = 0
        
        for stack in game_state.board.stacks.values():
            if stack.controlling_player == self.player_number:
                my_stacks += 1
                my_height += stack.stack_height
            else:
                opponent_stacks += 1
                opponent_height += stack.stack_height
        
        score += (my_stacks - opponent_stacks) * self.WEIGHT_STACK_CONTROL
        score += (my_height - opponent_height) * self.WEIGHT_STACK_HEIGHT
        
        return score
    
    def _evaluate_territory(self, game_state: GameState) -> float:
        """Evaluate territory control"""
        my_player = self.get_player_info(game_state)
        if not my_player:
            return 0.0
        
        my_territory = my_player.territory_spaces
        
        # Compare with opponents
        opponent_territory = 0
        for player in game_state.players:
            if player.player_number != self.player_number:
                opponent_territory = max(opponent_territory, player.territory_spaces)
        
        return (my_territory - opponent_territory) * self.WEIGHT_TERRITORY
    
    def _evaluate_rings_in_hand(self, game_state: GameState) -> float:
        """Evaluate rings remaining in hand"""
        my_player = self.get_player_info(game_state)
        if not my_player:
            return 0.0
        
        # Having rings in hand is good (more placement options)
        return my_player.rings_in_hand * self.WEIGHT_RINGS_IN_HAND
    
    def _evaluate_center_control(self, game_state: GameState) -> float:
        """Evaluate control of center positions"""
        score = 0.0
        center_positions = self._get_center_positions(game_state)
        
        for pos_key in center_positions:
            if pos_key in game_state.board.stacks:
                stack = game_state.board.stacks[pos_key]
                if stack.controlling_player == self.player_number:
                    score += self.WEIGHT_CENTER_CONTROL
                else:
                    score -= self.WEIGHT_CENTER_CONTROL * 0.5
        
        return score
    
    def _evaluate_opponent_threats(self, game_state: GameState) -> float:
        """Evaluate opponent threats (stacks near our stacks)"""
        score = 0.0
        my_stacks = [s for s in game_state.board.stacks.values() 
                     if s.controlling_player == self.player_number]
        
        for my_stack in my_stacks:
            adjacent = self._get_adjacent_positions(my_stack.position, game_state)
            for adj_pos in adjacent:
                adj_key = adj_pos.to_key()
                if adj_key in game_state.board.stacks:
                    adj_stack = game_state.board.stacks[adj_key]
                    if adj_stack.controlling_player != self.player_number:
                        # Opponent stack adjacent to ours is a threat
                        threat_level = adj_stack.stack_height - my_stack.stack_height
                        score -= threat_level * self.WEIGHT_OPPONENT_THREAT * 0.5
        
        return score
    
    def _evaluate_move(self, move_dict: Dict[str, Any], game_state: GameState) -> float:
        """
        Evaluate a specific move
        
        Args:
            move_dict: Move dictionary
            game_state: Current game state
            
        Returns:
            Move evaluation score
        """
        score = 0.0
        to_pos = move_dict["to"]
        to_key = to_pos.to_key()
        
        # Prefer center positions
        if to_key in self._get_center_positions(game_state):
            score += 15.0
        
        # Prefer positions with many adjacent friendly stacks
        adjacent = self._get_adjacent_positions(to_pos, game_state)
        friendly_adjacent = 0
        enemy_adjacent = 0
        
        for adj_pos in adjacent:
            adj_key = adj_pos.to_key()
            if adj_key in game_state.board.stacks:
                stack = game_state.board.stacks[adj_key]
                if stack.controlling_player == self.player_number:
                    friendly_adjacent += 1
                else:
                    enemy_adjacent += 1
        
        score += friendly_adjacent * 3.0
        score += enemy_adjacent * 2.0  # Also good to be near enemies for influence
        
        # For movement moves, prefer moving stronger stacks
        if move_dict["type"] == "move" and move_dict.get("from"):
            from_key = move_dict["from"].to_key()
            if from_key in game_state.board.stacks:
                stack = game_state.board.stacks[from_key]
                score += stack.stack_height * 2.0
        
        return score
    
    def _get_center_positions(self, game_state: GameState) -> set:
        """Get center position keys for the board"""
        center = set()
        board_type = game_state.board.type
        size = game_state.board.size
        
        if board_type.value == "square8":
            # Center 2x2 of 8x8 board
            for x in [3, 4]:
                for y in [3, 4]:
                    center.add(f"{x},{y}")
        
        elif board_type.value == "square19":
            # Center 3x3 of 19x19 board
            for x in [8, 9, 10]:
                for y in [8, 9, 10]:
                    center.add(f"{x},{y}")
        
        elif board_type.value == "hexagonal":
            # Center hexagon (distance 0-2 from origin)
            for x in range(-2, 3):
                for y in range(-2, 3):
                    z = -x - y
                    if abs(x) <= 2 and abs(y) <= 2 and abs(z) <= 2:
                        center.add(f"{x},{y},{z}")
        
        return center
    
    def _get_valid_moves_for_phase(self, game_state: GameState) -> List[Dict[str, Any]]:
        """Get valid moves (simplified version from RandomAI)"""
        # Import the movement logic from RandomAI (code reuse)
        from .random_ai import RandomAI
        temp_random_ai = RandomAI(self.player_number, self.config)
        return temp_random_ai._get_valid_moves_for_phase(game_state)
    
    def _get_adjacent_positions(self, position: Position, game_state: GameState) -> List[Position]:
        """Get adjacent positions (simplified version from RandomAI)"""
        from .random_ai import RandomAI
        temp_random_ai = RandomAI(self.player_number, self.config)
        return temp_random_ai._get_adjacent_positions(position, game_state)
    
    def _create_move_object(self, move_dict: Dict[str, Any], game_state: GameState) -> Move:
        """Create a Move object from move dictionary"""
        return Move(
            id=str(uuid.uuid4()),
            type=move_dict["type"],
            player=self.player_number,
            **{"from": move_dict.get("from")},
            to=move_dict["to"],
            timestamp=datetime.now(),
            thinkTime=random.randint(500, 1500),
            moveNumber=len(game_state.move_history) + 1
        )
