"""
Pydantic Models for RingRift Game State
Mirrors TypeScript types from src/shared/types/game.ts
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class BoardType(str, Enum):
    """Board type enumeration"""
    SQUARE8 = "square8"
    SQUARE19 = "square19"
    HEXAGONAL = "hexagonal"


class GamePhase(str, Enum):
    """Game phase enumeration"""
    RING_PLACEMENT = "ring_placement"
    MOVEMENT = "movement"
    CAPTURE = "capture"
    LINE_PROCESSING = "line_processing"
    TERRITORY_PROCESSING = "territory_processing"


class GameStatus(str, Enum):
    """Game status enumeration"""
    WAITING = "waiting"
    ACTIVE = "active"
    FINISHED = "finished"
    PAUSED = "paused"
    ABANDONED = "abandoned"
    COMPLETED = "completed"


class AIType(str, Enum):
    """AI type enumeration"""
    RANDOM = "random"
    HEURISTIC = "heuristic"
    MINIMAX = "minimax"
    MCTS = "mcts"


class Position(BaseModel):
    """Board position (2D or 3D for hexagonal)"""
    x: int
    y: int
    z: Optional[int] = None
    
    def to_key(self) -> str:
        """Convert position to string key"""
        if self.z is not None:
            return f"{self.x},{self.y},{self.z}"
        return f"{self.x},{self.y}"


class RingStack(BaseModel):
    """Ring stack on the board"""
    position: Position
    rings: List[int]  # Player numbers from bottom to top
    stack_height: int = Field(alias="stackHeight")
    cap_height: int = Field(alias="capHeight")
    controlling_player: int = Field(alias="controllingPlayer")
    
    class Config:
        populate_by_name = True


class MarkerInfo(BaseModel):
    """Marker information"""
    player: int
    position: Position
    type: str  # 'regular' or 'collapsed'


class Player(BaseModel):
    """Player state"""
    id: str
    username: str
    type: str
    player_number: int = Field(alias="playerNumber")
    is_ready: bool = Field(alias="isReady")
    time_remaining: int = Field(alias="timeRemaining")
    ai_difficulty: Optional[int] = Field(None, alias="aiDifficulty")
    rings_in_hand: int = Field(alias="ringsInHand")
    eliminated_rings: int = Field(alias="eliminatedRings")
    territory_spaces: int = Field(alias="territorySpaces")
    
    class Config:
        populate_by_name = True


class TimeControl(BaseModel):
    """Time control settings"""
    initial_time: int = Field(alias="initialTime")
    increment: int
    type: str
    
    class Config:
        populate_by_name = True


class Move(BaseModel):
    """Move representation"""
    id: str
    type: str
    player: int
    from_pos: Optional[Position] = Field(None, alias="from")
    to: Position
    timestamp: datetime
    think_time: int = Field(alias="thinkTime")
    move_number: int = Field(alias="moveNumber")
    
    class Config:
        populate_by_name = True


class BoardState(BaseModel):
    """Current board state"""
    type: BoardType
    size: int
    stacks: Dict[str, RingStack] = {}
    markers: Dict[str, MarkerInfo] = {}
    collapsed_spaces: Dict[str, int] = Field(default_factory=dict, alias="collapsedSpaces")
    eliminated_rings: Dict[str, int] = Field(default_factory=dict, alias="eliminatedRings")
    
    class Config:
        populate_by_name = True


class GameState(BaseModel):
    """Complete game state"""
    id: str
    board_type: BoardType = Field(alias="boardType")
    board: BoardState
    players: List[Player]
    current_phase: GamePhase = Field(alias="currentPhase")
    current_player: int = Field(alias="currentPlayer")
    move_history: List[Move] = Field(default_factory=list, alias="moveHistory")
    time_control: TimeControl = Field(alias="timeControl")
    spectators: List[str] = Field(default_factory=list)
    game_status: GameStatus = Field(alias="gameStatus")
    winner: Optional[int] = None
    created_at: datetime = Field(alias="createdAt")
    last_move_at: datetime = Field(alias="lastMoveAt")
    is_rated: bool = Field(alias="isRated")
    max_players: int = Field(alias="maxPlayers")
    total_rings_in_play: int = Field(alias="totalRingsInPlay")
    total_rings_eliminated: int = Field(alias="totalRingsEliminated")
    victory_threshold: int = Field(alias="victoryThreshold")
    territory_victory_threshold: int = Field(alias="territoryVictoryThreshold")
    
    class Config:
        populate_by_name = True


class AIConfig(BaseModel):
    """AI configuration"""
    difficulty: int = Field(ge=1, le=10)
    think_time: Optional[int] = Field(None, alias="thinkTime")
    randomness: Optional[float] = Field(None, ge=0, le=1)
    
    class Config:
        populate_by_name = True
