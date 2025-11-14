"""
RingRift AI Service - FastAPI Application
Provides AI move selection and position evaluation endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
import logging

from .ai.random_ai import RandomAI
from .ai.heuristic_ai import HeuristicAI
from .models import GameState, Move, AIConfig, AIType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RingRift AI Service",
    description="AI move selection and evaluation service for RingRift",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI instances cache
ai_instances: Dict[str, Any] = {}


class MoveRequest(BaseModel):
    """Request model for AI move selection"""
    game_state: GameState
    player_number: int
    difficulty: int = Field(ge=1, le=10, default=5)
    ai_type: Optional[AIType] = None


class MoveResponse(BaseModel):
    """Response model for AI move selection"""
    move: Optional[Move]
    evaluation: float
    thinking_time_ms: int
    ai_type: str
    difficulty: int


class EvaluationRequest(BaseModel):
    """Request model for position evaluation"""
    game_state: GameState
    player_number: int


class EvaluationResponse(BaseModel):
    """Response model for position evaluation"""
    score: float
    breakdown: Dict[str, float]


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "RingRift AI Service",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check for container orchestration"""
    return {"status": "healthy"}


@app.post("/ai/move", response_model=MoveResponse)
async def get_ai_move(request: MoveRequest):
    """
    Get AI-selected move for current game state
    
    Args:
        request: MoveRequest containing game state and AI configuration
        
    Returns:
        MoveResponse with selected move and evaluation
    """
    try:
        import time
        start_time = time.time()
        
        # Select AI type based on difficulty if not specified
        ai_type = request.ai_type or _select_ai_type(request.difficulty)
        
        # Get or create AI instance
        ai_key = f"{ai_type.value}-{request.difficulty}-{request.player_number}"
        
        if ai_key not in ai_instances:
            config = AIConfig(
                difficulty=request.difficulty,
                randomness=_get_randomness_for_difficulty(request.difficulty)
            )
            ai_instances[ai_key] = _create_ai_instance(
                ai_type, 
                request.player_number, 
                config
            )
        
        ai = ai_instances[ai_key]
        
        # Get move from AI
        move = ai.select_move(request.game_state)
        
        # Evaluate position
        evaluation = ai.evaluate_position(request.game_state)
        
        thinking_time = int((time.time() - start_time) * 1000)
        
        logger.info(
            f"AI move: type={ai_type.value}, difficulty={request.difficulty}, "
            f"time={thinking_time}ms, eval={evaluation:.2f}"
        )
        
        return MoveResponse(
            move=move,
            evaluation=evaluation,
            thinking_time_ms=thinking_time,
            ai_type=ai_type.value,
            difficulty=request.difficulty
        )
        
    except Exception as e:
        logger.error(f"Error generating AI move: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/evaluate", response_model=EvaluationResponse)
async def evaluate_position(request: EvaluationRequest):
    """
    Evaluate current position from a player's perspective
    
    Args:
        request: EvaluationRequest with game state and player number
        
    Returns:
        EvaluationResponse with position score and breakdown
    """
    try:
        # Use heuristic AI for evaluation
        config = AIConfig(difficulty=5, randomness=0)
        ai = HeuristicAI(request.player_number, config)
        
        score = ai.evaluate_position(request.game_state)
        breakdown = ai.get_evaluation_breakdown(request.game_state)
        
        return EvaluationResponse(
            score=score,
            breakdown=breakdown
        )
        
    except Exception as e:
        logger.error(f"Error evaluating position: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/ai/cache")
async def clear_ai_cache():
    """Clear cached AI instances"""
    global ai_instances
    ai_instances.clear()
    logger.info("AI cache cleared")
    return {"status": "cache cleared", "instances_removed": len(ai_instances)}


def _select_ai_type(difficulty: int) -> AIType:
    """Auto-select AI type based on difficulty"""
    if difficulty <= 2:
        return AIType.RANDOM
    elif difficulty <= 5:
        return AIType.HEURISTIC
    elif difficulty <= 8:
        return AIType.MINIMAX
    else:
        return AIType.MCTS


def _get_randomness_for_difficulty(difficulty: int) -> float:
    """Get randomness factor for difficulty level"""
    randomness_map = {
        1: 0.5, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.05,
        6: 0.02, 7: 0.01, 8: 0, 9: 0, 10: 0
    }
    return randomness_map.get(difficulty, 0.1)


def _create_ai_instance(ai_type: AIType, player_number: int, config: AIConfig):
    """Factory function to create AI instances"""
    if ai_type == AIType.RANDOM:
        return RandomAI(player_number, config)
    elif ai_type == AIType.HEURISTIC:
        return HeuristicAI(player_number, config)
    # elif ai_type == AIType.MINIMAX:
    #     return MinimaxAI(player_number, config)
    # elif ai_type == AIType.MCTS:
    #     return MCTSAI(player_number, config)
    else:
        # Default to heuristic
        return HeuristicAI(player_number, config)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
