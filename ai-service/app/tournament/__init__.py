"""Tournament system for AI agent evaluation with Elo ratings."""

from .agents import AIAgent, AIAgentRegistry, AgentType
from .elo import EloRating, EloCalculator
from .scheduler import Match, MatchStatus, TournamentScheduler, RoundRobinScheduler, SwissScheduler
from .runner import MatchResult, TournamentRunner, TournamentResults
from .unified_elo_db import (
    EloDatabase,
    UnifiedEloRating,
    MatchRecord,
    get_elo_database,
    reset_elo_database,
)

__all__ = [
    "AIAgent",
    "AIAgentRegistry",
    "AgentType",
    "EloRating",
    "EloCalculator",
    "Match",
    "MatchStatus",
    "TournamentScheduler",
    "RoundRobinScheduler",
    "SwissScheduler",
    "MatchResult",
    "TournamentRunner",
    "TournamentResults",
    # Unified Elo database
    "EloDatabase",
    "UnifiedEloRating",
    "MatchRecord",
    "get_elo_database",
    "reset_elo_database",
]
