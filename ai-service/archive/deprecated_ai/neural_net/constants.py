"""Re-export constants from canonical location for backwards compatibility.

This stub exists to support legacy imports from archive/deprecated_ai/_neural_net_legacy.py.
The canonical constants are defined in app/ai/neural_net/constants.py.
"""

# Import all constants from canonical location
# Use direct imports to avoid circular dependencies
import sys
from pathlib import Path

# Add the project root to sys.path if needed
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Re-export all constants from canonical module
from app.ai.neural_net.constants import (
    # Invalid move marker
    INVALID_MOVE_INDEX,
    # Canonical sizes
    MAX_N,
    MAX_PLAYERS,
    # Square board policy sizes
    POLICY_SIZE_8x8,
    POLICY_SIZE_19x19,
    POLICY_SIZE,
    # Square8 layout
    SQUARE8_PLACEMENT_SPAN,
    SQUARE8_MOVEMENT_BASE,
    SQUARE8_MOVEMENT_SPAN,
    SQUARE8_LINE_FORM_BASE,
    SQUARE8_LINE_FORM_SPAN,
    SQUARE8_TERRITORY_CLAIM_BASE,
    SQUARE8_TERRITORY_CLAIM_SPAN,
    SQUARE8_SPECIAL_BASE,
    SQUARE8_SKIP_PLACEMENT_IDX,
    SQUARE8_SWAP_SIDES_IDX,
    SQUARE8_SKIP_RECOVERY_IDX,
    SQUARE8_LINE_CHOICE_BASE,
    SQUARE8_LINE_CHOICE_SPAN,
    SQUARE8_TERRITORY_CHOICE_BASE,
    SQUARE8_TERRITORY_CHOICE_SPAN,
    SQUARE8_EXTRA_SPECIAL_BASE,
    SQUARE8_NO_PLACEMENT_ACTION_IDX,
    SQUARE8_NO_MOVEMENT_ACTION_IDX,
    SQUARE8_SKIP_CAPTURE_IDX,
    SQUARE8_NO_LINE_ACTION_IDX,
    SQUARE8_NO_TERRITORY_ACTION_IDX,
    SQUARE8_SKIP_TERRITORY_PROCESSING_IDX,
    SQUARE8_FORCED_ELIMINATION_IDX,
    MAX_DIST_SQUARE8,
    SQUARE8_EXTRA_SPECIAL_SPAN,
    # Square19 layout
    NUM_SQUARE_DIRS,
    NUM_LINE_DIRS,
    MAX_DIST_SQUARE19,
    TERRITORY_SIZE_BUCKETS,
    TERRITORY_MAX_PLAYERS,
    SQUARE19_PLACEMENT_SPAN,
    SQUARE19_MOVEMENT_BASE,
    SQUARE19_MOVEMENT_SPAN,
    SQUARE19_LINE_FORM_BASE,
    SQUARE19_LINE_FORM_SPAN,
    SQUARE19_TERRITORY_CLAIM_BASE,
    SQUARE19_TERRITORY_CLAIM_SPAN,
    SQUARE19_SPECIAL_BASE,
    SQUARE19_SKIP_PLACEMENT_IDX,
    SQUARE19_SWAP_SIDES_IDX,
    SQUARE19_SKIP_RECOVERY_IDX,
    SQUARE19_LINE_CHOICE_BASE,
    SQUARE19_LINE_CHOICE_SPAN,
    SQUARE19_TERRITORY_CHOICE_BASE,
    SQUARE19_TERRITORY_CHOICE_SPAN,
    SQUARE19_EXTRA_SPECIAL_BASE,
    SQUARE19_NO_PLACEMENT_ACTION_IDX,
    SQUARE19_NO_MOVEMENT_ACTION_IDX,
    SQUARE19_SKIP_CAPTURE_IDX,
    SQUARE19_NO_LINE_ACTION_IDX,
    SQUARE19_NO_TERRITORY_ACTION_IDX,
    SQUARE19_SKIP_TERRITORY_PROCESSING_IDX,
    SQUARE19_FORCED_ELIMINATION_IDX,
    SQUARE19_EXTRA_SPECIAL_SPAN,
    # Hex board
    HEX_BOARD_SIZE,
    HEX_MAX_DIST,
    HEX_DIRS,
    NUM_HEX_DIRS,
    HEX_PLACEMENT_SPAN,
    HEX_MOVEMENT_BASE,
    HEX_MOVEMENT_SPAN,
    HEX_SPECIAL_BASE,
    P_HEX,
    # Hex8 board
    HEX8_BOARD_SIZE,
    HEX8_MAX_DIST,
    HEX8_PLACEMENT_SPAN,
    HEX8_MOVEMENT_BASE,
    HEX8_MOVEMENT_SPAN,
    HEX8_SPECIAL_BASE,
    POLICY_SIZE_HEX8,
    # Board type mappings
    BOARD_POLICY_SIZES,
    BOARD_SPATIAL_SIZES,
    get_policy_size_for_board,
    get_spatial_size_for_board,
)
