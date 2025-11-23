from dataclasses import dataclass
from app.models import BoardType


@dataclass
class TrainConfig:
    """Configuration for training run"""
    board_type: BoardType = BoardType.SQUARE8
    episodes_per_iter: int = 4
    epochs_per_iter: int = 4
    # Number of self-play + training + evaluation cycles to run in the
    # high-level training loop. This was previously hard-coded; exposing it
    # here makes iterative retraining a first-class configuration parameter.
    iterations: int = 2
    learning_rate: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 1e-4
    history_length: int = 3
    seed: int = 42
    max_moves_per_game: int = 200
    k_elo: int = 32
    policy_weight: float = 1.0

    # Model identity used to derive checkpoint filenames. This is kept in sync
    # with NeuralNetAI, which expects checkpoints under
    # "<repo_root>/ai-service/models/<nn_model_id>.pth".
    model_id: str = "ringrift_v1"
     
    # Paths
    data_dir: str = "ai-service/app/training/data"
    model_dir: str = "ai-service/models"
    log_dir: str = "ai-service/app/logs"
     
    def __post_init__(self):
        import os
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)