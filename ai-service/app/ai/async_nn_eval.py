from __future__ import annotations

"""
Async/batched neural network evaluation helpers.

This module provides a thin, thread-safe wrapper around NeuralNetAI.evaluate_batch
so search engines can batch speculative GameState evaluations without risking
concurrent model access. It is intended for per-AI-instance use (not global),
keeping history/perspective handling inside NeuralNetAI as the single source.
"""

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np

from ..models import GameState
from .neural_net import NeuralNetAI


class AsyncNeuralBatcher:
    """Thread-safe batcher for NeuralNetAI.

    - `evaluate(...)` runs a synchronous batched evaluation under a lock.
    - `submit(...)` schedules a batched evaluation on a single worker thread,
      also protected by the same lock.

    The lock ensures any direct calls to NeuralNetAI from the main thread
    will serialize safely with background evaluations.
    """

    def __init__(
        self,
        neural_net: NeuralNetAI,
        max_workers: int = 1,
    ) -> None:
        self.neural_net = neural_net
        self._lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=max_workers
        )

    def evaluate(
        self,
        game_states: List[GameState],
    ) -> Tuple[List[float], np.ndarray]:
        with self._lock:
            return self.neural_net.evaluate_batch(game_states)

    def submit(
        self,
        game_states: List[GameState],
    ) -> Future:
        if self._executor is None:
            raise RuntimeError("AsyncNeuralBatcher is shut down")

        def _run():
            with self._lock:
                return self.neural_net.evaluate_batch(game_states)

        return self._executor.submit(_run)

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

