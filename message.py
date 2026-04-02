from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

Emotion = str
Topic = str
Cell = Tuple[Emotion, Topic]


@dataclass(frozen=True)
class EIMMessage:
    """
    Sparse EIM: only a few (emotion, topic) cells activated.
    Values can be in [-1, 1] (polarity) or [0, 1] (intensity), depending on your pipeline.
    """
    time: int
    source: str  # "reddit" | "gov" | "media" | etc.
    cells: Dict[Cell, float]  # e.g., {("anger","topic_b"): -0.25}

    def intensity(self) -> float:
        # Used for stimulus; treat magnitude as intensity by default
        if not self.cells:
            return 0.0
        return float(np.mean([abs(v) for v in self.cells.values()]))

    def topics(self) -> List[Topic]:
        return list({t for (_, t) in self.cells.keys()})
