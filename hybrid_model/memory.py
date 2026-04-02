from __future__ import annotations
import math
from dataclasses import dataclass

from .message import Emotion, Topic


def safe_log(x: float) -> float:
    return math.log(max(x, 1e-9))


@dataclass
class MemoryChunk:
    """
    One chunk corresponds to a (emotion, topic, source) "trace" with accumulated evidence.
    We keep frequency + last_seen + running value (valence-like tag).
    """
    emotion: Emotion
    topic: Topic
    source: str
    count: float = 0.0
    last_seen: int = 0
    value_sum: float = 0.0  # accumulate valence/polarity contributions

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.count if self.count > 0 else 0.0
