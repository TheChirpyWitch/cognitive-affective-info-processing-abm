from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .memory import MemoryChunk, safe_log
from .message import EIMMessage, Emotion, Topic


@dataclass
class Agent:
    agent_id: int
    belief: float  # b in [-1, 1]
    emotion: float  # E in [0, 1] (arousal/threat proxy)
    kappa: float  # impulse control in [0, 1]
    memory: Dict[Tuple[Emotion, Topic, str], MemoryChunk] = field(default_factory=dict)

    def encode_message(self, msg: EIMMessage, t: int, omega: float = 1.0) -> None:
        """
        ACT-R-ish encoding:
          enc_strength = 1 + omega * |cell_value|
        This treats emotional magnitude as stronger trace strength.
        """
        for (emo, topic), val in msg.cells.items():
            enc_strength = 1.0 + omega * abs(val)
            key = (emo, topic, msg.source)
            chunk = self.memory.get(key)
            if chunk is None:
                chunk = MemoryChunk(emotion=emo, topic=topic, source=msg.source, last_seen=t)
                self.memory[key] = chunk
            chunk.count += enc_strength
            chunk.value_sum += enc_strength * val
            chunk.last_seen = t

    def _activation(self, chunk: MemoryChunk, t: int, decay_d: float, source_bias: Dict[str, float]) -> float:
        """
        Simplified activation:
          A = log(count) - d*log(1+age) + source_bias + noise
        """
        age = max(0, t - chunk.last_seen)
        B = safe_log(chunk.count) - decay_d * safe_log(1.0 + age)
        bias = source_bias.get(chunk.source, 0.0)
        noise = random.gauss(0.0, 0.15)
        return B + bias + noise

    def retrieve_topk(
        self,
        t: int,
        k: int = 10,
        decay_d: float = 0.5,
        context_topics: Optional[Iterable[Topic]] = None,
        topic_spread: float = 0.25,
        source_bias: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[MemoryChunk, float]]:
        """
        Retrieve top-K chunks by activation, optionally with context spreading
        toward current topics (very lightweight, not full ACT-R).
        """
        if source_bias is None:
            source_bias = {}

        ctx = set(context_topics) if context_topics is not None else set()

        scored: List[Tuple[MemoryChunk, float]] = []
        for chunk in self.memory.values():
            A = self._activation(chunk, t, decay_d, source_bias)

            # Minimal "spreading activation": boost if chunk topic is in context
            if ctx and chunk.topic in ctx:
                A += topic_spread

            scored.append((chunk, A))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def cognitive_evidence(self, retrieved: List[Tuple[MemoryChunk, float]]) -> Tuple[float, Dict[str, float], float, float]:
        """
        Returns:
          E_cog  : retrieval-weighted mean chunk value
          pi_src : softmax-weighted source composition
          conflict: weighted variance of values
          effort : inverse total activation mass
        """
        if not retrieved:
            return 0.0, {"reddit": 0.0, "gov": 0.0, "media": 0.0}, 0.0, 1.0

        A = np.array([a for (_, a) in retrieved], dtype=float)
        # stable softmax
        w = np.exp(A - A.max())
        w = w / (w.sum() + 1e-12)

        vals = np.array([c.mean_value for (c, _) in retrieved], dtype=float)
        E_cog = float((w * vals).sum())

        # source composition
        pi_src: Dict[str, float] = {}
        for (chunk, _a), wi in zip(retrieved, w):
            pi_src[chunk.source] = pi_src.get(chunk.source, 0.0) + float(wi)

        conflict = float((w * (vals - E_cog) ** 2).sum())
        effort = float(1.0 / (w.sum() + 1e-12))  # with normalized w, effort ~ 1, but keep placeholder
        return E_cog, pi_src, conflict, effort

    def affect_gate(self, a0: float, a1: float, a2: float) -> float:
        """
        g(E) = sigmoid(a0 + a1*E - a2*kappa)
        """
        x = a0 + a1 * self.emotion - a2 * self.kappa
        return 1.0 / (1.0 + math.exp(-x))
