from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class ModelParams:
    # Memory / encoding
    omega: float = 1.0
    decay_d: float = 0.55
    retrieve_k: int = 10
    topic_spread: float = 0.35
    source_bias: Dict[str, float] = field(default_factory=lambda: {"gov": 0.15, "media": 0.05, "reddit": 0.0})

    # Emotion dynamics
    delta_E: float = 0.20     # inertia/decay
    beta_E: float = 0.25      # contagion strength
    lam_E: float = 0.35       # stimulus forcing
    gamma_conflict: float = 0.10  # optional: conflict -> arousal
    clip_E: Tuple[float, float] = (0.0, 1.0)

    # Belief update
    eta_delib: float = 0.10
    eta_heur: float = 0.10
    mu_gov: float = 0.05
    mu_media: float = 0.02
    mu_reddit: float = 0.00
    mu_conflict: float = 0.05
    clip_b: Tuple[float, float] = (-1.0, 1.0)

    # Affect gate
    gate_a0: float = -0.25
    gate_a1: float = 2.0
    gate_a2: float = 0.75

    # Messaging
    inbox_capacity: int = 8
    share_prob_base: float = 0.10
    share_prob_emotion: float = 0.50
    share_prob_alignment: float = 0.20  # if belief aligns with message valence
