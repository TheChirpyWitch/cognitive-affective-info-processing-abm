from __future__ import annotations
import random
from typing import Dict, List

import networkx as nx
import numpy as np

from .agent import Agent
from .message import EIMMessage
from .model import CognitiveAffectiveABM
from .params import ModelParams


def make_demo_messages(t: int) -> List[EIMMessage]:
    # Three sources, one "topic_b" gets negative-anger framing sometimes
    topics = ["topic_a", "topic_b", "topic_c"]
    emotions = ["anger", "fear", "hope", "neutral"]

    msgs: List[EIMMessage] = []
    for _ in range(20):
        src = random.choice(["reddit", "gov", "media"])
        topic = random.choice(topics)
        emo = random.choice(emotions)

        # crude stylized values (would plug real sentiment/emotion outputs here)
        if src == "reddit":
            val = random.uniform(-1.0, 1.0) * 0.8
        elif src == "media":
            val = random.uniform(-1.0, 1.0) * 0.5
        else:  # gov
            val = random.uniform(-1.0, 1.0) * 0.3

        # make "topic_b" occasionally anger-negative to mimic your example
        if topic == "topic_b" and emo == "anger":
            val = random.choice([-0.25, -0.4, 0.2])

        msgs.append(EIMMessage(time=t, source=src, cells={(emo, topic): float(val)}))
    return msgs


def build_demo() -> CognitiveAffectiveABM:
    random.seed(7)
    np.random.seed(7)

    n = 50
    G = nx.watts_strogatz_graph(n=n, k=4, p=0.15, seed=7)

    agents: Dict[int, Agent] = {}
    for i in range(n):
        agents[i] = Agent(
            agent_id=i,
            belief=float(np.random.uniform(-0.2, 0.2)),
            emotion=float(np.random.uniform(0.0, 0.2)),
            kappa=float(np.random.uniform(0.2, 0.9)),
        )

    params = ModelParams()
    model = CognitiveAffectiveABM(G=G, agents=agents, params=params)
    return model
