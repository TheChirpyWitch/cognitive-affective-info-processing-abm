from __future__ import annotations
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .agent import Agent
from .message import EIMMessage
from .params import ModelParams


class CognitiveAffectiveABM:
    def __init__(self, G: nx.Graph, agents: Dict[int, Agent], params: ModelParams):
        self.G = G
        self.agents = agents
        self.p = params
        self.t = 0
        self.inbox: Dict[int, List[EIMMessage]] = {i: [] for i in agents.keys()}

        # history storage
        self.history: List[Dict[str, float]] = []
        self.last_outgoing_count: int = 0

    def _neighbors_mean(self, values: Dict[int, float], i: int) -> float:
        nbrs = list(self.G.neighbors(i))
        if not nbrs:
            return values[i]
        return float(np.mean([values[j] for j in nbrs]))

    def deliver(self, receiver: int, msg: EIMMessage):
        inbox = self.inbox[receiver]
        if len(inbox) < self.p.inbox_capacity:
            inbox.append(msg)

    def step(self, exogenous_messages: List[EIMMessage]):
        """
        One timestep:
          1) exogenous broadcast (optional) -> deliver to some agents (here: everyone sees a sample)
          2) agents encode what they saw
          3) agents retrieve -> cognition
          4) update emotion
          5) update belief
          6) decide to share -> message passing to neighbors (EIM payload)
        """
        t = self.t

        # 1) Exogenous exposure: for simplicity, each agent samples a few messages from the pool
        for i, agent in self.agents.items():
            sample = random.sample(exogenous_messages, k=min(3, len(exogenous_messages))) if exogenous_messages else []
            for msg in sample:
                self.deliver(i, msg)

        # Snapshot current beliefs/emotions for synchronous updates
        prev_b = {i: a.belief for i, a in self.agents.items()}
        prev_E = {i: a.emotion for i, a in self.agents.items()}

        # 2–5) Per agent: encode, retrieve, update emotion, update belief
        new_b: Dict[int, float] = {}
        new_E: Dict[int, float] = {}

        for i, agent in self.agents.items():
            seen = self.inbox[i]
            # encode messages into memory
            for msg in seen:
                agent.encode_message(msg, t=t, omega=self.p.omega)

            # cognition: context topics from seen messages
            ctx_topics = []
            for msg in seen:
                ctx_topics.extend(msg.topics())
            retrieved = agent.retrieve_topk(
                t=t,
                k=self.p.retrieve_k,
                decay_d=self.p.decay_d,
                context_topics=ctx_topics,
                topic_spread=self.p.topic_spread,
                source_bias=self.p.source_bias
            )
            E_cog, pi_src, conflict, _effort = agent.cognitive_evidence(retrieved)

            # stimulus from seen messages (exogenous forcing)
            stim = float(np.mean([m.intensity() for m in seen])) if seen else 0.0

            # emotion update (AgentZero++ style)
            mean_nbr_E = self._neighbors_mean(prev_E, i)
            Ei = (1 - self.p.delta_E) * prev_E[i] + self.p.beta_E * mean_nbr_E + self.p.lam_E * stim + self.p.gamma_conflict * conflict
            Ei = float(np.clip(Ei, *self.p.clip_E))
            new_E[i] = Ei

            # affect gate
            agent.emotion = Ei  # update before computing gate
            g = agent.affect_gate(self.p.gate_a0, self.p.gate_a1, self.p.gate_a2)

            # deliberative update from cognition (source-aware)
            delib = self.p.eta_delib * (
                E_cog
                + self.p.mu_gov * pi_src.get("gov", 0.0)
                + self.p.mu_media * pi_src.get("media", 0.0)
                + self.p.mu_reddit * pi_src.get("reddit", 0.0)
                - self.p.mu_conflict * conflict
            )

            # heuristic update: imitate neighbor mean belief
            mean_nbr_b = self._neighbors_mean(prev_b, i)
            heur = self.p.eta_heur * (mean_nbr_b - prev_b[i])

            bi = prev_b[i] + (1 - g) * delib + g * heur
            bi = float(np.clip(bi, *self.p.clip_b))
            new_b[i] = bi

        # Commit synchronous updates
        for i, agent in self.agents.items():
            agent.belief = new_b[i]
            agent.emotion = new_E[i]

        # 6) Sharing / message passing: agents forward a simple EIM derived from what they saw
        # Clear outgoing first, then deliver
        outgoing: List[Tuple[int, int, EIMMessage]] = []

        for i, agent in self.agents.items():
            seen = self.inbox[i]
            if not seen:
                continue

            # Choose one message to potentially forward (could be "most intense")
            msg = max(seen, key=lambda m: m.intensity())

            # Simple share probability: base + emotion + belief alignment with message sign
            msg_val = float(np.mean(list(msg.cells.values()))) if msg.cells else 0.0
            align = 1.0 if (agent.belief * msg_val) > 0 else 0.0
            p_share = self.p.share_prob_base + self.p.share_prob_emotion * agent.emotion + self.p.share_prob_alignment * align
            p_share = max(0.0, min(1.0, p_share))

            if random.random() < p_share:
                for j in self.G.neighbors(i):
                    # Optionally mutate/transform message to reflect "retelling" (kept simple here)
                    forwarded = EIMMessage(time=t, source="peer", cells=msg.cells)
                    outgoing.append((i, j, forwarded))

        self.last_outgoing_count = len(outgoing)

        # reset inbox and deliver outgoing for next step
        self.inbox = {i: [] for i in self.agents.keys()}
        for _sender, receiver, msg in outgoing:
            self.deliver(receiver, msg)

        # log metrics at end of step (t -> t+1)
        self.t += 1
        snap = self.snapshot()
        snap["outgoing_messages"] = float(self.last_outgoing_count)
        self.history.append(snap)

    def plot_history(self):
        """Plot mean/variance of belief and emotion over time + outgoing message volume."""
        if not self.history:
            raise ValueError("No history yet. Run the model for a few steps first.")

        t = [h["t"] for h in self.history]
        mean_b = [h["mean_belief"] for h in self.history]
        var_b = [h["var_belief"] for h in self.history]
        mean_E = [h["mean_emotion"] for h in self.history]
        var_E = [h["var_emotion"] for h in self.history]
        out = [h.get("outgoing_messages", 0.0) for h in self.history]

        plt.figure()
        plt.plot(t, mean_b)
        plt.xlabel("t")
        plt.ylabel("Mean belief")
        plt.title("Mean belief over time")
        plt.show()

        plt.figure()
        plt.plot(t, var_b)
        plt.xlabel("t")
        plt.ylabel("Belief variance")
        plt.title("Belief variance (polarization proxy) over time")
        plt.show()

        plt.figure()
        plt.plot(t, mean_E)
        plt.xlabel("t")
        plt.ylabel("Mean emotion")
        plt.title("Mean emotion over time")
        plt.show()

        plt.figure()
        plt.plot(t, var_E)
        plt.xlabel("t")
        plt.ylabel("Emotion variance")
        plt.title("Emotion variance over time")
        plt.show()

        plt.figure()
        plt.plot(t, out)
        plt.xlabel("t")
        plt.ylabel("Outgoing messages")
        plt.title("Message volume over time")
        plt.show()

    def snapshot(self) -> Dict[str, float]:
        beliefs = np.array([a.belief for a in self.agents.values()])
        emotions = np.array([a.emotion for a in self.agents.values()])
        return {
            "t": self.t,
            "mean_belief": float(beliefs.mean()),
            "var_belief": float(beliefs.var()),
            "mean_emotion": float(emotions.mean()),
            "var_emotion": float(emotions.var()),
        }
