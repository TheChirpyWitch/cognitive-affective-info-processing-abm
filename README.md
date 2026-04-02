# Hybrid Cognitive-Affective ABM

An agent-based model (ABM) that simulates how people form and update beliefs when exposed to emotionally charged information from different sources (Reddit, media, government). It combines a memory retrieval model (ACT-R-inspired) with an emotion contagion model (AgentZero-style) and a dual-process belief update mechanism.

---

## Project Structure

```
claude/
├── main.py                  # Entry point: runs the demo simulation
└── hybrid_model/
    ├── __init__.py
    ├── message.py           # EIMMessage — the information unit passed between agents
    ├── memory.py            # MemoryChunk — how a single memory trace is stored
    ├── agent.py             # Agent — encodes messages, retrieves memories, updates state
    ├── params.py            # ModelParams — all tunable hyperparameters
    ├── model.py             # CognitiveAffectiveABM — the simulation engine
    └── demo.py              # Helpers to build a toy scenario and generate fake messages
```

---

## Core Concepts

### 1. Messages: Emotional Imprinting Matrix (EIM)

**File:** `message.py`

Information in this model is not plain text — it is represented as an **Emotional Imprinting Matrix (EIM)**: a sparse dictionary mapping `(emotion, topic)` pairs to a float value (polarity/intensity).

```python
EIMMessage(
    time=3,
    source="reddit",
    cells={("anger", "topic_b"): -0.25}
)
```

This captures the idea that a piece of content activates specific emotional frames around specific topics. The `.intensity()` method returns the mean absolute value across cells, used as a stimulus signal.

---

### 2. Memory: ACT-R-Inspired Traces

**File:** `memory.py`

Each agent maintains a memory of what they have been exposed to. A `MemoryChunk` represents a single trace keyed by `(emotion, topic, source)`. It tracks:

- `count` — how many times this trace has been reinforced (weighted by encoding strength)
- `value_sum` — accumulated valence (positive/negative signal)
- `last_seen` — timestep of most recent exposure (used for decay)

The `mean_value` property gives the average polarity of a chunk, representing the agent's "learned sentiment" about that emotion-topic-source combination.

---

### 3. Agent: Cognition and Affect

**File:** `agent.py`

Each agent has:
- `belief` — a scalar in `[-1, 1]` representing their position on an issue
- `emotion` — a scalar in `[0, 1]` representing arousal/threat level
- `kappa` — impulse control `[0, 1]`; higher means more deliberative

**Encoding** (`encode_message`): When an agent sees a message, its cells are written into memory. Encoding strength is `1 + omega * |value|`, meaning emotionally intense content leaves a stronger trace.

**Retrieval** (`retrieve_topk`): Activates the top-K memory chunks using a simplified ACT-R activation formula:

```
A = log(count) - d * log(1 + age) + source_bias + noise
```

- Chunks decay with age (recency matters)
- Chunks from trusted sources get a bias boost
- Chunks whose topic matches the current message context get a spread bonus

**Cognitive Evidence** (`cognitive_evidence`): Aggregates retrieved chunks into:
- `E_cog` — softmax-weighted mean value (what the agent "thinks")
- `pi_src` — source composition (how much gov/media/reddit is influencing them)
- `conflict` — weighted variance across retrieved values (internal disagreement)

**Affect Gate** (`affect_gate`): A sigmoid function that determines how much the agent relies on heuristics (emotional/social) vs. deliberation (cognitive):

```
g(E) = sigmoid(a0 + a1 * E - a2 * kappa)
```

When emotion is high or kappa is low, `g` approaches 1 and the agent mostly imitates neighbors (heuristic). When calm and high-kappa, the agent deliberates.

---

### 4. Parameters

**File:** `params.py`

All model hyperparameters are collected in `ModelParams`. Key groups:

| Group | Parameters | What they control |
|---|---|---|
| Memory | `omega`, `decay_d`, `retrieve_k`, `topic_spread`, `source_bias` | Encoding strength, memory decay rate, retrieval breadth, source trust |
| Emotion | `delta_E`, `beta_E`, `lam_E`, `gamma_conflict` | Emotion inertia, contagion from neighbors, stimulus forcing, conflict-driven arousal |
| Belief | `eta_delib`, `eta_heur`, `mu_*` | Step sizes for deliberative and heuristic updates, source weighting |
| Affect gate | `gate_a0`, `gate_a1`, `gate_a2` | Shape of the deliberation/heuristic tradeoff curve |
| Messaging | `inbox_capacity`, `share_prob_*` | How many messages an agent can hold, sharing likelihood |

---

### 5. Simulation Engine

**File:** `model.py`

`CognitiveAffectiveABM` runs agents on a social network (`networkx.Graph`). Each call to `.step()` performs one synchronous timestep:

1. **Exogenous exposure** — each agent randomly samples a few messages from the external pool (e.g., news, Reddit)
2. **Encoding** — sampled messages are written into each agent's memory
3. **Retrieval** — agent pulls top-K memory chunks given the current message context
4. **Emotion update** — new emotion is a weighted mix of:
   - Prior emotion (inertia)
   - Neighbor mean emotion (contagion)
   - Message stimulus (external forcing)
   - Memory conflict (internal stress)
5. **Belief update** — blended by the affect gate:
   - Deliberative path: driven by `E_cog` and source composition from memory
   - Heuristic path: move toward neighbor mean belief
6. **Sharing** — agents stochastically forward the most intense message they saw to their neighbors; sharing probability increases with emotion and belief-message alignment

After each step, a snapshot `{t, mean_belief, var_belief, mean_emotion, var_emotion, outgoing_messages}` is appended to `model.history`.

---

### 6. Demo

**File:** `demo.py`

`make_demo_messages(t)` generates 20 synthetic messages per timestep across three sources and three topics, with stylized intensity by source (Reddit > media > gov). Topic B with anger is given a negative valence to create a detectable signal.

`build_demo()` constructs a 50-agent Watts-Strogatz small-world network with randomized initial beliefs and emotions.

---

## Running the Simulation

```bash
python main.py
```

This runs 60 timesteps of the demo and plots five time series:

- **Mean belief** — does the population drift positive or negative?
- **Belief variance** — a proxy for polarization
- **Mean emotion** — overall arousal level
- **Emotion variance** — spread of emotional states
- **Message volume** — how many peer-to-peer forwards happened per step

---

## Dependencies

```
numpy
networkx
matplotlib
```
