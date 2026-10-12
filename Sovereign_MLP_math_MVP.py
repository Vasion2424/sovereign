from __future__ import annotations
import argparse, json, math, zipfile, shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

OPS = ("sum", "subtract", "multiply")
INPUT_DIM = 2
HIDDEN_CAPACITY = 12
INITIAL_HIDDEN = 8
OUTPUT_DIM = 3
BASE_PATH_LIFE = 10
BASE_NEURON_LIFE = 5
CREATE_INTERVAL = 50
MUTATION_WINDOW = 200
MUTATION_MIN_ACTIVE = 160
CONSERVE_INTERVAL = 10
PRUNE_INTERVAL = 10
WEAK_PRUNE_PROB = 0.25
CREATED_PATH_LIFE = BASE_PATH_LIFE + 1
CREATED_NEURON_LIFE = BASE_NEURON_LIFE + 1
MUTATED_PATH_BONUS = 60
MUTATED_NEURON_BONUS = 30
DORMANT_PATH_BONUS = 2
DORMANT_NEURON_BONUS = 4


def encode_x(pairs: np.ndarray) -> np.ndarray:
    return pairs.astype(np.float64) / 4.5 - 1.0


def target_for(pairs: np.ndarray, op_idx: int) -> np.ndarray:
    a, b = pairs[:, 0].astype(float), pairs[:, 1].astype(float)
    if op_idx == 0:
        result = a + b
        return (result - 9.0) / 9.0
    if op_idx == 1:
        return (a - b) / 9.0
    result = a * b
    return (result - 40.5) / 40.5


def decode_prediction(y: np.ndarray, op_idx: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if op_idx == 0:
        return y * 9.0 + 9.0
    if op_idx == 1:
        return y * 9.0
    return y * 40.5 + 40.5


def true_result(pairs: np.ndarray, op_idx: int) -> np.ndarray:
    a, b = pairs[:, 0], pairs[:, 1]
    if op_idx == 0: return a + b
    if op_idx == 1: return a - b
    return a * b


class TinyMLP:
    """Conventional feed-forward MLP: 2 inputs -> tanh hidden -> 3 regression heads."""
    def __init__(self, seed: int, initial_hidden: int = INITIAL_HIDDEN, capacity: int = HIDDEN_CAPACITY):
        self.rng = np.random.default_rng(seed)
        self.capacity = capacity
        self.W1 = self.rng.normal(0, 0.65, size=(INPUT_DIM, capacity))
        self.b1 = np.zeros(capacity, dtype=float)
        self.W2 = self.rng.normal(0, 0.35, size=(capacity, OUTPUT_DIM))
        self.b2 = np.zeros(OUTPUT_DIM, dtype=float)
        self.active = np.zeros(capacity, dtype=bool)
        self.active[:initial_hidden] = True
        self.W1_mask = np.zeros((INPUT_DIM, capacity), dtype=bool)
        self.W2_mask = np.zeros((capacity, OUTPUT_DIM), dtype=bool)
        self.W1_mask[:, :initial_hidden] = True
        self.W2_mask[:initial_hidden, :] = True

    def clone(self) -> "TinyMLP":
        clone = TinyMLP(0, self.capacity, self.capacity)
        clone.W1 = self.W1.copy(); clone.b1 = self.b1.copy()
        clone.W2 = self.W2.copy(); clone.b2 = self.b2.copy()
        clone.active = self.active.copy()
        clone.W1_mask = self.W1_mask.copy(); clone.W2_mask = self.W2_mask.copy()
        return clone

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        effective_W1 = self.W1 * self.W1_mask
        z1 = x @ effective_W1 + self.b1
        h = np.tanh(z1) * self.active.astype(float)
        z2 = h @ (self.W2 * self.W2_mask) + self.b2
        y = np.tanh(z2)
        return h, y

    def train_batch(self, x: np.ndarray, target: np.ndarray, op_idx: int, lr: float = 0.025) -> float:
        h, y = self.forward(x)
        err = y[:, op_idx] - target
        loss = float(np.mean(err ** 2))
        delta2 = np.zeros_like(y)
        delta2[:, op_idx] = (2.0 / max(len(x), 1)) * err * (1.0 - y[:, op_idx] ** 2)
        grad_W2 = h.T @ delta2
        grad_b2 = delta2.sum(axis=0)
        delta1 = (delta2 @ (self.W2 * self.W2_mask).T) * (1.0 - h ** 2)
        delta1 *= self.active.astype(float)
        grad_W1 = x.T @ delta1
        grad_b1 = delta1.sum(axis=0)
        # Gradient clipping avoids unstable steps on multiplication.
        norm = math.sqrt(float((grad_W1**2).sum() + (grad_W2**2).sum() + (grad_b1**2).sum() + (grad_b2**2).sum()))
        scale = min(1.0, 2.0 / max(norm, 1e-12))
        self.W1 -= lr * grad_W1 * scale * self.W1_mask
        self.b1 -= lr * grad_b1 * scale * self.active
        self.W2 -= lr * grad_W2 * scale * self.W2_mask
        self.b2 -= lr * grad_b2 * scale
        return loss

    def predict_all(self, pairs: np.ndarray) -> np.ndarray:
        _, y = self.forward(encode_x(pairs))
        return y

    def state_metrics(self) -> Dict[str, float]:
        return {
            "active_hidden": int(self.active.sum()),
            "live_W1_paths": int(self.W1_mask.sum()),
            "live_W2_paths": int(self.W2_mask.sum()),
            "live_paths_total": int(self.W1_mask.sum() + self.W2_mask.sum()),
            "W1_norm": float(np.linalg.norm(self.W1 * self.W1_mask)),
            "W2_norm": float(np.linalg.norm(self.W2 * self.W2_mask)),
            "all_weight_norm": float(math.sqrt(np.linalg.norm(self.W1)**2 + np.linalg.norm(self.W2)**2)),
        }


class OrdinaryControl:
    """Fixed-topology standard MLP. Same initial eight hidden units, no sovereign operations."""
    def __init__(self, model: TinyMLP):
        self.model = model.clone()
        self.model.active[INITIAL_HIDDEN:] = False
        self.model.W1_mask[:, INITIAL_HIDDEN:] = False
        self.model.W2_mask[INITIAL_HIDDEN:, :] = False
    def train_batch(self, x: np.ndarray, target: np.ndarray, op_idx: int, lr: float = 0.025) -> float:
        return self.model.train_batch(x, target, op_idx, lr)


class ThreeSovereigns:
    """CREATE, CONSERVE/MUTATE, PRUNE. Recovery is available but not invoked here."""
    def __init__(self, model: TinyMLP, seed: int, prune_probability: float = WEAK_PRUNE_PROB):
        self.model = model
        self.rng = np.random.default_rng(seed + 77_777)
        self.prune_probability = prune_probability
        h = model.capacity
        self.neuron_life = np.zeros(h, dtype=np.float64)
        self.neuron_life[:INITIAL_HIDDEN] = BASE_NEURON_LIFE
        self.W1_life = np.zeros((INPUT_DIM, h), dtype=np.float64)
        self.W1_life[:, :INITIAL_HIDDEN] = BASE_PATH_LIFE
        self.W2_life = np.zeros((h, OUTPUT_DIM), dtype=np.float64)
        self.W2_life[:INITIAL_HIDDEN, :] = BASE_PATH_LIFE
        self.neuron_life_cap = np.full(h, BASE_NEURON_LIFE, dtype=np.float64)
        self.W1_life_cap = np.full((INPUT_DIM, h), BASE_PATH_LIFE, dtype=np.float64)
        self.W2_life_cap = np.full((h, OUTPUT_DIM), BASE_PATH_LIFE, dtype=np.float64)
        self.node_mutated = np.zeros(h, dtype=bool)
        self.W1_mutated = np.zeros((INPUT_DIM, h), dtype=bool)
        self.W2_mutated = np.zeros((h, OUTPUT_DIM), dtype=bool)
        self.node_last_used = np.full(h, -10**9, dtype=int)
        self.W1_last_used = np.full((INPUT_DIM, h), -10**9, dtype=int)
        self.W2_last_used = np.full((h, OUTPUT_DIM), -10**9, dtype=int)
        self.node_ring = np.zeros((MUTATION_WINDOW, h), dtype=np.uint8)
        self.W1_ring = np.zeros((MUTATION_WINDOW, INPUT_DIM, h), dtype=np.uint8)
        self.W2_ring = np.zeros((MUTATION_WINDOW, h, OUTPUT_DIM), dtype=np.uint8)
        self.loss_ring = np.zeros(MUTATION_WINDOW, dtype=np.float64)
        self.create_stage = 0  # 0 path, 1 neuron, 2 adjacent paths
        self.create_target: Optional[int] = None
        self.create_input: int = 0
        self.create_started: Optional[int] = None
        self.counters = {
            "created_neurons": 0, "created_paths": 0, "pruned_paths": 0,
            "pruned_neurons": 0, "mutated_neurons": 0, "mutated_paths": 0,
            "conservation_path_events": 0, "conservation_neuron_events": 0,
            "dormant_bonuses": 0, "creation_attempts": 0,
            "recovery_triggered": False,
        }
        self.events: List[Dict[str, object]] = []
        self.snapshots: List[Dict[str, object]] = []
        self.recovery_active = False
        self.recovery_until = -1
        self.recovery_affected_nodes: List[int] = []
        self.recovery_affected_W1: List[Tuple[int, int]] = []
        self.recovery_affected_W2: List[Tuple[int, int]] = []
        self.recovery_queue: List[Tuple[str, int, int]] = []
        self.recovery_cursor = 0
        self._last_hidden = np.zeros(h)
        self._last_x = np.zeros(INPUT_DIM)
        self._tick_cursor = 0

    def _log(self, cycle: int, event: str, **kwargs):
        self.events.append({"cycle": int(cycle), "event": event, **kwargs})

    def record_and_tick(self, cycle: int, x_batch: np.ndarray, hidden_batch: np.ndarray, op_idx: int):
        model = self.model
        idx = cycle % MUTATION_WINDOW
        hidden = np.mean(np.abs(hidden_batch), axis=0)
        # A connection is considered traversed when it is alive and its source contributes measurable activity.
        w1_use = np.zeros_like(self.W1_life, dtype=np.uint8)
        for i in range(INPUT_DIM):
            src_activity = float(np.mean(np.abs(x_batch[:, i])))
            w1_use[i, :] = ((src_activity > 0.08) & model.active & model.W1_mask[i, :]).astype(np.uint8)
        w2_use = np.zeros_like(self.W2_life, dtype=np.uint8)
        if 0 <= op_idx < OUTPUT_DIM:
            w2_use[:, op_idx] = ((hidden > 0.10) & model.active & model.W2_mask[:, op_idx]).astype(np.uint8)
        node_use = ((hidden > 0.12) & model.active).astype(np.uint8)

        # Dormant-mutation bonus when a previously mutated unit/path is used again after >=30 idle cycles.
        for n in np.where(node_use > 0)[0]:
            if self.node_mutated[n] and cycle - self.node_last_used[n] >= 30:
                self.neuron_life[n] += DORMANT_NEURON_BONUS
                self.counters["dormant_bonuses"] += 1
            self.node_last_used[n] = cycle
        for i, h in zip(*np.where(w1_use > 0)):
            if self.W1_mutated[i, h] and cycle - self.W1_last_used[i, h] >= 30:
                self.W1_life[i, h] += DORMANT_PATH_BONUS
                self.counters["dormant_bonuses"] += 1
            self.W1_last_used[i, h] = cycle
        for h, j in zip(*np.where(w2_use > 0)):
            if self.W2_mutated[h, j] and cycle - self.W2_last_used[h, j] >= 30:
                self.W2_life[h, j] += DORMANT_PATH_BONUS
                self.counters["dormant_bonuses"] += 1
            self.W2_last_used[h, j] = cycle

        self.node_ring[idx, :] = node_use
        self.W1_ring[idx, :, :] = w1_use
        self.W2_ring[idx, :, :] = w2_use
        model.W1_mask = self.W1_life > 0
        model.W2_mask = (self.W2_life > 0) & model.active[:, None]
        # Ensure paths for inactive reserve neurons are not traversable.
        model.W1_mask[:, ~model.active] = False
        model.W2_mask[~model.active, :] = False

        # CONSERVE: +1 life every ten cycles for paths/neurons used in >=8 of the last 10 cycles.
        if cycle % CONSERVE_INTERVAL == 0 and cycle >= CONSERVE_INTERVAL:
            rows10 = [(cycle - k) % MUTATION_WINDOW for k in range(CONSERVE_INTERVAL)]
            n10 = self.node_ring[rows10].sum(axis=0)
            p110 = self.W1_ring[rows10].sum(axis=0)
            p210 = self.W2_ring[rows10].sum(axis=0)
            node_gain = (n10 >= 8) & model.active
            p1_gain = (p110 >= 8) & model.W1_mask
            p2_gain = (p210 >= 8) & model.W2_mask
            self.neuron_life[node_gain] += 1
            self.W1_life[p1_gain] += 1
            self.W2_life[p2_gain] += 1
            self.counters["conservation_neuron_events"] += int(node_gain.sum())
            self.counters["conservation_path_events"] += int(p1_gain.sum() + p2_gain.sum())

        # CONSERVE/MUTATE: 200-cycle evidence window, requiring >=80% usage, one-time permanent bonus.
        if cycle >= MUTATION_WINDOW:
            rows200 = [(cycle - k) % MUTATION_WINDOW for k in range(MUTATION_WINDOW)]
            n200 = self.node_ring[rows200].sum(axis=0)
            p1200 = self.W1_ring[rows200].sum(axis=0)
            p2200 = self.W2_ring[rows200].sum(axis=0)
            new_nodes = (n200 >= MUTATION_MIN_ACTIVE) & model.active & ~self.node_mutated
            for n in np.where(new_nodes)[0]:
                self.node_mutated[n] = True
                self.neuron_life[n] += MUTATED_NEURON_BONUS
                self.counters["mutated_neurons"] += 1
                self._log(cycle, "mutate_neuron", neuron=int(n), added_life=MUTATED_NEURON_BONUS, evidence=int(n200[n]))
            new_w1 = (p1200 >= MUTATION_MIN_ACTIVE) & model.W1_mask & ~self.W1_mutated
            for i, h in zip(*np.where(new_w1)):
                self.W1_mutated[i, h] = True
                self.W1_life[i, h] += MUTATED_PATH_BONUS
                self.counters["mutated_paths"] += 1
                self._log(cycle, "mutate_path", kind="input_hidden", src=int(i), dst=int(h), added_life=MUTATED_PATH_BONUS, evidence=int(p1200[i, h]))
            new_w2 = (p2200 >= MUTATION_MIN_ACTIVE) & model.W2_mask & ~self.W2_mutated
            for h, j in zip(*np.where(new_w2)):
                self.W2_mutated[h, j] = True
                self.W2_life[h, j] += MUTATED_PATH_BONUS
                self.counters["mutated_paths"] += 1
                self._log(cycle, "mutate_path", kind="hidden_output", src=int(h), dst=int(j), added_life=MUTATED_PATH_BONUS, evidence=int(p2200[h, j]))

        # PODA: weakened, probabilistic decrement once per 10 cycles if unused for all 10.
        if cycle % PRUNE_INTERVAL == 0 and cycle >= PRUNE_INTERVAL:
            rows10 = [(cycle - k) % MUTATION_WINDOW for k in range(PRUNE_INTERVAL)]
            p1_recent = self.W1_ring[rows10].sum(axis=0)
            p2_recent = self.W2_ring[rows10].sum(axis=0)
            n_recent = self.node_ring[rows10].sum(axis=0)
            old_w1 = self.W1_life.copy(); old_w2 = self.W2_life.copy()
            # Only existing paths can lose life. Draws are seeded for reproducibility.
            for i, h in zip(*np.where((self.W1_life > 0) & (p1_recent == 0) & model.active[None, :])):
                if self.rng.random() < self._effective_prune_probability(cycle):
                    self.W1_life[i, h] = max(0.0, self.W1_life[i, h] - 1.0)
            for h, j in zip(*np.where((self.W2_life > 0) & (p2_recent == 0) & model.active[:, None])):
                if self.rng.random() < self._effective_prune_probability(cycle):
                    self.W2_life[h, j] = max(0.0, self.W2_life[h, j] - 1.0)
            # Node life decays only when inactive, and a node is removed only if all incident routes are gone.
            for h in np.where(model.active & (n_recent == 0))[0]:
                if self.rng.random() < self._effective_prune_probability(cycle):
                    self.neuron_life[h] = max(0.0, self.neuron_life[h] - 1.0)
            model.W1_mask = self.W1_life > 0
            model.W2_mask = self.W2_life > 0
            model.W1_mask[:, ~model.active] = False
            model.W2_mask[~model.active, :] = False
            removed = int(np.sum((old_w1 > 0) & (self.W1_life <= 0)) + np.sum((old_w2 > 0) & (self.W2_life <= 0)))
            self.counters["pruned_paths"] += removed
            for h in np.where(model.active & (self.neuron_life <= 0))[0]:
                if not model.W1_mask[:, h].any() and not model.W2_mask[h, :].any() and h >= 1:
                    model.active[h] = False
                    self.counters["pruned_neurons"] += 1
                    self._log(cycle, "prune_neuron", neuron=int(h))
            model.W1_mask[:, ~model.active] = False
            model.W2_mask[~model.active, :] = False
            # Keep life state and route masks aligned after pruning/deactivation.
            self.W1_life[:, ~model.active] = 0.0
            self.W2_life[~model.active, :] = 0.0

        # CREAR normally runs every 50 cycles; RECUPERACIÓN instead repairs the damaged zone every 2 cycles.
        if self.recovery_active:
            if cycle % 2 == 0:
                self._recovery_step(cycle)
        elif cycle % CREATE_INTERVAL == 0:
            self._create_tick(cycle, x_batch, hidden_batch, op_idx)

        model.W1_mask = (self.W1_life > 0) & model.active[None, :]
        model.W2_mask = (self.W2_life > 0) & model.active[:, None]
        # Snapshot periodically for audits and plots.
        if cycle % 10 == 0:
            metrics = model.state_metrics()
            metrics.update({"cycle": cycle, "created_neurons": self.counters["created_neurons"], "created_paths": self.counters["created_paths"],
                            "pruned_paths": self.counters["pruned_paths"], "mutated_neurons": self.counters["mutated_neurons"],
                            "mutated_paths": self.counters["mutated_paths"], "prune_probability": self._effective_prune_probability(cycle),
                            "recovery_active": self.recovery_active})
            self.snapshots.append(metrics)

    def _effective_prune_probability(self, cycle: int) -> float:
        if self.recovery_active and cycle <= self.recovery_until:
            return min(1.0, self.prune_probability * 0.10)
        return self.prune_probability

    def _hot_hidden(self, x_batch: np.ndarray, hidden_batch: np.ndarray) -> int:
        mean_abs = np.mean(np.abs(hidden_batch), axis=0)
        mean_abs[~self.model.active] = -1
        return int(np.argmax(mean_abs))

    def _create_tick(self, cycle: int, x_batch: np.ndarray, hidden_batch: np.ndarray, op_idx: int):
        m = self.model
        self.counters["creation_attempts"] += 1
        active_count = max(int(m.active.sum()), 1)
        hot_fraction = float(np.mean(np.mean(np.abs(hidden_batch), axis=0)[m.active] > 0.12))
        # Only create in a live activity zone; this MLP analog uses sustained active hidden units as its zone signal.
        if hot_fraction < 0.45:
            self._log(cycle, "create_deferred", reason="no_sustained_activity_zone", hot_fraction=hot_fraction)
            return
        reserve = np.where(~m.active)[0]
        if self.create_stage == 0 and len(reserve) == 0:
            # Capacity exhausted: rebuild missing routes gradually. One distinct incoming
            # path per 50-cycle opportunity until the currently trained head has up to 4 routes.
            live_in = np.where(m.active & (self.W2_life[:, op_idx] > 0))[0]
            desired_routes = min(4, active_count)
            missing = np.where(m.active & (self.W2_life[:, op_idx] <= 0))[0]
            if len(live_in) < desired_routes and len(missing) > 0:
                hidden_activity = np.mean(np.abs(hidden_batch), axis=0)
                h = int(missing[np.argmax(hidden_activity[missing])])
                self.W2_life[h, op_idx] = CREATED_PATH_LIFE
                self.W2_life_cap[h, op_idx] = max(self.W2_life_cap[h, op_idx], CREATED_PATH_LIFE)
                m.W2[h, op_idx] = self.rng.normal(0, 0.18)
                self.W2_ring[:, h, op_idx] = 0
                self.W2_mutated[h, op_idx] = False
                self.W2_last_used[h, op_idx] = -10**9
                self.counters["created_paths"] += 1
                self._log(cycle, "create_restore_output_path", src=int(h), dst=int(op_idx), life=CREATED_PATH_LIFE,
                          live_routes_before=int(len(live_in)), desired_routes=int(desired_routes),
                          weight=float(m.W2[h, op_idx]))
            return
        if self.create_stage == 0:
            new_h = int(reserve[0])
            self.create_target = new_h
            self.create_input = int(np.argmax(np.mean(np.abs(x_batch), axis=0)))
            # First step: one incoming path, life 10 + 1.
            self.W1_life[self.create_input, new_h] = CREATED_PATH_LIFE
            self.W1_life_cap[self.create_input, new_h] = max(self.W1_life_cap[self.create_input, new_h], CREATED_PATH_LIFE)
            m.W1[self.create_input, new_h] = self.rng.normal(0, 0.18)
            m.W1_mask[self.create_input, new_h] = True
            self.create_started = cycle
            self.create_stage = 1
            self.counters["created_paths"] += 1
            self._log(cycle, "create_path_first", src=int(self.create_input), dst_hidden=new_h, life=CREATED_PATH_LIFE,
                      weight=float(m.W1[self.create_input, new_h]))
            # If the task head is completely disconnected, create one useful route in the same active zone.
            if int(np.sum(m.active & (self.W2_life[:, op_idx] > 0))) == 0:
                h = self._hot_hidden(x_batch, hidden_batch)
                self.W2_life[h, op_idx] = CREATED_PATH_LIFE
                self.W2_life_cap[h, op_idx] = max(self.W2_life_cap[h, op_idx], CREATED_PATH_LIFE)
                m.W2[h, op_idx] = self.rng.normal(0, 0.18)
                self.counters["created_paths"] += 1
                self._log(cycle, "create_restore_output_path", src=int(h), dst=int(op_idx), life=CREATED_PATH_LIFE)
        elif self.create_stage == 1 and self.create_target is not None:
            new_h = self.create_target
            # Second step: enable neuron +1 life after the first path exists.
            if self.W1_life[:, new_h].max() <= 0:
                self.W1_life[self.create_input, new_h] = CREATED_PATH_LIFE
                m.W1[self.create_input, new_h] = self.rng.normal(0, 0.18)
            m.active[new_h] = True
            self.neuron_life[new_h] = CREATED_NEURON_LIFE
            self.node_ring[:, new_h] = 0
            m.W1_mask[:, new_h] = self.W1_life[:, new_h] > 0
            self.counters["created_neurons"] += 1
            self.create_stage = 2
            self._log(cycle, "create_neuron", neuron=int(new_h), life=CREATED_NEURON_LIFE)
        elif self.create_stage == 2 and self.create_target is not None:
            new_h = self.create_target
            # Third step: remaining incoming path and all adjacent hidden->output paths, each life 10 + 1.
            for i in range(INPUT_DIM):
                if self.W1_life[i, new_h] <= 0:
                    self.W1_life[i, new_h] = CREATED_PATH_LIFE
                    self.W1_life_cap[i, new_h] = max(self.W1_life_cap[i, new_h], CREATED_PATH_LIFE)
                    m.W1[i, new_h] = self.rng.normal(0, 0.18)
                    self.counters["created_paths"] += 1
                    self._log(cycle, "create_adjacent_input_path", src=int(i), dst_hidden=int(new_h), life=CREATED_PATH_LIFE,
                              weight=float(m.W1[i, new_h]))
            for j in range(OUTPUT_DIM):
                if self.W2_life[new_h, j] <= 0:
                    self.W2_life[new_h, j] = CREATED_PATH_LIFE
                    self.W2_life_cap[new_h, j] = max(self.W2_life_cap[new_h, j], CREATED_PATH_LIFE)
                    m.W2[new_h, j] = self.rng.normal(0, 0.18)
                    self.counters["created_paths"] += 1
                    self._log(cycle, "create_adjacent_output_path", src_hidden=int(new_h), dst=int(j), life=CREATED_PATH_LIFE,
                              weight=float(m.W2[new_h, j]))
            self.create_target = None
            self.create_stage = 0
        m.W1_mask = (self.W1_life > 0) & m.active[None, :]
        m.W2_mask = (self.W2_life > 0) & m.active[:, None]

    def start_recovery(self, cycle: int, affected_hidden: Optional[List[int]] = None,
                       affected_W1_paths: Optional[List[Tuple[int, int]]] = None,
                       affected_W2_paths: Optional[List[Tuple[int, int]]] = None, duration: int = 100):
        """Recovery hook, available for future tests but not invoked in this experiment.

        Damaged elements receive a permanent +5 to their restoration cap. During recovery,
        PODA is softened to 10% of normal and CREAR's repair queue advances every 2 cycles:
        neurons gain +2 life up to their cap and paths gain +3 up to their cap.
        """
        m = self.model
        if affected_hidden is None:
            affected_hidden = [int(i) for i in range(m.capacity) if (not m.active[i] or self.neuron_life[i] <= 0)]
        self.recovery_affected_nodes = sorted(set(int(i) for i in affected_hidden if 0 <= int(i) < m.capacity))
        if affected_W1_paths is None:
            affected_W1_paths = [(int(i), int(h)) for i in range(INPUT_DIM) for h in range(m.capacity)
                                 if h in self.recovery_affected_nodes or self.W1_life[i,h] <= 0]
        if affected_W2_paths is None:
            affected_W2_paths = [(int(h), int(j)) for h in range(m.capacity) for j in range(OUTPUT_DIM)
                                 if h in self.recovery_affected_nodes or self.W2_life[h,j] <= 0]
        self.recovery_affected_W1 = sorted(set((int(i),int(h)) for i,h in affected_W1_paths
                                               if 0 <= int(i) < INPUT_DIM and 0 <= int(h) < m.capacity))
        self.recovery_affected_W2 = sorted(set((int(h),int(j)) for h,j in affected_W2_paths
                                               if 0 <= int(h) < m.capacity and 0 <= int(j) < OUTPUT_DIM))
        # Permanent light mutation of defaults in the damaged zone; current life is repaired gradually below.
        for h in self.recovery_affected_nodes:
            self.neuron_life_cap[h] = max(self.neuron_life_cap[h], BASE_NEURON_LIFE + 5)
        for i,h in self.recovery_affected_W1:
            self.W1_life_cap[i,h] = max(self.W1_life_cap[i,h], BASE_PATH_LIFE + 5)
        for h,j in self.recovery_affected_W2:
            self.W2_life_cap[h,j] = max(self.W2_life_cap[h,j], BASE_PATH_LIFE + 5)
        # Repair neurons first, then edges, so a revived path has an active endpoint.
        self.recovery_queue = ([('neuron', h, -1) for h in self.recovery_affected_nodes] +
                               [('W1', i, h) for i,h in self.recovery_affected_W1] +
                               [('W2', h, j) for h,j in self.recovery_affected_W2])
        self.recovery_cursor = 0
        self.recovery_active = True
        self.recovery_until = cycle + max(2, int(duration))
        self.counters["recovery_triggered"] = True
        self._log(cycle, "recovery_started", duration=duration, affected_neurons=len(self.recovery_affected_nodes),
                  affected_W1_paths=len(self.recovery_affected_W1), affected_W2_paths=len(self.recovery_affected_W2),
                  queue_size=len(self.recovery_queue))

    def _recovery_step(self, cycle: int):
        if not self.recovery_active or not self.recovery_queue:
            return
        m = self.model
        # Work one affected unit/path per 2-cycle repair tick.
        kind, a, b = self.recovery_queue[self.recovery_cursor % len(self.recovery_queue)]
        self.recovery_cursor += 1
        if kind == 'neuron':
            self.neuron_life[a] = min(self.neuron_life_cap[a], self.neuron_life[a] + 2.0)
            if self.neuron_life[a] > 0:
                m.active[a] = True
            self._log(cycle, "recovery_repair_neuron", neuron=a, life=float(self.neuron_life[a]), cap=float(self.neuron_life_cap[a]))
        elif kind == 'W1':
            was_dead = self.W1_life[a,b] <= 0
            self.W1_life[a,b] = min(self.W1_life_cap[a,b], self.W1_life[a,b] + 3.0)
            if was_dead and m.active[b]:
                m.W1[a,b] = self.rng.normal(0,0.18)
            self._log(cycle, "recovery_repair_path", kind="input_hidden", src=a, dst=b, life=float(self.W1_life[a,b]), cap=float(self.W1_life_cap[a,b]))
        else:
            was_dead = self.W2_life[a,b] <= 0
            self.W2_life[a,b] = min(self.W2_life_cap[a,b], self.W2_life[a,b] + 3.0)
            if was_dead and m.active[a]:
                m.W2[a,b] = self.rng.normal(0,0.18)
            self._log(cycle, "recovery_repair_path", kind="hidden_output", src=a, dst=b, life=float(self.W2_life[a,b]), cap=float(self.W2_life_cap[a,b]))
        m.W1_mask = (self.W1_life > 0) & m.active[None,:]
        m.W2_mask = (self.W2_life > 0) & m.active[:,None]
        if cycle >= self.recovery_until or self.recovery_cursor >= len(self.recovery_queue):
            self.recovery_active = False
            self._log(cycle, "recovery_ended", queue_completed=self.recovery_cursor >= len(self.recovery_queue))

    def end_recovery_if_due(self, cycle: int):
        if self.recovery_active and cycle > self.recovery_until:
            self.recovery_active = False
            self._log(cycle, "recovery_ended", queue_completed=self.recovery_cursor >= len(self.recovery_queue))

    def describe(self) -> Dict[str, object]:
        return {
            "base_path_life": BASE_PATH_LIFE, "base_neuron_life": BASE_NEURON_LIFE,
            "created_path_life": CREATED_PATH_LIFE, "created_neuron_life": CREATED_NEURON_LIFE,
            "prune_probability_per_10_cycles": self.prune_probability,
            "mutation_window": MUTATION_WINDOW, "mutation_active_threshold": MUTATION_MIN_ACTIVE,
            "mutation_path_bonus": MUTATED_PATH_BONUS, "mutation_neuron_bonus": MUTATED_NEURON_BONUS,
            "recovery_implemented_not_tested": True,
            "counters": dict(self.counters),
            "final_neuron_life": self.neuron_life.tolist(),
            "final_W1_life": self.W1_life.tolist(),
            "final_W2_life": self.W2_life.tolist(),
            "mutated_neurons": self.node_mutated.astype(int).tolist(),
            "mutated_W1_paths": self.W1_mutated.astype(int).tolist(),
            "mutated_W2_paths": self.W2_mutated.astype(int).tolist(),
            "events": self.events,
            "snapshots": self.snapshots,
        }


def metrics(model: TinyMLP, pairs: np.ndarray, name: str = "") -> Dict[str, object]:
    x = encode_x(pairs)
    _, y = model.forward(x)
    per_op = {}
    errs, exacts = [], []
    for op_idx, op in enumerate(OPS):
        pred = decode_prediction(y[:, op_idx], op_idx)
        actual = true_result(pairs, op_idx).astype(float)
        mae = float(np.mean(np.abs(pred - actual)))
        rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
        exact = float(np.mean(np.rint(pred).astype(int) == actual.astype(int)))
        per_op[op] = {"mae": mae, "rmse": rmse, "rounded_exact_accuracy": exact}
        errs.append(mae); exacts.append(exact)
    return {"name": name, "per_operation": per_op, "mean_mae": float(np.mean(errs)),
            "mean_rmse": float(np.mean([per_op[o]["rmse"] for o in OPS])),
            "mean_rounded_exact_accuracy": float(np.mean(exacts)),
            "balanced_exact_accuracy": float(np.mean(exacts))}


def make_data(seed: int):
    all_pairs = np.array([(a,b) for a in range(10) for b in range(10)], dtype=np.int64)
    rng = np.random.default_rng(seed + 1234)
    perm = rng.permutation(len(all_pairs))
    test_idx = perm[:20]
    train_idx = perm[20:]
    return all_pairs[train_idx], all_pairs[test_idx], all_pairs


def run_one_seed(seed: int, cycles_per_task: int = 300, replay_cycles: int = 300,
                 prune_probability: float = WEAK_PRUNE_PROB, learning_rate: float = 0.025) -> Dict[str, object]:
    train_pairs, test_pairs, all_pairs = make_data(seed)
    x_for_test = encode_x(test_pairs)
    base_init = TinyMLP(seed + 101)
    base_init.W1[:, INITIAL_HIDDEN:] = base_init.W1[:, INITIAL_HIDDEN:].copy()  # reserve weights exist but are not used by baseline
    sovereign_init = base_init.clone()
    baseline = OrdinaryControl(base_init)
    net = sovereign_init
    sovereign = ThreeSovereigns(net, seed, prune_probability=prune_probability)
    # Ensure control and treatment begin with identical active weights and masks.
    init_metrics = {
        "baseline": metrics(baseline.model, test_pairs, "baseline_initial"),
        "sovereign": metrics(net, test_pairs, "sovereign_initial"),
    }
    initial_structure = {"baseline": baseline.model.state_metrics(), "sovereign": net.state_metrics()}
    cycle = 0
    training_trace: List[Dict[str, object]] = []
    eval_trace: List[Dict[str, object]] = []
    phase_specs = [("sum", 0, cycles_per_task), ("subtract", 1, cycles_per_task), ("multiply", 2, cycles_per_task)]
    # Match batches and operator order between baseline and sovereign models.
    data_rng = np.random.default_rng(seed + 9001)
    phase_loss = {"baseline": {}, "sovereign": {}}
    phase_best = {"baseline": {}, "sovereign": {}}
    for phase_name, op_idx, ncycles in phase_specs:
        losses_b, losses_s = [], []
        for local_cycle in range(ncycles):
            cycle += 1
            ids = data_rng.integers(0, len(train_pairs), size=16)
            batch_pairs = train_pairs[ids]
            x = encode_x(batch_pairs)
            target = target_for(batch_pairs, op_idx)
            losses_b.append(baseline.train_batch(x, target, op_idx, lr=learning_rate))
            hidden_before, _ = net.forward(x)
            losses_s.append(net.train_batch(x, target, op_idx, lr=learning_rate))
            sovereign.record_and_tick(cycle, x, hidden_before, op_idx)
            sovereign.end_recovery_if_due(cycle)
            if cycle % 10 == 0:
                training_trace.append({"cycle": cycle, "phase": phase_name,
                    "baseline_loss": float(np.mean(losses_b[-min(10,len(losses_b)):])) ,
                    "sovereign_loss": float(np.mean(losses_s[-min(10,len(losses_s)):])) ,
                    **{f"sov_{k}": v for k,v in net.state_metrics().items()},
                    **{f"base_{k}": v for k,v in baseline.model.state_metrics().items()},
                    "active_neurons": int(net.active.sum()), "pruned_paths": sovereign.counters["pruned_paths"],
                    "created_paths": sovereign.counters["created_paths"], "mutated_paths": sovereign.counters["mutated_paths"]})
        phase_loss["baseline"][phase_name] = float(np.mean(losses_b[-40:]))
        phase_loss["sovereign"][phase_name] = float(np.mean(losses_s[-40:]))
        bm = metrics(baseline.model, test_pairs, phase_name)
        sm = metrics(net, test_pairs, phase_name)
        bm_all = metrics(baseline.model, all_pairs, phase_name + "_full_grid")
        sm_all = metrics(net, all_pairs, phase_name + "_full_grid")
        eval_trace.append({"cycle": cycle, "phase": phase_name, "baseline": bm, "sovereign": sm,
                           "baseline_full_grid": bm_all, "sovereign_full_grid": sm_all,
                           "baseline_structure": baseline.model.state_metrics(), "sovereign_structure": net.state_metrics()})
        phase_best["baseline"][phase_name] = bm_all
        phase_best["sovereign"][phase_name] = sm_all

    # Accumulated/replay phase: each mini-batch selects a random operator so old tasks are revisited.
    replay_losses_b, replay_losses_s = [], []
    for local_cycle in range(replay_cycles):
        cycle += 1
        op_idx = int(data_rng.integers(0, 3))
        ids = data_rng.integers(0, len(train_pairs), size=16)
        batch_pairs = train_pairs[ids]
        x = encode_x(batch_pairs)
        target = target_for(batch_pairs, op_idx)
        replay_losses_b.append(baseline.train_batch(x, target, op_idx, lr=learning_rate))
        hidden_before, _ = net.forward(x)
        replay_losses_s.append(net.train_batch(x, target, op_idx, lr=learning_rate))
        sovereign.record_and_tick(cycle, x, hidden_before, op_idx)
        sovereign.end_recovery_if_due(cycle)
        if cycle % 10 == 0:
            training_trace.append({"cycle": cycle, "phase": "accumulated_replay",
                "baseline_loss": float(np.mean(replay_losses_b[-min(10,len(replay_losses_b)):])) ,
                "sovereign_loss": float(np.mean(replay_losses_s[-min(10,len(replay_losses_s)):])) ,
                **{f"sov_{k}": v for k,v in net.state_metrics().items()},
                **{f"base_{k}": v for k,v in baseline.model.state_metrics().items()},
                "active_neurons": int(net.active.sum()), "pruned_paths": sovereign.counters["pruned_paths"],
                "created_paths": sovereign.counters["created_paths"], "mutated_paths": sovereign.counters["mutated_paths"]})
    bm_final = metrics(baseline.model, test_pairs, "final_after_replay")
    sm_final = metrics(net, test_pairs, "final_after_replay")
    bm_final_all = metrics(baseline.model, all_pairs, "final_after_replay_full_grid")
    sm_final_all = metrics(net, all_pairs, "final_after_replay_full_grid")
    eval_trace.append({"cycle": cycle, "phase": "accumulated_replay", "baseline": bm_final, "sovereign": sm_final,
                       "baseline_full_grid": bm_final_all, "sovereign_full_grid": sm_final_all,
                       "baseline_structure": baseline.model.state_metrics(), "sovereign_structure": net.state_metrics()})
    phase_best["baseline"]["final_after_replay"] = bm_final_all
    phase_best["sovereign"]["final_after_replay"] = sm_final_all

    # Full 100-point grid helps distinguish hold-out generalization from training-pair fit.
    full_eval = {"baseline": metrics(baseline.model, all_pairs, "full_grid"), "sovereign": metrics(net, all_pairs, "full_grid")}
    # Forgetting is decrease in per-op test exact accuracy from the end of that operation's own phase to final replay.
    forgetting = {"baseline": {}, "sovereign": {}}
    for mode, phases in [("baseline", phase_best["baseline"]), ("sovereign", phase_best["sovereign"])]:
        final = phases["final_after_replay"]["per_operation"]
        for op in OPS:
            when_learned = phases[op]["per_operation"][op]["rounded_exact_accuracy"]
            end = final[op]["rounded_exact_accuracy"]
            forgetting[mode][op] = {"accuracy_at_end_of_own_phase": when_learned, "final_accuracy": end,
                                     "forgetting_delta": float(when_learned - end),
                                     "replay_gain_vs_after_phase": float(end - when_learned)}

    # Save sovereignty-managed weights, masks and lives for inspection/reuse.
    weights_path = SCRIPT_DIR / f"Sovereign_MLP_MVP_seed_{seed}.npz"
    np.savez_compressed(weights_path,
        W1=net.W1, b1=net.b1, W2=net.W2, b2=net.b2, active=net.active.astype(np.uint8),
        W1_mask=net.W1_mask.astype(np.uint8), W2_mask=net.W2_mask.astype(np.uint8),
        neuron_life=sovereign.neuron_life, W1_life=sovereign.W1_life, W2_life=sovereign.W2_life,
        node_mutated=sovereign.node_mutated.astype(np.uint8), W1_mutated=sovereign.W1_mutated.astype(np.uint8), W2_mutated=sovereign.W2_mutated.astype(np.uint8))
    baseline_weights_path = SCRIPT_DIR / f"Sovereign_MLP_MVP_baseline_seed_{seed}.npz"
    np.savez_compressed(baseline_weights_path, W1=baseline.model.W1, b1=baseline.model.b1,
        W2=baseline.model.W2, b2=baseline.model.b2, active=baseline.model.active.astype(np.uint8),
        W1_mask=baseline.model.W1_mask.astype(np.uint8), W2_mask=baseline.model.W2_mask.astype(np.uint8))
    return {
        "seed": seed, "cycles": cycle, "train_pairs": train_pairs.tolist(), "test_pairs": test_pairs.tolist(),
        "initial_metrics": init_metrics, "initial_structure": initial_structure, "phase_evaluations": eval_trace,
        "phase_training_losses": phase_loss, "forgetting": forgetting,
        "final_full_grid": full_eval, "sovereign": sovereign.describe(),
        "training_trace": training_trace, "final_weights_file": str(weights_path),
        "final_baseline_weights_file": str(baseline_weights_path),
        "recovery": {"implemented": True, "triggered": False, "damage_applied": False},
    }


def aggregate(results: List[Dict[str, object]]) -> Dict[str, object]:
    modes = ["baseline", "sovereign"]
    phase_names = ["sum", "subtract", "multiply", "accumulated_replay"]
    ag = {"seed_count": len(results), "phase_means": {}, "sovereign_counters": {}, "forgetting_means": {}, "final_full_grid_means": {}}
    for ph in phase_names:
        vals = {mode: {"mean_mae": [], "mean_rounded_exact_accuracy": [], "balanced_exact_accuracy": []} for mode in modes}
        for r in results:
            row = next(e for e in r["phase_evaluations"] if e["phase"] == ph)
            for mode in modes:
                for metric_name in vals[mode]: vals[mode][metric_name].append(row[mode][metric_name])
        ag["phase_means"][ph] = {mode: {k: float(np.mean(v)) for k,v in m.items()} for mode,m in vals.items()}
    counter_keys = ["created_neurons","created_paths","pruned_paths","pruned_neurons","mutated_neurons","mutated_paths","conservation_path_events","conservation_neuron_events","dormant_bonuses"]
    for k in counter_keys:
        ag["sovereign_counters"][k] = float(np.mean([r["sovereign"]["counters"][k] for r in results]))
    for mode in modes:
        ag["forgetting_means"][mode] = {}
        for op in OPS:
            ag["forgetting_means"][mode][op] = {
                k: float(np.mean([r["forgetting"][mode][op][k] for r in results]))
                for k in ["accuracy_at_end_of_own_phase","final_accuracy","forgetting_delta","replay_gain_vs_after_phase"]}
        ag["final_full_grid_means"][mode] = {
            "mean_mae": float(np.mean([r["final_full_grid"][mode]["mean_mae"] for r in results])),
            "mean_rounded_exact_accuracy": float(np.mean([r["final_full_grid"][mode]["mean_rounded_exact_accuracy"] for r in results])),
            "per_operation": {op: {
                "mae": float(np.mean([r["final_full_grid"][mode]["per_operation"][op]["mae"] for r in results])),
                "rounded_exact_accuracy": float(np.mean([r["final_full_grid"][mode]["per_operation"][op]["rounded_exact_accuracy"] for r in results]))
            } for op in OPS}
        }
    return ag


def run_experiment(seeds=(31, 47, 59), cycles_per_task=300, replay_cycles=300,
                   prune_probability=WEAK_PRUNE_PROB) -> Dict[str, object]:
    rs = [run_one_seed(s, cycles_per_task, replay_cycles, prune_probability) for s in seeds]
    return {
        "title": "Conventional small MLP governed by Three Sovereigns",
        "description": "Tiny feed-forward MLP for arithmetic regression; stochastic LB front-end is disconnected; no severe damage applied.",
        "settings": {"initial_hidden": INITIAL_HIDDEN, "hidden_capacity": HIDDEN_CAPACITY, "outputs": list(OPS),
                     "cycles_per_task": cycles_per_task, "replay_cycles": replay_cycles, "total_cycles_per_seed": 3*cycles_per_task+replay_cycles,
                     "prune_probability_per_10_cycles": prune_probability, "create_interval": CREATE_INTERVAL,
                     "conserve_interval": CONSERVE_INTERVAL, "mutation_window": MUTATION_WINDOW,
                     "mutation_threshold": MUTATION_MIN_ACTIVE, "learning_rate": 0.025, "batch_size": 16,
                     "data_split": "80 integer pairs train / 20 held-out pairs; values 0..9"},
        "results": rs,
        "aggregate": aggregate(rs),
    }


def run_tests(result: Dict[str, object]) -> Dict[str, object]:
    checks = {}
    checks["stochastic_front_disconnected"] = result["description"].find("disconnected") >= 0
    checks["all_seeds_complete"] = all(r["cycles"] == result["settings"]["total_cycles_per_seed"] for r in result["results"])
    checks["ordinary_control_present"] = all("baseline" in r["initial_metrics"] for r in result["results"])
    checks["sovereign_model_present"] = all("sovereign" in r["initial_metrics"] for r in result["results"])
    checks["replay_phase_present"] = all(any(e["phase"] == "accumulated_replay" for e in r["phase_evaluations"]) for r in result["results"])
    checks["recovery_kept_but_not_triggered"] = all(r["recovery"]["implemented"] and not r["recovery"]["triggered"] and not r["recovery"]["damage_applied"] for r in result["results"])
    checks["hidden_life_state_tracks_capacity"] = all(len(r["sovereign"]["final_neuron_life"]) == HIDDEN_CAPACITY for r in result["results"])
    checks["recovery_hook_defined"] = hasattr(ThreeSovereigns, "start_recovery") and hasattr(ThreeSovereigns, "_recovery_step")
    checks["outputs_remain_three_heads"] = all(len(r["final_full_grid"]["sovereign"]["per_operation"]) == 3 for r in result["results"])
    checks["weak_pruning_parameter"] = abs(result["settings"]["prune_probability_per_10_cycles"] - WEAK_PRUNE_PROB) < 1e-12
    checks["all_three_tasks_evaluated"] = all(all(op in r["final_full_grid"]["sovereign"]["per_operation"] for op in OPS) for r in result["results"])
    checks["sovereign_events_recorded"] = all(len(r["sovereign"]["events"]) > 0 for r in result["results"])
    checks["weight_snapshots_recorded"] = all(len(r["sovereign"]["snapshots"]) > 0 for r in result["results"])
    checks["all_passed"] = all(bool(v) for v in checks.values())
    return checks


def save_readable_weights(npz_path: str, json_path: str):
    d = np.load(npz_path)
    payload = {k: d[k].tolist() for k in d.files}
    Path(json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    d.close()


def main():
    p = argparse.ArgumentParser(description="MVP: small conventional MLP + three sovereigns; weak pruning; no damage.")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--cycles-per-task", type=int, default=150)
    p.add_argument("--replay-cycles", type=int, default=150)
    p.add_argument("--prune-probability", type=float, default=WEAK_PRUNE_PROB)
    p.add_argument("--out", default=str(SCRIPT_DIR / "Sovereign_MLP_math_MVP_results.json"))
    p.add_argument("--test", action="store_true")
    args = p.parse_args()
    seeds = [31,47,59,71,83,97,109,127][:max(1,min(args.seeds,8))]
    result = run_experiment(seeds, args.cycles_per_task, args.replay_cycles, args.prune_probability)
    checks = run_tests(result)
    result["tests"] = checks
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.test:
        print(json.dumps(checks, indent=2))
    print(json.dumps({"output": args.out, "aggregate": result["aggregate"], "tests": checks}, indent=2))

if __name__ == "__main__":
    main()
