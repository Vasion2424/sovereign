# sovereign
# MVP: Conventional MLP Governed by the Three Sovereigns

## Objective

A short, reproducible benchmark designed to evaluate whether **CREATE, CONSERVE/MUTATE, and weak PRUNE** can govern the dynamic structure of a conventional neural network during basic mathematical learning tasks. The stochastic LB generator is fully decoupled. **No artificial damage was applied** (neither severe nor mild); the RECOVERY mechanism is implemented but remained inactive throughout this run.

## Minimal Configuration

- Feedforward MLP architecture: **2 inputs → 8 active hidden neurons (maximum capacity: 12) → 3 regression outputs**.
- Outputs represent addition, subtraction, and multiplication of integers in the range [0, 9]. Each output head is trained sequentially per phase while sharing the hidden layer.
- Conventional supervised training via mini-batches and backpropagation; learning rate: 0.025; batch size: 16.
- **3 seeds × 600 cycles per seed**: 150 addition cycles, 150 subtraction cycles, 150 multiplication cycles, and 150 accumulated replay cycles. Total: 1,800 cycles per model variant across the three seeds.
- Seed-stratified datasets: 80 pairs allocated for training and 20 held-out pairs reserved for evaluation. The complete 100-pair grid is also reported; however, it includes training examples and should not be interpreted as an independent test set.
- Baseline control: Fixed 8-neuron MLP initialized with identical weights and training data. Governed variant: CREATE mechanism permitted topological expansion from 8 up to 12 hidden neurons.

## Governed Rules in Effect

- **CREATE:** Evaluates activation every 50 cycles given sufficient activity. Appends reserve neurons in stages (path, neuron, adjacent paths); if reserve capacity is exhausted, it incrementally reconstructs a path toward an active output with sparse connections.
- **CONSERVE:** Every 10 cycles, appends 1 life point to paths/neurons utilized in at least 8 of the preceding 10 cycles.
- **MUTATE:** 200-cycle observation window; a unit/path registering at least 160 active uses receives a one-time allocation of +30 life points (neuron) or +60 life points (path). Reactivation bonuses after prolonged inactivity are also preserved.
- **Weak PRUNE:** Every 10 cycles, an unutilized connection within that window has a **25% probability** of losing 1 life point. No deliberate structural damage was introduced.
- **RECOVERY:** Code available (gradual repair every 2 cycles, temporary soft pruning), but remained unactivated.

## Results

### Post-Replay Accuracy

| Metric | Fixed MLP | Governed MLP (Sovereigns) |
|---|---:|---:|
| Mean Accuracy on 20 Held-out Pairs / Seed | 43.33% | 42.22% |
| Mean MAE on Held-out Pairs | 1.716 | 1.689 |
| Mean Accuracy on Full 100-Pair Grid | 39.78% | 39.44% |
| Mean MAE on Full Grid | 1.688 | 1.699 |

Across the full grid, performance is virtually identical: the fixed MLP achieves 39.78% compared to 39.44% for the governed variant. On reserved evaluation pairs, the fixed MLP exhibits marginally higher mean accuracy, whereas the governed variant yields a slightly lower MAE. **There is no evidence of overall superiority for the governed framework within this short sample.**

### Phase Progress (Mean Accuracy on Held-out Pairs)

| After Phase | Fixed MLP | Governed MLP |
|---|---:|---:|
| Post-Addition | 22.78% | 22.22% |
| Post-Subtraction | 40.56% | 40.00% |
| Post-Multiplication | 37.22% | 31.67% |
| Accumulated Replay | 43.33% | 42.22% |

These figures aggregate all three output heads per phase, including heads that have not yet undergone training during early phases. Performance progression post-replay improves total evaluation metrics across both models, though it does not conclusively prove the mitigation of catastrophic forgetting between tasks.

### Structural Evolution and Weight Dynamics

- Active hidden neurons in the governed variant: **8 → 12.0 out of maximum 12**.
- Mean final connected paths: **40.0** in the fixed MLP vs. **58.3** in the governed variant.
- Average events per seed: **4.0 neurons created**, **20.0 paths created/restored**, **1.7 paths pruned**, **9.3 neurons mutated**, and **27.7 paths mutated**.
- Effective Weight L2 Norm: **4.034 initially → 3.646 with Sovereigns**; Final Fixed MLP: **4.103**. Masked/dead weights do not contribute to the effective norm calculation.
- No hidden neurons were removed during this short evaluation sample. Weak pruning operated predominantly on individual connections.

![Accuracy, Structure, and Weight Evolution](assets/Sovereign_MLP_MVP_evolution.png)

![Final Connected Weights for Seed 31](assets/Sovereign_MLP_MVP_weights.png)

## Verification and Limitations

- **13/13 automated checks PASS**.
- RECOVERY state was verified as available and unactivated.
- No artificial structural damage was introduced at any level.
- Three seeds and 150 updates per operation constitute a smoke test / MVP, not sufficient statistical evidence.
- Multiplication represents the most challenging task: final full-grid accuracy stabilized near 10.7% under the governed variant. Small network capacity and brief training schedules bound the conclusions that can be drawn.
- The architecture features three distinct regression heads trained sequentially per phase; it does not take an operational selector as an input token.

## Conclusion

The MVP demonstrates that the three Sovereigns can successfully supervise connection growth and survival in a small conventional MLP during task execution. However, in this brief run, final mathematical performance remains equivalent to the fixed control model; we cannot currently claim that the Sovereigns reduce catastrophic forgetting or enhance generalization. The most clearly observed effect is structural: four new neurons were instantiated and overall connectivity expanded, while weakened pruning removed minimal paths. **No damage was applied, and recovery was not actively tested.**

## Reproducibility

```bash
python Sovereign_MLP_math_MVP.py --test
python Sovereign_MLP_math_MVP.py --seeds 3 --cycles-per-task 150 --replay-cycles 150 --test --out Sovereign_MLP_math_MVP_results.json
