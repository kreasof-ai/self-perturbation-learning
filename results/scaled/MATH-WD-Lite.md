> Note: This results are based on `kreasof-ai/MATH-WD-Lite` test set, the evaluation script can be seen inside `/notebook/evaluation` directory.

---

### Supervised baseline

- Model: `RLHFlow/Llama3.1-8B-PRM-Deepseek-Data`
- Results:

```
Accuracy: 0.2750
Mean Δ(max wrong): -0.4024
Mean Δ(avg wrong): -0.1223

Length-Score Correlation: 0.0622

Level vs Accuracy Correlation: -0.1000
Level vs Δ(max wrong) Correlation: -0.1000

Accuracy by Level:
Level 1: 0.2667
Level 2: 0.3514
Level 3: 0.1290
Level 4: 0.2632
Level 5: 0.3333

Δ(max wrong) by Level:
Level 1: -0.2493
Level 2: -0.5187
Level 3: -0.4840
Level 4: -0.3258
Level 5: -0.3607
```

---

### SPL 2 million

- Base model: `answerdotai/ModernBERT-large`
- Total row: 2 million
- Data generation: Embedding guided perturbation
- Embedding model: `minishlab/potion-base-2M`
- Checkpoints: `kreasof-ai/SPL-Large-Experimental`
- Results:

```
Accuracy: 0.3438
Mean Δ(max wrong): -0.1281
Mean Δ(avg wrong): -0.0148

Length-Score Correlation: -0.0116

Level vs Accuracy Correlation: -0.7000
Level vs Δ(max wrong) Correlation: -0.3000

Accuracy by Level:
Level 1: 0.4000
Level 2: 0.3514
Level 3: 0.3871
Level 4: 0.3684
Level 5: 0.2564

Δ(max wrong) by Level:
Level 1: -0.0424
Level 2: -0.1701
Level 3: -0.1028
Level 4: -0.1580
Level 5: -0.1123
```

---

### SPL 100K (LLM Perturbation)

- Base model: `answerdotai/ModernBERT-base`
- Total row: 100000
- Data generation: LLM guided perturbation
- LLM: `gemini-2.0-flash-lite`
- Checkpoints: `kreasof-ai/SPL-better-dataset-base-f32`
- Results:

```
Accuracy: 0.3812
Mean Δ(max wrong): -0.3965
Mean Δ(avg wrong): -0.0940

Length-Score Correlation: 0.0286

Level vs Accuracy Correlation: -1.0000
Level vs Δ(max wrong) Correlation: -1.0000

Accuracy by Level:
Level 1: 0.6000
Level 2: 0.4865
Level 3: 0.4194
Level 4: 0.3158
Level 5: 0.2308

Δ(max wrong) by Level:
Level 1: 0.0429
Level 2: -0.0100
Level 3: -0.2590
Level 4: -0.3063
Level 5: -1.1293
```
