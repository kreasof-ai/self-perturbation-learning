> Note: This results are based on `kreasof-ai/MATH-WD-Lite` test set, the evaluation script can be seen inside `/notebook/evaluation` directory.

---

### Supervised baseline

- Model: `RLHFlow/Llama3.1-8B-PRM-Deepseek-Data`
- Results:

```
Accuracy: 0.2750
Mean Δ(max wrong): -0.1199
Mean Δ(avg wrong): -0.0037

Length-Score Correlation: 0.0259

Level vs Accuracy Correlation: -0.3000
Level vs Δ(max wrong) Correlation: -0.4000

Accuracy by Level:
Level 1: 0.3333
Level 2: 0.2973
Level 3: 0.2258
Level 4: 0.2368
Level 5: 0.3077

Δ(max wrong) by Level:
Level 1: -0.0474
Level 2: -0.1157
Level 3: -0.1394
Level 4: -0.1621
Level 5: -0.0951
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
Accuracy: 0.3063
Mean Δ(max wrong): -0.1294
Mean Δ(avg wrong): -0.0131

Length-Score Correlation: -0.0093

Level vs Accuracy Correlation: -0.3000
Level vs Δ(max wrong) Correlation: -0.3000

Accuracy by Level:
Level 1: 0.3333
Level 2: 0.3514
Level 3: 0.3226
Level 4: 0.3684
Level 5: 0.1795

Δ(max wrong) by Level:
Level 1: -0.0403
Level 2: -0.2312
Level 3: -0.0806
Level 4: -0.1329
Level 5: -0.1025
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
