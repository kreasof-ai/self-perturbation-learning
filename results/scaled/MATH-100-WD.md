> Note: This results are based on early version of MATH-WD benchmark (only consist of 100 test sample). The updated benchmark consist larger sample.

---

### Supervised baseline

- Model: `RLHFlow/Llama3.1-8B-PRM-Deepseek-Data`
- Results:

```
Accuracy: 0.2300
Mean Δ(max wrong): -0.2058
Mean Δ(avg wrong): -0.0354

Length-Score Correlation: -0.0005

Level vs Accuracy Correlation: -0.2000
Level vs Δ(max wrong) Correlation: 0.0000

Accuracy by Level:
Level 1: 0.4545
Level 2: 0.1200
Level 3: 0.1053
Level 4: 0.3636
Level 5: 0.2174

Δ(max wrong) by Level:
Level 1: -0.0525
Level 2: -0.3010
Level 3: -0.2881
Level 4: -0.2079
Level 5: -0.1055
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
Accuracy: 0.3000
Mean Δ(max wrong): -0.0733
Mean Δ(avg wrong): 0.0401

Length-Score Correlation: -0.0141

Level vs Accuracy Correlation: -0.9000
Level vs Δ(max wrong) Correlation: 0.1000

Accuracy by Level:
Level 1: 0.3636
Level 2: 0.4000
Level 3: 0.3158
Level 4: 0.2727
Level 5: 0.1739

Δ(max wrong) by Level:
Level 1: -0.0651
Level 2: -0.0646
Level 3: -0.0731
Level 4: -0.1024
Level 5: -0.0587
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
Accuracy: 0.3600
Mean Δ(max wrong): -0.2243
Mean Δ(avg wrong): 0.0742

Length-Score Correlation: -0.0320

Level vs Accuracy Correlation: 0.9000
Level vs Δ(max wrong) Correlation: 0.5000

Accuracy by Level:
Level 1: 0.2727
Level 2: 0.2400
Level 3: 0.3158
Level 4: 0.3636
Level 5: 0.5652

Δ(max wrong) by Level:
Level 1: -0.3149
Level 2: -0.1775
Level 3: -0.4598
Level 4: -0.1812
Level 5: -0.0785
```