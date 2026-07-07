# NodeSense Research Log

A running record of progress, decisions, and results. Update before each advisor meeting.

## 2026-07-06 — Working end-to-end demo system

Built the full pipeline on synthetic CICIDS-style data so every layer is
real and demonstrable before the dataset download:

- `data.py` generates flow sessions for benign traffic + 5 attack classes,
  with distributions modeled on each attack's flow-level signature, and a
  loader that maps real CICIDS-2018 CSVs onto the same 20 features.
- Benchmarks on synthetic test set (binary anomaly detection): random
  forest AUC 0.998, autoencoder AUC 0.959, transformer saturates (1.000 —
  expected on synthetic data; real CICIDS numbers are the ones to report).
- Transformer exported to ONNX (335KB, committed) and served by FastAPI
  with per-request KernelSHAP explanations (~120ms/explanation on CPU).
- Dashboard streams model-classified alerts over WebSocket; clicking an
  alert requests SHAP values for that alert's actual feature vector.

Decision: torch's new dynamo ONNX exporter specializes attention reshapes
to the traced batch size; exported with `dynamo=False` (legacy exporter)
to get a dynamic batch dimension, which KernelSHAP needs.

## Week 1 to 2

**Goal:** Dataset acquisition, exploratory analysis, environment setup.

- [ ] Download CICIDS-2018 dataset
- [ ] Set up local environment and repo
- [ ] Run initial EDA, document class balance and feature distributions
- [ ] Complete annotated bibliography

**Findings:**

**Decisions:**

**Blockers:**

---

## Week 3 to 4

**Goal:** Preprocessing pipeline, baseline models.

- [ ] Build feature engineering pipeline
- [ ] Train random forest baseline
- [ ] Train autoencoder baseline
- [ ] Record baseline F1 and AUC-ROC

**Findings:**

**Decisions:**

**Blockers:**

---

## Week 5 to 6

**Goal:** Transformer model.

- [ ] Implement transformer classifier
- [ ] Hyperparameter search (layers, heads, sequence length)
- [ ] Compare against baselines

**Findings:**

**Decisions:**

**Blockers:**

---

## Week 7

**Goal:** Explainability.

- [ ] Integrate SHAP
- [ ] Analyze attention vs SHAP correlation
- [ ] Evaluate explanation quality

**Findings:**

**Decisions:**

**Blockers:**

---

## Week 8 to 9

**Goal:** Deployment.

- [ ] Export model to ONNX
- [ ] Deploy backend to Hugging Face Spaces
- [ ] Benchmark inference latency

**Findings:**

**Decisions:**

**Blockers:**

---

## Week 10 to 11

**Goal:** Dashboard and write-up.

- [ ] Build React dashboard
- [ ] Deploy frontend
- [ ] Final paper write-up
- [ ] Prepare presentation

**Findings:**

**Decisions:**

**Blockers:**
