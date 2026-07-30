# TowerWatch — GNN Anomaly Detection for Lattice Tower Structural Monitoring

> **Working spec for Claude Code.** Build target: 5 days (~35 hours). Author: Duy Le.
> Purpose: interview artefact for Research Associate, Project SMaRTER, UTS Telecommunications Research Institute (interview Tue 4 Aug 2026).

---

## 0. Project intent (read this first)

**Context that shapes the framing.** Amplitel and UTS TRU have already delivered Project SMaRTER's Milestone 1 — a "fingerprint model" characterising each tower's baseline condition, built from LLMs/AI agents mining historical maintenance text, linked to material degradation models, structural reliability frameworks, and FEM models. It won Best Telco AI Initiative at the 2026 CommsDay Edison Awards. The publicly stated next phase is Milestones 2–4: AI-enhanced monitoring and maintenance scheduling, real-time sensor/camera-based "smart towers," and robotics.

**This project is deliberately not a fingerprint model.** It targets the layer that comes *after* one: given a tower's baseline is already characterised, use live sensor data to detect and localise deviation from that baseline, and turn that signal into a maintenance-priority output. That's the Milestone 1→2/3 boundary — complementary to the existing LLM/text/FEM approach, not a smaller rebuild of it.

Detect and **localise** structural damage in a steel lattice tower from vibration sensor data, using a graph neural network that treats the tower's own structural topology as the graph. Then turn that signal into a simple maintenance-priority ranking — the connective tissue toward Milestone 2.

Three-stage narrative, deliberately in this order:

1. **Synthetic stage** — build a physics-based 3D truss FE model of a lattice mast, inject damage as brace stiffness reduction, simulate accelerometer response. Full pipeline validated end-to-end on data we control.
2. **Real stage** — transfer the *same* pipeline to **LUMO**, a real open-access 9 m steel lattice mast benchmark with physically induced (removable brace) damage and a year of outdoor environmental variability.
3. **Prioritisation stage** — combine anomaly score, localisation, and severity trend into a ranked maintenance-priority output across multiple tower/damage instances, with a naive inspection-cost-vs-risk tradeoff. Small in scope, but it's what makes the project legibly about Milestone 2 rather than only Milestone 1/3.

This mirrors the sim-to-real problem Project SMaRTER will actually face: abundant simulation, scarce real labelled failure data.

**Architecture note.** Design the feature/scoring pipeline so each window is scored independently (no dependence on batch statistics beyond the healthy-fit scaler). This isn't implemented as a live streaming service, but the architecture should honestly support the claim "this could sit behind a live feed" — relevant to Milestone 3's real-time framing.

**Non-goals.** No Isaac Sim (hardware-incompatible — see §1). No Gazebo robot patrol (cut for scope — see §9). No LLM/text-mining component (that's the existing fingerprint model's job, not this project's — don't scope-creep into duplicating it). No production deployment. No claim this is novel research.

---

## 1. Verified constraints

| Constraint | Status | Notes |
|---|---|---|
| GPU: GTX 1050 Ti, 4 GB VRAM | **Fine** | Graph is ~20–60 nodes. Trains in minutes on CPU alone. No Colab needed. |
| Isaac Sim | **Excluded** | Min spec is RTX 3070 / 8 GB VRAM; GPUs without RT cores are explicitly unsupported. GTX 1050 Ti fails both. Do not attempt. |
| PyTorch Geometric | **Avoid as hard dependency** | Install is historically brittle (CUDA/torch version pinning). Implement GCN layers directly with a dense normalised adjacency matmul — ~15 lines, and correct at this graph size. PyG optional stretch only. |
| LUMO dataset | **Available, CC-BY 3.0** | DOI 10.25835/0027803. Individual exemplary ZIPs ≈ 530–650 MB each. Full set 3.6 GB. Healthy + DAM3/DAM4/DAM6 damage states. Healthy-state FE model (.inp) also provided. |
| Disk space | **Check before starting** | Need ~6 GB free. Duy recently had a full C drive — verify WSL2 has headroom on day 1, not day 4. |
| Time | 5 days, Mon 27 – Fri 31 Jul | Leaves Sat 1 – Mon 3 Aug for interview prep and travel. |

---

## 2. Stack

```
python 3.10+
numpy, scipy          # FE model, modal analysis, signal generation
torch                 # GNN (CPU is sufficient)
scikit-learn          # PCA / IsolationForest baselines, metrics
pandas, matplotlib
h5py or pyarrow       # only if LUMO format requires it — check README first
```

No PyG, no Lightning, no hydra. Keep the dependency surface small enough that a reviewer can `pip install -r requirements.txt` and run it.

---

## 3. Repository layout

```
towerwatch/
├── README.md                  # the deliverable — written as a mini-paper
├── requirements.txt
├── configs/
│   └── default.yaml
├── src/
│   ├── fem/
│   │   ├── truss.py           # 3D space truss assembly + modal analysis
│   │   ├── geometry.py        # lattice mast geometry generator
│   │   └── damage.py          # damage injection (brace stiffness reduction)
│   ├── simulate/
│   │   ├── response.py        # modal superposition → acceleration time histories
│   │   └── environment.py     # temperature-driven stiffness modulation
│   ├── features/
│   │   ├── windows.py         # windowing + statistical features
│   │   └── spectral.py        # PSD, band powers, spectral centroid, peak picking
│   ├── graph/
│   │   └── build.py           # sensor graph from truss connectivity
│   ├── models/
│   │   ├── gcn_ae.py          # graph autoencoder (dense adjacency)
│   │   └── baselines.py       # PCA reconstruction, IsolationForest
│   ├── data/
│   │   └── lumo.py            # LUMO ingestion + adapter to common schema
│   └── evaluate/
│       ├── metrics.py         # ROC-AUC, FAR@TPR, localisation top-k
│       ├── priority.py        # maintenance-priority ranking stub (Milestone 2 stand-in)
│       └── plots.py
├── scripts/
│   ├── 01_generate_synthetic.py
│   ├── 02_extract_features.py
│   ├── 03_train.py
│   ├── 04_evaluate.py
│   ├── 05_lumo_transfer.py
│   └── 06_priority_ranking.py
├── notebooks/
│   └── results.ipynb          # figure generation only, not the source of truth
└── figures/
```

**Rule:** every script must be runnable head-to-tail from the repo root with no manual steps. `python scripts/01_generate_synthetic.py` must just work.

---

## 4. Day 1 — Physics-based synthetic tower

### 4.1 Geometry (`src/fem/geometry.py`)

Mirror LUMO's real geometry so the synthetic and real graphs are structurally comparable:

- 9 m tall, three 3 m segments
- Triangular cross-section (3 legs), isosceles
- 7 bracing levels per segment
- Base nodes fully fixed

Generate:
- `nodes`: `(N, 3)` array of coordinates
- `elements`: `(M, 2)` array of node index pairs
- `element_type`: `leg` | `diagonal` | `horizontal` (needed so damage targets braces, matching LUMO's removable braces)

### 4.2 Truss FE + modal analysis (`src/fem/truss.py`)

Standard 3D space-truss (2-node bar, 3 DOF/node):

- Element stiffness `k_e = (E·A / L) · outer(c, c)` where `c` is the direction-cosine vector, expanded to the 6×6 element matrix
- Assemble global `K` by DOF mapping
- Lumped mass matrix `M` (half element mass to each node) — adequate here, and much simpler than consistent mass
- Apply base fixity by removing constrained DOFs
- Solve `scipy.linalg.eigh(K_ff, M_ff)` → eigenvalues `λ`, natural frequencies `f = sqrt(λ)/(2π)`, mode shapes

**Acceptance criteria:**
- First bending frequency lands in a physically plausible range for a 9 m steel lattice mast (expect single-digit to low tens of Hz). If it comes out at 0.01 Hz or 5000 Hz, the assembly is wrong — check units (use SI throughout: metres, Pascals, kg).
- `K` is symmetric positive-definite after constraint removal.
- Mode shapes are `M`-orthonormal.
- Rigid-body modes absent (all frequencies > 0).

### 4.3 Damage injection (`src/fem/damage.py`)

Damage = stiffness reduction on a **brace** element: `EA → (1−d)·EA`, with `d ∈ {0.3, 0.5, 0.7, 1.0}` (1.0 ≈ LUMO's fully removed brace).

Support single-brace and multi-brace scenarios. Record ground truth: which element, which severity, which nodes it connects.

**Acceptance criterion:** damage measurably shifts natural frequencies downward, and the shift is larger for more severe damage. Plot `Δf` vs `d` and confirm monotonicity — if it isn't monotonic, something is wrong.

---

## 5. Day 2 — Response simulation + features

### 5.1 Response simulation (`src/simulate/response.py`)

Modal superposition under ambient excitation:

- Excitation: band-limited Gaussian white noise applied at nodes (proxy for wind buffeting)
- Retain the first ~10–20 modes
- Add modal damping (ζ ≈ 0.5–2 % is realistic for bolted steel)
- Integrate each modal coordinate (Duhamel or a simple Newmark-β / state-space step)
- Recover nodal accelerations, add measurement noise (start at ~2 % RMS)
- Sampling rate: match LUMO once its README is read; use a sensible default (e.g. 256–1000 Hz) until then

Output shape: `(n_windows, n_sensors, window_length)` plus a label table.

### 5.2 Environmental confounder (`src/simulate/environment.py`)

**This is the experiment that makes the project credible.** Temperature variation is the dominant nuisance factor in real SHM — it shifts natural frequencies by amounts comparable to real damage, and naive detectors false-alarm on it.

Model: temperature drives Young's modulus, `E(T) = E₀·(1 − α·(T − T₀))`. Sample `T` over a realistic daily/seasonal range. Generate healthy data across the full temperature range and damaged data at temperatures overlapping it.

Deliverable from this: a plot showing frequency shift due to temperature is the *same order* as frequency shift due to damage. This single figure is worth more in the interview than an extra model.

### 5.3 Features (`src/features/`)

Per sensor, per window:
- Time domain: RMS, variance, kurtosis, crest factor, peak-to-peak
- Frequency domain: Welch PSD → band powers over a fixed band grid, spectral centroid, dominant peak frequency and amplitude

Standardise features using **healthy-data statistics only** (fit scaler on healthy train split, apply to everything). Leaking damaged-state statistics into the scaler is the single easiest way to accidentally fake good results — don't.

**Acceptance criteria:**
- Feature matrix has no NaN/inf
- Healthy and damaged distributions visibly separate on at least one spectral feature (sanity check — if nothing separates, the simulation is too noisy or the damage too small)

---

## 6. Day 3 — Models

### 6.1 Baselines first (`src/models/baselines.py`)

Do **not** write the GNN before the baselines exist. A GNN that can't beat PCA is a finding you need to know on Day 3, not Day 5.

- **PCA reconstruction novelty detection** — fit PCA on healthy features, score by reconstruction error. This is the classic SHM baseline.
- **Isolation Forest** — fit on healthy, score anomaly.

### 6.2 Graph construction (`src/graph/build.py`)

- Nodes = sensor locations
- Edges = structural connectivity between them, derived from the truss element list (two sensor nodes are connected if a member joins them, optionally 2-hop)
- Build the symmetric normalised adjacency `Â = D^(-1/2)(A + I)D^(-1/2)` once, as a dense tensor

### 6.3 Graph autoencoder (`src/models/gcn_ae.py`)

```
Encoder:  X (n_nodes, n_feat) → GCN → ReLU → GCN → Z (n_nodes, latent)
Decoder:  Z → GCN → ReLU → GCN → X̂ (n_nodes, n_feat)
Loss:     MSE(X, X̂), trained on HEALTHY WINDOWS ONLY
```

GCN layer, plain PyTorch:
```python
class GCNLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
    def forward(self, x, a_hat):      # x: (B, N, d_in), a_hat: (N, N)
        return self.lin(a_hat @ x)
```

Scoring:
- **Detection** — global anomaly score = mean node reconstruction error across the graph
- **Localisation** — per-node reconstruction error; the node(s) with highest error should sit adjacent to the damaged brace

The localisation capability is the reason to use a GNN rather than a flat autoencoder. Make sure it's demonstrated, because it's the answer to "why a graph model?" — which Dr Egodawela will ask.

**Acceptance criteria:**
- Training loss decreases and converges
- Healthy validation error distribution is clearly below damaged error distribution
- ROC-AUC computed against the baselines on identical splits

---

## 7. Day 4 — LUMO real-data transfer

### 7.1 Before downloading anything

Read `README.pdf` (1.8 MB) from the LUMO dataset page first. Extract: file format, sampling rate, channel/sensor layout, damage-state naming convention (DAM3/DAM4/DAM6, positions `111` / `010`), units.

Download **one** exemplary ZIP first (~600 MB), not all of them. Confirm you can parse it before committing more bandwidth and disk.

Source: https://data.uni-hannover.de/dataset/lumo — licence CC-BY 3.0, so cite Wernitz et al. (2022) in the README.

### 7.2 Adapter (`src/data/lumo.py`)

Write LUMO into the **same schema** the synthetic pipeline already emits: `(n_windows, n_sensors, window_length)` + labels. Everything downstream — features, graph, model, evaluation — should then run unchanged. If you find yourself special-casing LUMO in the feature or model code, the abstraction is wrong; fix the adapter instead.

Graph for LUMO: build from the documented sensor layout (18 uniaxial accelerometers over six measurement levels). The provided healthy FE model (`lumo_fem_healthy.inp`) can be parsed for node/element connectivity if the README's layout diagram is ambiguous — it's an Abaqus-format text file, readable with a small parser.

### 7.3 Experiments

1. Train the graph autoencoder on LUMO **healthy** windows only, test on healthy + DAM3/DAM4/DAM6 → detection ROC-AUC
2. Localisation: does peak node error correspond to the level where braces were removed?
3. **Temperature robustness** — LUMO includes a temperature sensor and spans a full year. Show detection performance with and without temperature conditioning/normalisation. Real degradation here is expected and is a *result*, not a failure.

**Fallback:** if LUMO parsing stalls beyond ~4 hours, stop, document exactly where it stalled in the README under "Known limitations", and ship the synthetic-only result. A working synthetic system with an honest note beats a broken half-transfer.

---

## 8. Day 5 — Priority stub, then package and narrate

### 8.0 Maintenance-priority ranking stub (~1–2 hours, do this first)

This is the piece that makes the project legibly about Milestone 2 ("AI-enhanced monitoring and maintenance... optimised, cost-effective maintenance schedule"), not just Milestone 1/3. Keep it small and honestly scoped — a stub, not a scheduling engine.

`src/evaluate/priority.py`:
- Input: for each tower/damage instance already evaluated — anomaly score, localisation (which node/brace), and a severity-trend value (rate of change of anomaly score across recent windows, as a crude proxy for how fast something is deteriorating)
- Combine into a single priority score, e.g. `priority = severity_trend × current_anomaly_score`
- Attach a naive cost line per instance: a placeholder inspection cost vs. an estimated risk-of-delay cost, just enough to state the tradeoff explicitly rather than leave it implicit
- Output: a ranked table — which tower/brace needs attention first, and why

`scripts/06_priority_ranking.py` runs this over all evaluated instances (synthetic + LUMO) and writes the ranked table to `figures/priority_ranking.csv` or similar.

**Explicitly not in scope:** real cost data, real scheduling/routing optimisation, multi-period planning. State plainly in the README that this is a stand-in demonstrating the *shape* of a Milestone 2 problem, not a solved instance of it.

### 8.1 README structure

Write it as a short paper, not a tutorial:

1. **Problem** — one paragraph, framed in tower-asset terms (why structural monitoring of lattice masts matters, why labelled failures are scarce)
2. **Relationship to Project SMaRTER** — one paragraph: Milestone 1 (fingerprint model) already exists and is out of scope here; this project explores the monitoring layer that would sit downstream of it, feeding Milestones 2 and 3
3. **Approach** — physics-based synthetic generation → GNN autoencoder trained on healthy only → transfer to real LUMO benchmark → priority-ranking stub
4. **Results** — detection ROC-AUC table (GNN vs PCA vs IsolationForest, synthetic and LUMO), localisation top-k accuracy, temperature confounder figure, priority-ranking table
5. **What worked / what didn't** — be specific and honest
6. **Operational framing** — see §8.3
7. **Limitations** — 5-day prototype, scaled research mast not a real telecom tower, simulated wind loading, no real corrosion/fatigue mechanisms, priority ranking is illustrative not calibrated
8. **Citation** — Wernitz et al. (2022) for LUMO

### 8.2 Figures (aim for 4–5, no more)

1. Tower geometry with sensor nodes and graph edges overlaid
2. Frequency shift: damage severity vs temperature variation, on the same axes
3. Healthy vs damaged anomaly-score distributions
4. ROC curves, all models, both datasets
5. Localisation heat map on the tower geometry (per-node reconstruction error, damaged brace marked) — a clean 2D plot is sufficient; don't spend extra time pushing this to 3D, that budget now goes to §8.0 instead

Figure 5 still doubles as the "digital twin visualisation" talking point even as a 2D plot — a structural diagram coloured by predicted condition connects directly to Dr Egodawela's tower fingerprinting work.

### 8.3 Operational framing section (do not skip)

Most candidates stop at ROC-AUC. Add a short section on what the model would mean operationally:

- False positive → an unnecessary truck roll and climb crew dispatched. Costly and, given tower climbing's safety record, not risk-free.
- False negative → undetected degradation on a structural member.
- Therefore the operating point matters more than peak AUC. Report **false alarm rate at a fixed detection rate** alongside AUC, and state what you'd need from an asset owner to choose that threshold properly.

This is the section that signals you think like an industry-partnered researcher rather than a benchmark-chaser. It maps directly to the JD's "asset longevity and operational efficiency" line.

### 8.4 Code quality

The JD includes supervising undergraduate interns. Treat the repo as something an intern could pick up: docstrings on public functions, a `configs/default.yaml` rather than magic numbers scattered through scripts, and a README section on how to reproduce every figure. If asked about supervision, "I write code assuming someone else will need to extend it" is a real answer backed by a real artefact.

---

## 9. Explicitly cut

- **Isaac Sim** — hardware-incompatible (§1). Read the tutorials for conversational fluency instead; do not install.
- **Gazebo inspection patrol** — a robot patrol demo would consume a full day and duplicates what SearchBot already evidences. The damage-localisation visualisation (§8.2, figure 5) delivers the digital-twin talking point for ~1 hour instead of ~8.
- **Operational modal analysis (SSI/covariance-driven)** — the rigorous way to extract natural frequencies from ambient data, and genuinely the right tool, but too much for 5 days. Use PSD peak picking, and name SSI as future work. Knowing *why* SSI would be better is itself a good interview answer.
- **Real-time streaming / FastAPI service** — out of scope. Describe verbally how you'd approach it; the architecture note in §0 gives you a concrete basis for that answer.
- **LLM/text-mining over maintenance records** — this is what the existing fingerprint model already does (§0). Duplicating it would blur, not strengthen, the "what does your project add" story. Leave it out entirely.
- **Real cost data / scheduling optimisation for the priority stub (§8.0)** — the stub demonstrates the shape of the problem, not a solved instance of it. Don't chase realism here past the naive placeholder.

---

## 10. Daily gate checks

Stop and reassess if a gate isn't met by end of day.

| Day | Gate |
|---|---|
| 1 | FE model produces plausible modal frequencies; damage monotonically lowers them |
| 2 | Synthetic accelerations generated; feature matrix clean; healthy/damaged separate on ≥1 feature |
| 3 | GNN trained on healthy only; ROC-AUC computed vs both baselines; localisation demonstrated |
| 4 | LUMO parsed and run through the unchanged pipeline — **or** documented fallback |
| 5 | Priority-ranking stub produces a ranked table; README complete (including relationship-to-fingerprint-model paragraph); figures generated; repo runs clean from a fresh clone |

---

## 11. Honesty guardrail

This is a five-day exploratory prototype built to engage seriously with the problem domain — say exactly that. Do not present it as research, do not imply the synthetic tower is a validated model of a telecom tower, and do not round up partial results. If the LUMO transfer underperforms, that is a legitimate and interesting finding about domain shift, and reporting it plainly will land better with a research panel than a polished overclaim.

**On describing the relationship to the existing fingerprint model, specifically:** don't claim TowerWatch extends or improves on the real Amplitel/TRU fingerprint model — it doesn't; it's an independent, much smaller prototype that happens to sit downstream of it conceptually. The accurate framing is "your fingerprint model gives each tower a baseline; this explores turning ongoing sensor data into a live signal against that baseline" — a boundary observation, not a claimed contribution to their actual system.
