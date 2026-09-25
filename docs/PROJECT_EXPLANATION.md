# Project Explanation — Learning MILP-Optimal Sensing/Communication Scheduling for an ISAC UAV

Everything below comes from the actual code and data in this folder. Where the code does
something without stating why, the **"What the code does"** part is fact and the
**"Likely purpose"** part is interpretation, and they are labelled that way.

---

## 0. Who did what

**Provided by faculty advisor:** the MILP optimisation formulation and base MATLAB code. That
covers the system model, PsiComm (rate) and PsiSense (CRB) metrics, the objective, the `R_min`
constraint and the communicate-first rule.

**My work:**
1. **Refined the MILP code into a dataset generator.** The commented-out lines in
   `generate_dataset_milp.m` show the original setup: a single fixed `R_min` / `p_fraction`,
   fixed CU/ST coordinates, and a straight-line trajectory. I changed it to draw a random `R_min`
   and `p_fraction` per environment, random CU/ST positions and a random trajectory, and to
   export a per-waypoint feature table with the MILP label.
2. **Generated large datasets:** 6,000 trajectories (90,000 rows) for training and 60,690
   trajectories (910,350 rows) for testing, with CU/ST coordinates kept for evaluation.
3. **Designed and trained the models:** the 3-branch Hybrid Conv3D-LSTM and the lightweight CNN
   (Sections 5–6). This included preprocessing, feature grouping, scaling, regularisation, the
   class-prior bias and the temperature softmax.
4. **Inference:** prediction scripts for all unseen test trajectories.
5. **Evaluation:** training curves, Numba-accelerated CRB recomputation for the sensing CDFs,
   communication CDFs and sweeps, compiled into the results PDF.

**How to say it in an interview:** "My faculty gave me the MILP formulation. I turned it into
a large-scale randomised dataset generator, built and trained the deep models that imitate
the optimiser, and did the full evaluation."

---

## 1. The research problem in one paragraph

A UAV (drone) flies over a 500 m x 300 m area at 70 m altitude, from (0,0) to (500,300),
stopping at **15 waypoints** and hovering 10 s at each. On the ground there are
**10 communication users (CUs)** that want data and **10 sensing targets (STs)** whose position
the UAV wants to estimate with radar. At each waypoint the UAV picks one mode:

| Label | Mode | Contributes to |
|---|---|---|
| 1 | Sensing only | sensing |
| 2 | Communication only | communication |
| 3 | Joint (ISAC: sense + communicate) | both |

The goal is to **maximise sensing quality while guaranteeing the users receive at least
`R_min` bits in total**. This is solved exactly with a **Mixed-Integer Linear Program (MILP)**.
MILP is slow (a solver per environment, capped at 60 s), so the project **trains deep
networks to imitate the MILP decisions**. Once trained, the network gives the mode sequence
almost instantly. This is a *learning-to-optimise* / supervised-imitation approach.

**Objective of the study:** show that a learned model (Hybrid CNN-LSTM, and a plain CNN for
comparison) can reproduce MILP-quality decisions, measured by classification accuracy and by
the actual sensing and communication performance its decisions achieve.

---

## 2. Dataset generation — `01_dataset_generation/generate_dataset_milp.m`

### 2.1 One "environment"
For each environment the script randomly draws:
- `R_min` ~ Uniform(1e7, 3e8) bits: total communication requirement
- `p_fraction` ~ Uniform(0.1, 0.9): fraction of `R_min` that must be delivered *before* sensing is allowed
- 10 CU positions and 10 ST positions, uniform over the 500 x 300 area
- a random 15-point UAV trajectory from (0,0) to (500,300) with a nominal step of 60 m
  (made by `UAV_random_trajectory`, which is **not in this folder**)

Fixed system constants: Pt = 0.1 W, antenna gains Gt=10, Gc=5, Gs=5, processing gain Gp=10,
noise σ² = 1e-9 W, bandwidth 10 MHz split equally over 10 CUs, λ = 0.1 m, RCS = 1.

### 2.2 Communication metric `PsiComm` (per waypoint j)
For each CU: distance d = sqrt(H² + horizontal²), channel gain h = α₀/d²,
SNR = Pt·h/σ², rate = (B/M)·log₂(1+SNR), bits = T_h · rate.
`PsiComm_j` = **mean bits delivered to a CU while hovering at waypoint j**. Higher is better.

### 2.3 Sensing metric `PsiSense` (per waypoint j)
For each target the UAV measures range. Measurement variance grows as **d⁴**
(radar round trip). The script builds a 2x2 **Fisher Information Matrix (FIM)** for the
target's (x,y) from all waypoints 1..j:
J = Σ (1/σ²) · q qᵀ, where q is the unit direction from UAV to target.
The **Cramér-Rao Bound (CRB)** = trace(J⁻¹) = (Jxx+Jyy)/(Jxx·Jyy − Jxy²) is the lowest
possible mean-squared error of the position estimate.
`PsiSense_j` = mean CRB over the 10 targets. **Lower is better** (it is an error bound).
- At waypoint 1 there is only one measurement direction, so the FIM is singular and the
  CRB is infinite. The code stores it as **1e18**. That is why every trajectory's first row has `PsiSense = 1e18`.
- Sensing utility used by the MILP: `fs_j = 1 / PsiSense_j`.

### 2.4 The MILP
Decision variables: binary x_j1, x_j2, x_j3 for each of 15 waypoints (45 binaries).

- **Objective (minimise):** Σ_j [ −fs_j·x_j1 + η·x_j2 + (−fs_j + η)·x_j3 ], with η = 1e-4.
  This maximises sensing utility and adds a tiny penalty for using communication.
- **One mode per waypoint:** x_j1 + x_j2 + x_j3 = 1
- **Communication requirement:** Σ_j PsiComm_j·(x_j2 + x_j3) ≥ R_min
- **Early-communication rule:** walk through the waypoints and accumulate PsiComm. While the
  running total is below `p_fraction·R_min`, force x_j1 = x_j3 = 0, i.e. **mode 2 is forced**.
- Solved with MATLAB `intlinprog` (gap 1e-4, 60 s limit).

**What the objective implies (by inspection):**
- Mode 3 gives the same sensing reward as mode 1 but costs η, so the solver uses mode 3 only
  when it still needs communication bits. Otherwise it picks mode 1.
- Mode 2 gives no sensing reward and costs η, so it appears essentially only when the
  early rule forces it.

The data confirms this (training set, count by waypoint):

| Waypoint | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11–15 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Label 2 (comm) | 6000 | 4929 | 3646 | 2516 | 1615 | 933 | 472 | 168 | 35 | 6 | 0 |

So a typical MILP schedule is: **communicate first → a few joint stages → sensing only at the end.**

### 2.5 Output rows
Each waypoint becomes one row: `X, Y, MeanDistCU, MinDistCU, MaxDistCU, MeanDistST,
MinDistST, MaxDistST, PsiComm, PsiSense, Rmin, FractionP, Label`
(12 features + label). 15 consecutive rows = 1 trajectory.

**Honest note:** the script as written loops 200,000 environments and writes only these 13
columns. The datasets present differ from that: the training file has 6,000 trajectories with
rounded values, and the test file has 60,690 trajectories plus 40 CU/ST coordinate columns.
So they were produced by a variant of this script, or by runs of it with different settings.

---

## 3. The data files — `data/`

| File | Rows | Trajectories | Columns | Used for |
|---|---|---|---|---|
| `Copy of UAV90BIG1.xlsx` | 90,000 | 6,000 | 12 features + Label | training both models |
| `master_table_FULLdec.xlsx` | 910,350 | 60,690 | 12 features + CU1..CU10, ST1..ST10 (x,y) + Label | testing + graphs |
| `full_dec_cnn.xlsx` | 910,350 | 60,690 | 12 features + Label + Predicted_Label | CNN predictions |

Class balance (train and test are almost identical): **Label 1 = 62.5%, Label 2 = 22.6%, Label 3 = 15.0%**.
The CU/ST coordinates in the test file are **dropped before prediction**. They exist so the
graph scripts can *recompute* sensing performance for any label sequence.

---

## 4. What exactly is classified

**Sequence labelling (many-to-many):** input = one whole trajectory (15 waypoints x 12
features), output = one of 3 classes **for every waypoint**. Output shape (15, 3) softmax.
The model sees the whole trajectory at once, so it can use context from other waypoints.

---

## 5. Preprocessing (identical in all train/predict scripts)

| Step | What the code does | Likely purpose |
|---|---|---|
| Clean | `inf → 1e12`, `NaN → 0` | avoid NaN/inf crashing training |
| Reshape | rows → `(N, 15, 12)`, pads with zero rows if not a multiple of 15 (not needed here) | one sample = one trajectory |
| Labels | `1,2,3 → 0,1,2` and clip | Keras sparse CE needs 0-based classes |
| Split features into 3 groups | cols 0–6 (X, Y, Mean/Min/MaxDistCU, MeanDistST, MinDistST), cols 7–9 (MaxDistST, PsiComm, PsiSense), cols 10–11 (Rmin, FractionP) | give each kind of feature its own branch (see note) |
| Standardise | a separate `StandardScaler` per group | features are on very different scales (metres vs 1e8 bits) |
| Reshape for conv | Hybrid: `(N,15,1,F,1)` for Conv3D; CNN: `(N,15,F,1)` for Conv2D | shape needed by the conv layers |

**Observations a professor could raise:**
- With `C1 = 7`, `MaxDistST` lands in the "performance" group rather than the geometry group.
  A split of geometry(8) / performance(2) / constraints(2) was probably intended.
- `PsiSense` contains 1e18 in every first row. After standardisation this outlier dominates:
  the other `PsiSense` values all map to about the same number (≈ −0.27). The model effectively
  sees only "is this waypoint 1?" from that feature. A log transform would keep the information.
- The predict scripts **re-fit new scalers on the test data** instead of reusing the
  training scalers. Train and test come from the same distribution, so the effect is small,
  but the standard practice is to reuse the training scalers.

---

## 6. The two models

### Model A — Hybrid CNN-LSTM (`03_models/hybrid_cnn_lstm/train_hybrid_cnn_lstm.py`)
Per branch (×3, one per feature group):
`Conv3D(32 or 16, kernel 3x1x3) → BatchNorm → Dropout 0.5 → Conv3D(16 or 8, kernel 1x1x3) → BatchNorm → Dropout
→ TimeDistributed(Flatten → Dense 128) → Dropout → LSTM(128, return_sequences) → TimeDistributed(Dense 64)`

Merge: concat (192) → TimeDistributed Dense 128 → BatchNorm → Dropout → TimeDistributed Dense 3 → softmax.
Training: Adam lr 3e-4, batch 16, 500 epochs, sparse categorical cross-entropy, `shuffle=False`.

- **Conv3D kernel (3,1,3):** looks at 3 neighbouring waypoints × 3 neighbouring features, so it learns *local* patterns.
- **LSTM:** carries memory along the trajectory. The right mode depends on things like
  "how many bits have already been delivered", which is a *cumulative* quantity.

### Model B — Lightweight CNN (`03_models/cnn/train_cnn.py`)
Per branch: `GaussianNoise(0.15) → Conv2D(6 or 4 filters, kernel 3x1) → Dropout 0.6 → TimeDistributed(Flatten → Dense 16)`
Merge: concat (48) → TD Dense 32 → Dropout 0.6 → TD Dense 3 → **temperature softmax (T=1.3)**.
Extras: output bias initialised to `1.5·log(class priors)` (starts predictions near the class
frequencies, which helps with the imbalanced classes), gradient clipping `clipnorm=0.3`,
Adam lr 3e-5, batch 32, 500 epochs.

- No recurrent layer. The kernel (3,1) only sees **±1 neighbouring waypoint** for each feature separately.
- Heavy regularisation (noise, 0.6 dropout, tiny layers, low lr) makes it a small, conservative model.

### Why two models (likely purpose)
The CNN is a **baseline without sequence memory**. The Hybrid adds the LSTM. Comparing them
shows that **temporal/sequential context matters** for this scheduling problem. The CNN's
errors support that: it mostly confuses *joint (3)* with *sensing (1)*, and telling those
apart requires knowing whether the communication budget is already satisfied.

---

## 7. Prediction — `predict_*.py`
Load test file → drop the 40 CU/ST coordinate columns → remove `Label` → clean → reshape →
split → standardise (re-fit) → `model.predict` → `argmax` → +1 → save the file with a new
`Predicted_Label` column.
- CNN → `full_dec_cnn.xlsx` (present)
- Hybrid → `master_table_FULLdec_with_predictions.xlsx` (**not present in folder**)

---

## 8. Evaluation and graphs

| Graph (in results PDF) | Script | What it shows |
|---|---|---|
| Loss vs epoch, Accuracy vs epoch (Hybrid, CNN) | `training_curves/plot_loss_vs_epoch.py`, `plot_accuracy_vs_epoch.py` | Training curves from `training_log.csv` written by Keras `CSVLogger`. **Training set only, no validation curve.** Hybrid reaches ~96.7% train accuracy, CNN ~81%. |
| CDF of aggregate sensing performance | `cdf_plots/cdf_sensing_milp_vs_hybrid.py` (and `..._cnn.py`) | For each test trajectory, recompute the CRB **using only waypoints the labels mark as sensing (1 or 3)**, sum over the 15 waypoints, then plot the CDF of log10 over 60,690 trajectories, MILP labels vs model labels. |
| CDF of aggregate communication performance | `cdf_plots/cdf_communication_milp_vs_hybrid.py` | Per trajectory, sum `PsiComm` over waypoints labelled 2 or 3, then plot the CDF of log10. |
| Sensing vs number of STs / Communication vs number of CUs | `sweep_plots/*.py` | For K = 2, 4, 6, 8, 10: sum `PsiSense` (or `PsiComm`) of the **first K sensing (or comm) waypoints**, averaged over trajectories. |

**How to read them:**
- **CDF curves close together** means the model's decisions give almost the same performance
  distribution as the optimal MILP. That is the main claim.
- **Why the sensing CDF is a staircase at x ≈ 18.3–19.1:** a waypoint whose FIM is still
  singular (fewer than two sensing looks so far) adds 1e18. The aggregate is therefore about
  (number of "blind" waypoints) × 1e18: log10(2e18) = 18.3, log10(3e18) = 18.48, and so on. Each step is
  one more blind waypoint. In practice this CDF measures **how early sensing begins**.
- **Communication CDF:** the Hybrid curve is slightly to the left, so it delivers slightly fewer bits in the lower tail.

**Caveats you should know (from the code):**
1. The sweep scripts **do not change the number of STs/CUs in the simulation**. The x-axis
   is "number of sensing/communication waypoints counted" (the code's own comment says
   "effective sensing activity"). The axis labels "Number of STs/CUs" are an interpretation.
2. The sensing sweep y-axis has an offset of 1.267e11 while the curves differ by only about 1e6.
   The visible gap is < 0.01% of the value.
3. `sweep_sensing_and_communication_cnn.py` reads the **CNN** predictions but labels the curve "Hybrid".
4. In the results PDF, page 3 captions (a)/(b) are swapped relative to the figures, and the
   legends ("Predicted (HYB)") differ from the current scripts ("Predicted (CNN)").
   The figures were produced by slightly different versions of these scripts.

---

## 9. Numbers computed from the data files (read-only check, not in the PDF)

On the 60,690 test trajectories, from `full_dec_cnn.xlsx`:

| Metric | CNN |
|---|---|
| Per-waypoint accuracy vs MILP | **86.5%** |
| Recall: Sensing (1) / Comm (2) / Joint (3) | 93.8% / 95.9% / **41.3%** (55.5% of joint predicted as sensing) |
| Entire 15-waypoint sequence exactly matches MILP | 7.7% |
| Trajectories meeting `R_min` | MILP 100%, **CNN 40.2%** |
| Mean bits delivered per trajectory | CNN = 85% of MILP |

The same check on the Hybrid's predictions (`master_table_FULLdec_with_predictions.xlsx`, kept locally) gives
**92.9%** per-waypoint accuracy, recall 95.8% / 96.8% / **74.8%**, 32.0% exact sequence match, 62.5% of trajectories
meeting `R_min` and 96.5% of the MILP's mean bits. Script: `02_data_checks/evaluate_predictions.py`.
**Key insight:** per-waypoint accuracy alone overstates quality. The learned schedule can
violate the MILP's hard communication constraint. A good follow-up is to report
constraint-satisfaction rate and to add a feasibility repair step (for example, switch the last
sensing waypoints to joint mode until `R_min` is met).

---

## 10. End-to-end workflow

```
[MATLAB] random env (R_min, p, CUs, STs, trajectory)
   → compute PsiComm, PsiSense per waypoint
   → solve MILP → optimal label per waypoint
   → rows: 12 features + Label                    (data/*.xlsx)
[Python] clean → reshape (N,15,12) → 3 feature groups → StandardScaler
   → train Hybrid CNN-LSTM  /  train CNN          (500 epochs, training_log.csv, .h5)
   → predict labels on 60,690 unseen trajectories (Predicted_Label)
   → graphs: training curves; CDFs of recomputed sensing/comm (MILP vs model); sweeps
   → results PDF
```

---

## 11. Explaining it to a professor or interviewer (about 60 seconds)

> "I worked on scheduling for an integrated sensing and communication (ISAC) UAV. At each of
> 15 waypoints the drone must either sense ground targets, serve communication users, or do
> both. My faculty advisor provided a MILP formulation: it maximises sensing accuracy, measured
> by the inverse Cramér-Rao bound on target localisation, subject to a minimum total data-rate
> constraint and a rule that communication is prioritised early in the flight. I refined that
> code into a randomised dataset generator, with random requirements, user and target positions,
> and trajectories. I generated labelled data by solving the MILP over thousands of random
> environments, with 6,000 trajectories for training and about 60,000 for testing. Because solving a MILP per environment is too slow for
> real time, I trained deep networks to imitate it as a per-waypoint 3-class sequence-labelling
> problem. I compared a lightweight CNN against a hybrid Conv3D-LSTM. The hybrid reached about
> 97% training accuracy versus 81% for the CNN, and its CDFs of sensing and communication
> performance closely track the MILP optimum. The CNN, which has no sequence memory, mostly
> confused joint and sensing-only modes. That showed temporal context is essential, because
> the right choice depends on how much data has already been delivered."

---

## 12. Likely questions and answers

1. **Why MILP?** Modes are discrete (binary choice per waypoint) and the objective and
   constraints are linear in those binaries, so MILP gives the global optimum, which serves as ground-truth labels.
2. **Why use ML if MILP is optimal?** MILP solve time grows with problem size (a 60 s cap was
   set). A trained network needs one forward pass, which suits real-time or onboard decisions.
3. **What is the CRB?** The lower bound on the variance of any unbiased estimator. Here it is
   trace(FIM⁻¹) for the target's 2D position. Lower means better sensing.
4. **Why is the first PsiSense 1e18?** One range measurement from one direction cannot fix a
   2D position. The FIM is rank-1, so its inverse is infinite.
5. **Why d⁴ in sensing but d² in communication?** Radar signals travel out and back (two-way
   path loss). Communication is one-way.
6. **Why an LSTM?** Mode choice depends on cumulative state (bits delivered so far, sensing
   directions already collected). An LSTM carries that state along the sequence. A 3-wide conv cannot.
7. **Why 3 branches?** The feature groups have different meaning and scale. Separate branches
   and scalers let each learn its own filters. (Be ready to admit the split puts MaxDistST in the second group.)
8. **How did you handle class imbalance?** In the CNN, the output bias is initialised to the
   log class priors, softened by a temperature of 1.3. The Hybrid used no explicit weighting.
9. **Is there a validation set / overfitting check?** The curves are training-only. Testing was done
   on a separate 60,690-trajectory set. Be honest that a validation split and early stopping would improve rigour.
10. **Does the predicted schedule satisfy the constraint?** Not always. For the CNN, only about 40% of
    test trajectories meet R_min (checked from the data). Constraint-aware training or a repair step is the natural next step.
11. **Why CDFs instead of just accuracy?** Accuracy treats every error equally. A CDF of the
    resulting sensing/communication performance shows the real system-level cost of errors.
12. **Why StandardScaler?** Features span metres to 1e8 bits. Without scaling, gradients would be dominated by the large features.
13. **What is `shuffle=False`?** Batches are drawn in file order. Shuffling is normally preferred for SGD.
14. **Limitations?** Training-only curves, scalers re-fit on test data, the 1e18 outlier
    swamping PsiSense after scaling, the sweep x-axis not truly varying the number of STs/CUs, and no inference-time measurement.
