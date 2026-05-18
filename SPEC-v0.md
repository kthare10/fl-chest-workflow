# SPEC: Federated Learning Chest Imaging Workflow for Pegasus WMS

## 1. Overview

This specification defines a state-of-the-art [Pegasus WMS](https://pegasus.isi.edu/) workflow for **Federated Learning (FL)** on medical chest imaging datasets. The workflow is built upon the design patterns and scaffolding rules from the [pegasus-ai](https://github.com/pegasus-isi/pegasus-ai) plugin marketplace.

| Attribute | Value |
|---|---|
| **Workflow Name** | `fl-chest-workflow` |
| **Primary Dataset** | TCIA Collections (Naturally Decentralized, 3D/2D imaging) |
| **Secondary Dataset** | NIH Chest X-Ray14 (Open Access, 2D imaging) |
| **FL Framework** | PyTorch + NVIDIA FLARE / Flower (simulation mode mapped to Pegasus jobs) |
| **Parallelism** | Per-client (institution/site) parallelism with fan-in aggregation |
| **Architecture** | Hub-and-spoke: parallel local training → central aggregation |
| **Container** | Singularity/VIA Docker with PyTorch, MONAI, SimpleITK |
| **Runtime** | HTCondor (`condorpool`) with GPU support |

### 1.1 Motivation

- **TCIA** is the Tier-1 choice: naturally decentralized across real institutions, open access, and compute-heavy (3D imaging tells a compelling FL story).
- **NIH Chest X-Ray14** serves as a generalizability check: extreme domain shift and 2D imaging tests the global model on a completely different modality.
- **Why Pegasus?** Pegasus manages the complex DAG of per-client training, synchronization barriers (aggregation), and multi-dataset evaluation automatically.

---

## 2. Datasets & Heterogeneity Model

### 2.1 TCIA (The Cancer Imaging Archive)

| Property | Details |
|---|---|
| **Access** | Open Access (mostly) |
| **Modality** | CT, MRI, PET/CT (3D volumes), plus derived 2D slices |
| **Clients** | Each TCIA *collection* (institution) becomes an FL client |
| **Heterogeneity** | Natural: different scanner brands (GE, Siemens, Philips), protocols, slice thickness, contrast agents, patient demographics |
| **Task** | Multi-label pathology detection / classification |

**Client Partitioning:**
- Group studies by `Collection` metadata (e.g., `LIDC-IDRI`, `NSCLC-Radiomics`, `LDCT-and-Projection-data`).
- Each collection = one FL client.
- Expected: 5–15 clients depending on selected collections.

### 2.2 NIH Chest X-Ray14

| Property | Details |
|---|---|
| **Access** | Open Access (Box download) |
| **Modality** | 2D frontal-view X-rays |
| **Clients** | The entire dataset treated as a single "external" client or split by view position (PA vs AP) |
| **Heterogeneity** | Mixed: different patient populations, acquisition hardware, labeling noise |
| **Task** | 14-class thoracic disease classification |

**Usage in Workflow:**
- Evaluate the globally aggregated model on NIH Chest X-Ray14 as a **zero-shot generalization test**.
- Fine-tuning on NIH data is an optional second stage.

### 2.3 Heterogeneity Metrics to Capture

The workflow will compute and report:
1. **Label distribution skew** ( quantify with KL divergence across clients )
2. **Image intensity / texture shift** ( using histogram matching metrics )
3. **Scanner metadata divergence** ( when DICOM metadata is available )

---

## 3. Workflow Architecture (DAG)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FL CHEST IMAGING WORKFLOW                            │
└─────────────────────────────────────────────────────────────────────────────┘

Stage 1: Data Preparation (Parallel per client)
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ ingest_tcia │   │ ingest_tcia │   │ ingest_tcia │  ... (one per TCIA client)
│ client_0    │   │ client_1    │   │ client_N    │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │
       ▼                 ▼                 ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ preprocess  │   │ preprocess  │   │ preprocess  │
│ client_0    │   │ client_1    │   │ client_N    │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │
       ▼                 ▼                 ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ compute_    │   │ compute_    │   │ compute_    │
│ stats_c0    │   │ stats_c1    │   │ stats_cN    │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │
       └────────┬────────┴────────┬────────┘
                │                 │
                ▼                 ▼
Stage 2: Setup + Initialization
┌───────────────────────────────────────────────┐
│ initialize_model (central)                    │
│   → Outputs: global_model_initial.pt          │
│              model_arch.json                  │
└─────────────┬─────────────────────────────────┘
              │
              ▼
Stage 3: Federated Learning Rounds (Iterative)
┌──────────────────────────────────────────────────────────────┐
│ FOR round = 1 to ROUNDS:                                     │
│                                                              │
│   3a. Broadcast (implicit via job input sharing in Pegasus) │
│       global_model_round_{r-1}.pt → each client             │
│                                                              │
│   3b. Local Training (Parallel Fan-Out)                      │
│       ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│       │ train_c0 │  │ train_c1 │  │ train_cN │              │
│       └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│            │             │             │                     │
│            ▼             ▼             ▼                     │
│       ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│       │local_    │  │local_    │  │local_    │              │
│       │model_c0 │  │model_c1 │  │model_cN │              │
│       │metrics_  │  │metrics_  │  │metrics_  │              │
│       │ c0.json │  │ c1.json │  │ cN.json │              │
│       └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│            │             │             │                     │
│   3c. Aggregation (Fan-In Merge) ────────────────────────────┤
│       ┌─────────────────────────────────────────┐            │
│       │ aggregate (FedAvg / FedProx / SCAFFOLD) │            │
│       │   Inputs: all local_model_c*.pt         │            │
│       │   Outputs: global_model_round_r.pt      │            │
│       │            round_metrics.json             │            │
│       └─────────────────────────────────────────┘            │
│                                                              │
└──────────────────────────────────────────────────────────────┘

Stage 4: Final Evaluation (Parallel)
┌──────────────────────────────────────────────┐
│ evaluate_global (on held-out TCIA central)   │
│   → Outputs: final_tcia_metrics.json         │
└──────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────┐
│ evaluate_nih (zero-shot on Chest X-Ray14)    │
│   → Outputs: final_nih_metrics.json          │
└──────────────────────────────────────────────┘
              │
              ▼
Stage 5: Reporting
┌──────────────────────────────────────────────┐
│ generate_report                              │
│   Inputs: all round_metrics, final metrics   │
│   Outputs: FL_report.html, FL_report.pdf     │
└──────────────────────────────────────────────┘
```

### 3.1 Key Design Patterns Used

| Pattern | Source Reference | Usage |
|---|---|---|
| **Per-client parallelism** | PEGASUS.md "Per-Sample Parallelism" | Each TCIA collection is an independent training job |
| **Fan-In / Aggregation** | PEGASUS.md "Fan-In Merge" | `aggregate` job collects all `local_model_*.pt` |
| **Hub-and-Spoke ML** | PEGASUS.md "ML Pipeline Integration" | Global model is shared artifact; predict/evaluate reuse it |
| **Conditional DAG** | PEGASUS.md "Conditional DAG Construction" | `--skip-nih-eval`, `--skip-preprocessing` flags |
| **Per-tool resource dict** | PEGASUS.md "Per-Tool Resource Configuration" | Training gets GPU+high memory; aggregation gets CPU+high memory |
| **Structured logging** | PEGASUS.md "Structured Logging in Wrapper Scripts" | All wrappers use `logging` module |
| **Test mode** | PEGASUS.md "Built-in Test Mode" | `--test` flag downloads a single small TCIA collection |

---

## 4. Pipeline Steps (Wrapper Scripts)

All wrapper scripts live in `bin/` and follow the [pegasus-wrapper](claude-plugin-marketplace/plugins/pegasus-ai/skills/pegasus-wrapper/SKILL.md) rules.

### 4.1 `bin/ingest_tcia.py`

**Purpose:** Download and validate DICOM data for one TCIA collection.

**Arguments:**
```
--collection-name  STR   (e.g., "LIDC-IDRI")
--output-dir       PATH  (e.g., "data/LIDC-IDRI")
--manifest-csv     FILE  (replica catalog: list of SeriesInstanceUIDs)
```

**Outputs:**
- `data/{collection}/manifest.json`
- `data/{collection}/raw/` (DICOM files or symlinks)

**Behavior:**
- Validates DICOM headers (Modality, SliceThickness, Manufacturer).
- Computes per-study metadata (scanner type, number of slices).
- Exits non-zero if >10% of expected series are missing.

**Resource Profile:**
| memory | cores | gpus |
|---|---|---|
| 4 GB | 2 | 0 |

---

### 4.2 `bin/preprocess.py`

**Purpose:** Normalize, resample, and window DICOM volumes into standard tensors.

**Arguments:**
```
--input-dir     PATH   (e.g., "data/LIDC-IDRI")
--output-dir    PATH   (e.g., "preprocessed/LIDC-IDRI")
--spacing       FLOAT  (target spacing in mm, default 1.0)
--window-center INT    (default: 40 for lung window)
--window-width  INT    (default: 400)
```

**Outputs:**
- `preprocessed/{collection}/{study_uid}.pt` (PyTorch tensor)
- `preprocessed/{collection}/labels.json` (pathology labels)

**Behavior:**
- Uses MONAI `LoadImage`, `Spacingd`, `ScaleIntensityRanged`.
- Creates output subdirectories with `os.makedirs`.
- Skips studies with missing slices or inconsistent dimensions.

**Resource Profile:**
| memory | cores | gpus |
|---|---|---|
| 8 GB | 4 | 0 |

---

### 4.3 `bin/compute_stats.py`

**Purpose:** Compute label distribution and intensity statistics per client for heterogeneity reporting.

**Arguments:**
```
--input-dir    PATH   (preprocessed client data)
--output-json  FILE   (e.g., "stats/LIDC-IDRI_stats.json")
```

**Outputs:**
- `stats/{collection}_stats.json`

**Behavior:**
- Computes normalized class histogram.
- Computes mean/std of voxel intensities.
- Saves scanner manufacturer distribution.

---

### 4.4 `bin/initialize_model.py`

**Purpose:** Create the initial global model checkpoint.

**Arguments:**
```
--arch-config    FILE   (JSON file with model architecture params)
--output-model   FILE   (e.g., "global_model_initial.pt")
--output-config  FILE   (e.g., "model_arch.json")
```

**Outputs:**
- `global_model_initial.pt`
- `model_arch.json`

**Behavior:**
- Instantiates a DenseNet-121 or ResNet-50 (configurable).
- Saves state dict and architecture metadata.

**Resource Profile:**
| memory | cores | gpus |
|---|---|---|
| 2 GB | 1 | 0 |

---

### 4.5 `bin/train_client.py`

**Purpose:** Perform local FL training for one client for one round.

**Arguments:**
```
--client-id        STR   (e.g., "LIDC-IDRI")
--data-dir         PATH  (preprocessed client data)
--global-model     FILE  (input model checkpoint)
--output-model     FILE  (local updated model)
--output-metrics   FILE  (training metrics JSON)
--round            INT
--epochs           INT   (local epochs, default 5)
--batch-size       INT   (default 16)
--lr               FLOAT (default 1e-4)
--fedprox-mu       FLOAT (FedProx mu, default 0.0)
```

**Outputs:**
- `local_models/round_{r}/{collection}_local.pt`
- `local_models/round_{r}/{collection}_metrics.json`

**Behavior:**
- Loads global model, fine-tunes on local data.
- Supports FedProx regularization (`--fedprox-mu`).
- Saves train/validation loss and AUC per epoch.
- Uses `torch.save` for checkpointing.

**Resource Profile:**
| memory | cores | gpus |
|---|---|---|
| 16 GB | 4 | 1 |

> **Note on GPU:** Register with `.add_pegasus_profile(gpus=1)` and container `arguments="--nv"`.

---

### 4.6 `bin/aggregate.py`

**Purpose:** Aggregate local model updates into a new global model.

**Arguments:**
```
--input-models     FILE [FILE ...]  (all local_model_*.pt)
--prev-global      FILE             (global model from previous round)
--output-model     FILE             (new global model)
--output-metrics   FILE             (aggregation metrics)
--strategy         STR              ("fedavg", "fedprox", "scaffold")
--client-weights   FILE             (JSON mapping client->num_samples)
```

**Outputs:**
- `global_models/round_{r}.pt`
- `metrics/round_{r}_aggregation.json`

**Behavior:**
- Implements weighted FedAvg by default (`weight = n_samples / total`).
- SCAFFOLD variant requires `.add_args` for control variates files.
- Validates that all model state dict keys match.

**Resource Profile:**
| memory | cores | gpus |
|---|---|---|
| 32 GB | 8 | 0 |

> **Note:** Aggregation is memory-bound (loading all local models into RAM). High memory is needed.

---

### 4.7 `bin/evaluate.py`

**Purpose:** Evaluate a global model on a test dataset.

**Arguments:**
```
--model          FILE  (global model checkpoint)
--data-dir       PATH  (test data directory)
--output-metrics FILE  (e.g., "final_tcia_metrics.json")
--dataset-name   STR   ("tcia" or "nih")
```

**Outputs:**
- `{dataset_name}_metrics.json` (AUC, F1, accuracy per class)

**Behavior:**
- Runs inference on held-out test split.
- Computes ROC-AUC, PR-AUC, F1 per pathology label.
- Saves confusion matrix as PNG if `matplotlib` available.

**Resource Profile:**
| memory | cores | gpus |
|---|---|---|
| 8 GB | 2 | 1 |

---

### 4.8 `bin/generate_report.py`

**Purpose:** Generate an HTML/PDF report of the full FL experiment.

**Arguments:**
```
--round-metrics   FILE [FILE ...]  (all round metrics)
--final-metrics   FILE [FILE ...]  (TCIA + NIH final metrics)
--client-stats    FILE [FILE ...]  (all client stats)
--output-html     FILE
--output-pdf      FILE
```

**Outputs:**
- `FL_report.html`
- `FL_report.pdf`

**Behavior:**
- Uses Jinja2 template + matplotlib/seaborn plots.
- Plots: round vs. global AUC, client drift, label skew heatmap.

---

## 5. Workflow Generator (`workflow_generator.py`)

The generator follows the [pegasus-scaffold](claude-plugin-marketplace/plugins/pegasus-ai/skills/pegasus-scaffold/SKILL.md) pattern and the `workflow_generator_template.py` structure.

### 5.1 CLI Arguments

```python
parser.add_argument("--clients", nargs="+", required=True,
                    help="TCIA collection names to use as FL clients")
parser.add_argument("--rounds", type=int, default=10,
                    help="Number of FL training rounds")
parser.add_argument("--local-epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--strategy", choices=["fedavg", "fedprox", "scaffold"],
                    default="fedavg")
parser.add_argument("--fedprox-mu", type=float, default=0.0)
parser.add_argument("--nih-data-dir", type=str, default=None,
                    help="Path to NIH Chest X-Ray14 preprocessed data for evaluation")
parser.add_argument("--skip-nih-eval", action="store_true")
parser.add_argument("--skip-preprocessing", action="store_true")
parser.add_argument("--test", action="store_true",
                    help="Use a single small TCIA collection for testing")
parser.add_argument("-e", "--execution-site-name", default="condorpool")
parser.add_argument("-o", "--output", default="workflow.yml")
```

### 5.2 Tool Configurations

```python
TOOL_CONFIGS = {
    "ingest_tcia":   {"memory": "4 GB",  "cores": 2, "gpus": 0},
    "preprocess":    {"memory": "8 GB",  "cores": 4, "gpus": 0},
    "compute_stats": {"memory": "2 GB",  "cores": 1, "gpus": 0},
    "initialize_model": {"memory": "2 GB", "cores": 1, "gpus": 0},
    "train_client":  {"memory": "16 GB", "cores": 4, "gpus": 1},
    "aggregate":     {"memory": "32 GB", "cores": 8, "gpus": 0},
    "evaluate":      {"memory": "8 GB",  "cores": 2, "gpus": 1},
    "generate_report":{"memory": "4 GB", "cores": 2, "gpus": 0},
}
```

### 5.3 DAG Construction Logic

```python
def create_workflow(self, args):
    self.wf = Workflow(self.wf_name, infer_dependencies=True)

    # Stage 1: Data prep (parallel per client)
    preprocessed_dirs = {}
    client_stats_files = []
    for client in args.clients:
        pp_dir, stats_file = self._add_client_preparation(client, args)
        preprocessed_dirs[client] = pp_dir
        client_stats_files.append(stats_file)

    # Stage 2: Initialize model
    global_model, model_arch = self._add_initialize_job()

    # Stage 3: FL Rounds
    current_global = global_model
    all_round_metrics = []
    for r in range(1, args.rounds + 1):
        local_models = []
        local_metrics = []
        for client in args.clients:
            lm, lmet = self._add_train_job(
                client, r, current_global,
                preprocessed_dirs[client], args
            )
            local_models.append(lm)
            local_metrics.append(lmet)

        new_global, agg_metrics = self._add_aggregate_job(
            r, current_global, local_models, local_metrics, args
        )
        all_round_metrics.append(agg_metrics)
        current_global = new_global

    # Stage 4: Evaluation (parallel TCIA + NIH)
    final_tcia = self._add_evaluate_job(
        "tcia", current_global, preprocessed_dirs, args
    )
    final_nih = None
    if not args.skip_nih_eval and args.nih_data_dir:
        final_nih = self._add_evaluate_job(
            "nih", current_global, args.nih_data_dir, args
        )

    # Stage 5: Report
    self._add_report_job(
        all_round_metrics, [final_tcia, final_nih],
        client_stats_files, args
    )
```

### 5.4 File Objects & Dependency Rules

- `global_model_round_{r}.pt` is shared between `train_client` jobs as input and produced by `aggregate`.
- `local_model_{client}_round_{r}.pt` is produced by `train_client` and consumed by `aggregate`.
- **No directory scanning**: each model path is passed explicitly via `--input-models` using `nargs="+"` in the wrapper.
- **stage_out=True** only for: final models, metrics JSONs, report HTML/PDF.
- **stage_out=False** for: intermediate local models, preprocessed tensors, aggregation outputs.

---

## 6. Container Definition

### 6.1 `Docker/fl_chest_Dockerfile`

```dockerfile
FROM mambaorg/micromamba:1.5-jammy

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    xvfb \
    libgl1-mesa-glx \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER
RUN micromamba install -y -n base -c conda-forge -c pytorch -c nvidia \
    python=3.10 \
    pytorch=2.2 \
    torchvision \
    pytorch-cuda=12.1 \
    monai=1.3 \
    simpleitk=2.3 \
    pydicom=2.4 \
    numpy \
    pandas \
    scikit-learn \
    scikit-image \
    matplotlib \
    seaborn \
    jinja2 \
    requests \
    && micromamba clean --all --yes

RUN pip install --no-cache-dir \
    dcmtk \
    "nvflare>=2.4" \
    flower>=1.6

ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
```

### 6.2 Transformation Catalog Registration

```python
fl_container = Container(
    "fl_chest_container",
    container_type=Container.SINGULARITY,
    image="docker://USERNAME/fl-chest:latest",
    image_site="docker_hub",
    arguments="--nv",  # Enable NVIDIA GPU support
)

train_tx = Transformation(
    "train_client",
    site=exec_site_name,
    pfn=os.path.join(self.wf_dir, "bin/train_client.py"),
    is_stageable=True,
    container=fl_container,
).add_pegasus_profile(
    memory="16 GB", cores=4, gpus=1
)
```

---

## 7. Replica Catalog

Input files registered in the Replica Catalog:

| Logical Name | Physical Path | Purpose |
|---|---|---|
| `tcia_manifest.csv` | `data/tcia_manifest.csv` | Master manifest of all selected SeriesInstanceUIDs |
| `nih_manifest.csv` | `data/nih_manifest.csv` | NIH Chest X-Ray14 image list and labels |
| `model_arch.json` | `configs/densenet121.json` | Model architecture configuration |
| `bin/*.py` | *(not in RC — these are Transformations)* | Wrapper scripts |

> **Note:** DICOM data is fetched at runtime by `ingest_tcia.py` (API pattern) or pre-staged on shared storage and accessed via `transfer_input_files` (CondorIO pattern).

---

## 8. File Staging Strategy

| File Type | stage_out | Reason |
|---|---|---|
| Raw DICOM / downloaded data | `False` | Only needed for preprocessing |
| Preprocessed `.pt` tensors | `False` | Intermediate, consumed by training |
| Global model checkpoints | `False` | Intermediate between rounds |
| Local model checkpoints | `False` | Intermediate, consumed by aggregation |
| Round metrics JSON | `False` | Intermediate, consumed by report |
| Final global model (last round) | `True` | Deliverable |
| Final evaluation metrics | `True` | Deliverable |
| Client statistics JSON | `True` | Deliverable |
| Report HTML/PDF | `True` | Final deliverable |

---

## 9. Execution Model & Resource Planning

### 9.1 Estimated Runtime (per round, 10 clients)

| Step | Duration | Parallelism |
|---|---|---|
| Ingest | 10–30 min | 10 clients in parallel |
| Preprocess | 15–45 min | 10 clients in parallel |
| Local Training | 30–120 min | 10 clients in parallel |
| Aggregation | 1–5 min | Single job (barrier) |
| Evaluation | 5–15 min | TCIA + NIH in parallel |

**Total (10 rounds, 10 clients):**
- ~60–250 compute hours, but wall clock time ≈ 6–12 hours thanks to Pegasus parallelism.

### 9.2 Site Catalog

```python
exec_site = (
    Site("condorpool")
    .add_condor_profile(universe="vanilla")
    .add_pegasus_profile(style="condor")
    .add_profiles(Namespace.CONDOR, key="request_gpus", value="1")
)
```

---

## 10. Expected Outputs

```
output/
├── global_models/
│   └── round_10.pt                    (final aggregated model)
├── metrics/
│   ├── round_1_aggregation.json
│   ├── ...
│   ├── round_10_aggregation.json
│   ├── final_tcia_metrics.json
│   └── final_nih_metrics.json
├── client_stats/
│   ├── LIDC-IDRI_stats.json
│   └── ...
├── local_models/                      (optional, if retained)
│   └── round_10/
└── reports/
    ├── FL_report.html
    └── FL_report.pdf
```

---

## 11. Testing & Validation

### 11.1 `--test` Mode

When `--test` is passed:
1. Use a single small TCIA collection (e.g., `RIDER Lung CT`, ~10 studies).
2. Reduce `--rounds` to 2 and `--local-epochs` to 1.
3. Skip NIH evaluation.
4. Total runtime: < 30 minutes wall clock.

### 11.2 `run_manual.sh`

A standalone script for local validation:
```bash
#!/bin/bash
set -e
./bin/ingest_tcia.py --collection-name RIDER --output-dir test_data/RIDER
./bin/preprocess.py --input-dir test_data/RIDER --output-dir test_preprocessed/RIDER
./bin/initialize_model.py --output-model test_model.pt
./bin/train_client.py --client-id RIDER --data-dir test_preprocessed/RIDER \
  --global-model test_model.pt --output-model test_local.pt --round 1
```

---

## 12. Implementation Roadmap

| Phase | Task | Files |
|---|---|---|
| **1** | Scaffold workflow structure | `workflow_generator.py`, `bin/`, `Docker/` |
| **2** | Implement data ingestion wrappers | `bin/ingest_tcia.py`, `bin/preprocess.py` |
| **3** | Implement FL core | `bin/initialize_model.py`, `bin/train_client.py`, `bin/aggregate.py` |
| **4** | Implement evaluation & reporting | `bin/evaluate.py`, `bin/generate_report.py` |
| **5** | Build container & test locally | `Docker/fl_chest_Dockerfile`, `run_manual.sh` |
| **6** | Validate on small TCIA subset | `--test` mode |
| **7** | Full-scale run with multiple clients | condorpool submission |
| **8** | Add NIH Chest X-Ray14 evaluation | `bin/ingest_nih.py`, cross-dataset eval |

---

## 13. References

- [Pegasus WMS Python API](https://pegasus.isi.edu/documentation/python/Pegasus.api.html)
- [Pegasus Service Reference](https://pegasus.isi.edu/documentation/reference-guide/pegasus-service.html)
- [FL_Dataset_Evaluation.md](FL_Dataset_Evaluation.md) — Dataset selection rationale
- [PEGASUS.md](claude-plugin-marketplace/plugins/pegasus-ai/references/PEGASUS.md) — Pegasus development patterns
- [pegasus-scaffold SKILL.md](claude-plugin-marketplace/plugins/pegasus-ai/skills/pegasus-scaffold/SKILL.md)
- [pegasus-wrapper SKILL.md](claude-plugin-marketplace/plugins/pegasus-ai/skills/pegasus-wrapper/SKILL.md)
- [pegasus-dockerfile SKILL.md](claude-plugin-marketplace/plugins/pegasus-ai/skills/pegasus-dockerfile/SKILL.md)
