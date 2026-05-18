# Federated Learning Chest Imaging Workflow

A [Pegasus WMS](https://pegasus.isi.edu/) workflow for **Federated Learning (FL)**
on medical chest imaging datasets. The workflow trains a global deep-learning model
decentrally across multiple TCIA collections and evaluates it on both TCIA and
NIH Chest X-Ray14 data.

## Pipeline Overview

```
Stage 0: Pre-stage Data (on submit host, once)
  download_tcia  ──>  LIDC-IDRI.tar.gz, RIDER.tar.gz, ...

Stage 1: Data Preparation (parallel per client)
  ingest_tcia  ──>  preprocess  ──>  compute_stats

Stage 2: Model Initialization
  initialize_model  ──>  global_model_initial.pt

Stage 3: Federated Learning Rounds (iterative, compacted via SubWorkflows)
  For round = 1..R (each round = one SubWorkflow job):
    broadcast global model  ──>  train_client (parallel per client)
    local_models  ──>  aggregate  ──>  new_global_model

Stage 4: Final Evaluation (parallel)
  evaluate_tcia  ──>  final_tcia_metrics.json
  evaluate_nih   ──>  final_nih_metrics.json
  ensemble_eval  ──>  ensemble_metrics.json

Stage 5: Reporting
  generate_report  ──>  FL_report.html / FL_report.pdf
```

> **Note:** FL training rounds are automatically compacted into Pegasus SubWorkflows (one per round) to keep the parent DAG manageable. Inner job outputs stay in the subworkflow scratch; the SubWorkflow boundary stages them to the parent output directory. All jobs execute inside a Singularity container with GPU (`--nv`) support.

| Step | Tool | Description |
|------|------|-------------|
| 0. download | `bin/download_tcia.py` | **Pre-stage**: download & package DICOM data as tar.gz archives |
| 1. ingest | `bin/ingest_tcia.py` | Extract tar.gz, validate DICOMs, emit manifest.json |
| 2. preprocess | `bin/preprocess.py` | Normalize, resample, and window into `.pt` tensors |
| 2b. preprocess NIH | `bin/preprocess_nih.py` | Preprocess NIH Chest X-Ray14 PNG images |
| 3. compute_stats | `bin/compute_stats.py` | Compute per-client heterogeneity statistics |
| 4. initialize_model | `bin/initialize_model.py` | Create initial global model checkpoint |
| 5. train_client | `bin/train_client.py` | Local training for one client / one round |
| 6. aggregate | `bin/aggregate.py` | FedAvg / FedProx / SCAFFOLD aggregation |
| 7. evaluate | `bin/evaluate.py` | Evaluate global model on held-out data |
| 7b. ensemble | `bin/ensemble_evaluate.py` | Ensemble evaluation across rounds |
| 8. generate_report | `bin/generate_report.py` | Generate HTML/PDF experiment report |

## Directory Structure

```
fl-chest-workflow/
├── workflow_generator.py              # Pegasus workflow generator
├── bin/
│   ├── download_tcia.py               # Pre-stage TCIA data (run on submit host)
│   ├── ingest_tcia.py                 # Extract tar.gz, validate DICOMs
│   ├── preprocess.py                  # TCIA DICOM preprocessing
│   ├── preprocess_nih.py              # NIH Chest X-Ray14 preprocessing
│   ├── compute_stats.py               # Heterogeneity statistics
│   ├── initialize_model.py            # Model initialization
│   ├── train_client.py                # Local FL training
│   ├── aggregate.py                   # Model aggregation
│   ├── evaluate.py                    # Model evaluation
│   ├── ensemble_evaluate.py           # Ensemble evaluation
│   └── generate_report.py             # HTML/PDF report generation
├── Docker/
│   └── fl_chest_Dockerfile            # PyTorch + MONAI + SimpleITK container
├── configs/
│   └── model_arch.json                # DenseNet-121 architecture config
├── run_manual.sh                      # Local manual test script
└── README.md
```

## Prerequisites

- [Pegasus WMS](https://pegasus.isi.edu/) >= 5.0
- [HTCondor](https://htcondor.org/) >= 10.2
- Python 3.8+
- Singularity >= 3.0 (required; all jobs run inside containers with `--nv` GPU support)

## Setup

### 1. Build the Docker Container

```bash
cd fl-chest-workflow
docker build --platform linux/amd64 -t fl-chest:latest -f Docker/fl_chest_Dockerfile .
```

### 2. Local Manual Test (No Pegasus Required)

```bash
./run_manual.sh
```

This runs all pipeline steps end-to-end with **synthetic data** (no network required).

## Data Pre-Staging

The workflow separates data **download** (on submit host) from **ingestion** (inside workflow jobs).
This avoids SSL/network issues inside containers and allows data reuse across workflow reruns.

### Download TCIA Data

Run on the **submit host** (where you have normal network access):

```bash
# START HERE for testing (limits to 10 series per collection):
./bin/download_tcia.py \
    --collections LIDC-IDRI NSCLC-Radiomics RIDER \
    --output-dir data/ \
    --max-series 10

# Full download (collections like LIDC-IDRI have 15,000+ series):
./bin/download_tcia.py \
    --collections LIDC-IDRI NSCLC-Radiomics RIDER \
    --output-dir data/

# Resume a previous download (skip collections already packaged):
./bin/download_tcia.py \
    --collections LIDC-IDRI NSCLC-Radiomics RIDER \
    --output-dir data/ \
    --resume
```

This produces per-collection tar.gz archives:
```
data/
├── LIDC-IDRI.tar.gz
├── NSCLC-Radiomics.tar.gz
├── RIDER.tar.gz
└── download_manifest.json
```

**Important notes:**
- **Always start with `--max-series N`** for testing. LIDC-IDRI alone has ~15,000 series and takes days to download fully.
- Progress is logged every 100 series: `Progress [LIDC-IDRI]: 300/10000 series (2 failed, 298 successful)`
- Use `--resume` to skip already-packaged collections when restarting
- Use `--no-verify-ssl` if behind a restrictive proxy
- Rerunning a collection that failed mid-download will skip series already in `.tmp/` (resumable at the series level)

### NIH Chest X-Ray14 Data

Place NIH images and `Data_Entry_2017.csv` in a directory, then pass `--nih-data-dir` to the workflow generator.

## Usage

### Generate Workflow

The workflow generator automatically creates Singularity container transformations and compacts each FL round into a Pegasus SubWorkflow. These behaviors are built-in and require no extra flags.

**With pre-staged TCIA data:**

```bash
./workflow_generator.py \
    --clients LIDC-IDRI NSCLC-Radiomics RIDER \
    --tcia-data-dir data/ \
    --rounds 10 \
    --output workflow.yml
```

**With NIH evaluation:**

```bash
./workflow_generator.py \
    --clients LIDC-IDRI RIDER \
    --tcia-data-dir data/ \
    --nih-data-dir /path/to/nih_preprocessed \
    --rounds 10 \
    --output workflow.yml
```

**Test mode** (single client, 2 rounds, 1 local epoch, skips NIH):

```bash
./workflow_generator.py \
    --clients RIDER \
    --tcia-data-dir data/ \
    --test \
    --output test_workflow.yml
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--clients` | (required) | TCIA collection names to use as FL clients |
| `--tcia-data-dir` | None | Directory containing pre-staged tar.gz files |
| `--rounds` | 10 | Number of FL training rounds |
| `--local-epochs` | 5 | Local epochs per round |
| `--batch-size` | 16 | Training batch size |
| `--lr` | 1e-4 | Learning rate |
| `--strategy` | `fedavg` | Aggregation strategy (`fedavg`, `fedprox`, `scaffold`) |
| `--fedprox-mu` | 0.0 | FedProx mu parameter |
| `--nih-data-dir` | None | Path to preprocessed NIH Chest X-Ray14 data |
| `--skip-nih-eval` | false | Skip NIH evaluation |
| `--skip-preprocessing` | false | Skip ingestion/preprocessing (reuse existing) |
| `--test` | false | Test mode with single small collection |
| `-e`, `--execution-site-name` | `condorpool` | HTCondor execution site name |
| `-s`, `--skip-sites-catalog` | false | Skip site catalog creation |
| `-o`, `--output` | `workflow.yml` | Output workflow file |

### Submit Workflow

```bash
pegasus-plan --submit -s condorpool -o local workflow.yml
```

### Monitor Workflow

```bash
pegasus-status <run-directory>
pegasus-statistics <run-directory>
```

## Outputs

The workflow produces the following final outputs in the `output/` directory:

| Output | Description |
|--------|-------------|
| `models/global_model_r*.pt` | Aggregated global model per round |
| `models/global_model_initial.pt` | Initial global model |
| `metrics/round_*_aggregation.json` | Per-round aggregation metrics |
| `metrics/final_tcia_metrics.json` | Final TCIA evaluation metrics |
| `metrics/final_nih_metrics.json` | Final NIH evaluation metrics |
| `metrics/ensemble_*.json` | Ensemble evaluation metrics |
| `client_stats/*_stats.json` | Per-client heterogeneity statistics |
| `reports/FL_report.html` | HTML experiment report |
| `reports/FL_report.pdf` | PDF experiment report |

## Resource Requirements

| Step | Memory | Cores | GPUs |
|------|--------|-------|------|
| download_tcia | N/A | N/A | N/A |
| ingest_tcia | 4 GB | 2 | 0 |
| preprocess | 8 GB | 4 | 0 |
| compute_stats | 2 GB | 1 | 0 |
| initialize_model | 2 GB | 1 | 0 |
| train_client | 16 GB | 4 | 1 |
| aggregate | 32 GB | 8 | 0 |
| evaluate | 8 GB | 2 | 1 |
| ensemble_evaluate | 8 GB | 2 | 1 |
| generate_report | 4 GB | 2 | 0 |

## Running on FABRIC

The workflow can also be run on the [FABRIC testbed](https://fabric-testbed.net/)
by deploying a distributed Pegasus/HTCondor cluster across FABRIC sites.

See the [Pegasus-FABRIC Artifact](https://artifacts.fabric-testbed.net/artifacts/53da4088-a175-4f0c-9e25-a4a371032a39)
or the [pegasus-fabric.ipynb](https://github.com/fabric-testbed/jupyter-examples/blob/f7be0c75f22544c72d7b3e3fa42bbdfd9d8bb841/fabric_examples/complex_recipes/pegasus/pegasus-fabric.ipynb)
notebook.

## Troubleshooting

### SSL/TCIA API Errors

If you see SSL handshake errors from inside containers:
```
SSLError: [SSL: UNEXPECTED_EOF_WHILE_READING]
```

This is expected in restricted network environments. The workflow is designed to avoid this:
1. Run `download_tcia.py` on the submit host (where you have normal network)
2. Pass `--tcia-data-dir` to use the pre-staged tar.gz files

### Container Build Issues on Apple Silicon

If building the Docker image on macOS ARM64 fails with package resolution errors,
use `--platform linux/amd64`:
```bash
docker build --platform linux/amd64 -t fl-chest:latest -f Docker/fl_chest_Dockerfile .
```

### Test Mode

For quick validation without real data or GPUs:
```bash
./run_manual.sh          # Local test with synthetic data (CPU)
./workflow_generator.py --clients RIDER --test --output test.yml
```
