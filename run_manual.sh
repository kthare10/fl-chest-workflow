#!/bin/bash
#
# Manual test script for the FL Chest Imaging Workflow
# Tests each pipeline step locally before running with Pegasus.
#
# Usage:
#   ./run_manual.sh [--use-docker]
# Examples:
#   ./run_manual.sh
#   ./run_manual.sh --use-docker
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DATA_DIR="${SCRIPT_DIR}/test_data"
TEST_OUTPUT="${SCRIPT_DIR}/test_output"
CONTAINER_IMAGE="fl-chest:latest"

USE_DOCKER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --use-docker)
            USE_DOCKER=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--use-docker]"
            exit 1
            ;;
    esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()    { echo ""; echo -e "${GREEN}========================================${NC}"; echo -e "${GREEN}STEP: $1${NC}"; echo -e "${GREEN}========================================${NC}"; }

run_cmd() {
    if [ "$USE_DOCKER" = true ]; then
        docker run --rm \
            -v "${SCRIPT_DIR}:/workflow" \
            -w /workflow \
            "${CONTAINER_IMAGE}" \
            "$@"
    else
        "$@"
    fi
}

echo ""
echo "=============================================="
echo "  FL Chest Imaging Workflow — Manual Test"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Use Docker:    ${USE_DOCKER}"
echo "  Test Data Dir: ${TEST_DATA_DIR}"
echo "  Output Dir:    ${TEST_OUTPUT}"
echo ""

mkdir -p "${TEST_DATA_DIR}" "${TEST_OUTPUT}"

CLIENT="RIDER"

# ==============================================================
log_step "1. ingest_tcia"
run_cmd python3 "${SCRIPT_DIR}/bin/ingest_tcia.py" \
    --collection-name "${CLIENT}" \
    --output-dir "${TEST_OUTPUT}/data/${CLIENT}" \
    --output-manifest "${TEST_OUTPUT}/data/${CLIENT}/manifest.json" \
    --synthetic

if [ -f "${TEST_OUTPUT}/data/${CLIENT}/manifest.json" ]; then
    log_success "ingest_tcia completed"
else
    log_error "ingest_tcia failed — no manifest"
    exit 1
fi

# ==============================================================
log_step "2. preprocess"
run_cmd python3 "${SCRIPT_DIR}/bin/preprocess.py" \
    --input-dir "${TEST_OUTPUT}/data/${CLIENT}" \
    --output-dir "${TEST_OUTPUT}/preprocessed/${CLIENT}" \
    --output-labels "${TEST_OUTPUT}/preprocessed/${CLIENT}/labels.json" \
    --synthetic

if [ -f "${TEST_OUTPUT}/preprocessed/${CLIENT}/labels.json" ]; then
    log_success "preprocess completed"
else
    log_error "preprocess failed — no labels"
    exit 1
fi

# ==============================================================
log_step "3. compute_stats"
run_cmd python3 "${SCRIPT_DIR}/bin/compute_stats.py" \
    --input-dir "${TEST_OUTPUT}/preprocessed/${CLIENT}" \
    --output-json "${TEST_OUTPUT}/stats/${CLIENT}_stats.json" \
    --manifest "${TEST_OUTPUT}/data/${CLIENT}/manifest.json"

if [ -f "${TEST_OUTPUT}/stats/${CLIENT}_stats.json" ]; then
    log_success "compute_stats completed"
else
    log_error "compute_stats failed — no stats"
    exit 1
fi

# ==============================================================
log_step "4. initialize_model"
run_cmd python3 "${SCRIPT_DIR}/bin/initialize_model.py" \
    --arch-config "${SCRIPT_DIR}/configs/model_arch.json" \
    --output-model "${TEST_OUTPUT}/models/global_initial.pt" \
    --output-config "${TEST_OUTPUT}/models/model_arch.json"

if [ -f "${TEST_OUTPUT}/models/global_initial.pt" ]; then
    log_success "initialize_model completed"
else
    log_error "initialize_model failed — no model"
    exit 1
fi

# ==============================================================
log_step "5. train_client"
run_cmd python3 "${SCRIPT_DIR}/bin/train_client.py" \
    --client-id "${CLIENT}" \
    --data-dir "${TEST_OUTPUT}/preprocessed/${CLIENT}" \
    --arch-config "${TEST_OUTPUT}/models/model_arch.json" \
    --global-model "${TEST_OUTPUT}/models/global_initial.pt" \
    --output-model "${TEST_OUTPUT}/models/local_${CLIENT}_r1.pt" \
    --output-metrics "${TEST_OUTPUT}/metrics/${CLIENT}_r1.json" \
    --round 1 \
    --device cpu

if [ -f "${TEST_OUTPUT}/models/local_${CLIENT}_r1.pt" ]; then
    log_success "train_client completed"
else
    log_error "train_client failed — no local model"
    exit 1
fi

# ==============================================================
log_step "6. aggregate"
run_cmd python3 "${SCRIPT_DIR}/bin/aggregate.py" \
    --input-models "${TEST_OUTPUT}/models/local_${CLIENT}_r1.pt" \
    --prev-global "${TEST_OUTPUT}/models/global_initial.pt" \
    --output-model "${TEST_OUTPUT}/models/global_r1.pt" \
    --output-metrics "${TEST_OUTPUT}/metrics/round_1.json" \
    --client-metrics "${TEST_OUTPUT}/metrics/${CLIENT}_r1.json" \
    --strategy fedavg \
    --round 1

if [ -f "${TEST_OUTPUT}/models/global_r1.pt" ]; then
    log_success "aggregate completed"
else
    log_error "aggregate failed — no global model"
    exit 1
fi

# ==============================================================
log_step "7. evaluate"
run_cmd python3 "${SCRIPT_DIR}/bin/evaluate.py" \
    --model "${TEST_OUTPUT}/models/global_r1.pt" \
    --arch-config "${TEST_OUTPUT}/models/model_arch.json" \
    --data-dir "${TEST_OUTPUT}/preprocessed/${CLIENT}" \
    --output-metrics "${TEST_OUTPUT}/metrics/final_tcia.json" \
    --dataset-name tcia \
    --device cpu

if [ -f "${TEST_OUTPUT}/metrics/final_tcia.json" ]; then
    log_success "evaluate completed"
else
    log_error "evaluate failed — no metrics"
    exit 1
fi

# ==============================================================
log_step "8. generate_report (stub)"
run_cmd python3 "${SCRIPT_DIR}/bin/generate_report.py" \
    --round-metrics "${TEST_OUTPUT}/metrics/round_1.json" \
    --final-metrics "${TEST_OUTPUT}/metrics/final_tcia.json" \
    --client-stats "${TEST_OUTPUT}/stats/${CLIENT}_stats.json" \
    --output-html "${TEST_OUTPUT}/FL_report.html" \
    --output-pdf "${TEST_OUTPUT}/FL_report.pdf"

if [ -f "${TEST_OUTPUT}/FL_report.html" ]; then
    log_success "generate_report completed"
else
    log_error "generate_report failed — no report"
    exit 1
fi

# ==============================================================
echo ""
echo "=============================================="
echo "  MANUAL TEST COMPLETED SUCCESSFULLY!"
echo "=============================================="
echo "
Output files in: ${TEST_OUTPUT}"
find "${TEST_OUTPUT}" -type f | sort
log_success "All steps passed! Ready to run with Pegasus."
