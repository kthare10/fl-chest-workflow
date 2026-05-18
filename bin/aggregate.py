#!/usr/bin/env python3

"""Aggregate local model updates into a new global model.

Implements:
- Weighted FedAvg (default)
- FedProx (just averages, since local models already applied prox term)
- SCAFFOLD (placeholder for future implementation)

Computes sample-based weights from client metrics files.
"""

import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def extract_client_weights(local_metrics_files):
    """Compute client weights from metrics files.

    Returns dict mapping client_name -> num_train_samples.
    Falls back to equal weights if metrics don't contain sample counts.
    """
    weights = {}
    for mf in local_metrics_files:
        if not os.path.exists(mf):
            continue
        try:
            with open(mf, "r") as f:
                m = json.load(f)
            client_id = m.get("client_id", os.path.basename(mf))
            n = m.get("n_train", 1)
            weights[client_id] = max(n, 1)
        except Exception as e:
            logger.warning(f"Failed to read metrics {mf}: {e}")
            continue

    if not weights:
        logger.warning("No valid metrics found; using equal weights")
        return None
    return weights


def fedavg_aggregate(state_dicts, weights=None):
    """Weighted FedAvg: aggregate state dicts.

    Args:
        state_dicts: list of OrderedDict / dict from model.state_dict()
        weights: dict mapping client_id -> num_samples, or None for equal weights

    Returns:
        aggregated state_dict
    """
    import torch

    if not state_dicts:
        raise ValueError("No state dicts to aggregate")

    if weights is None:
        # Equal weights
        weight_list = [1.0] * len(state_dicts)
    else:
        weight_list = list(weights.values())

    total_weight = sum(weight_list)
    if total_weight == 0:
        total_weight = len(state_dicts)
        weight_list = [1.0] * len(state_dicts)

    # Normalize weights
    norm_weights = [w / total_weight for w in weight_list]

    # Initialize with first state dict * weight
    aggregated = {}
    for key in state_dicts[0].keys():
        aggregated[key] = state_dicts[0][key].clone() * norm_weights[0]

    # Accumulate remaining state dicts
    for i in range(1, len(state_dicts)):
        for key in aggregated:
            aggregated[key] += state_dicts[i][key] * norm_weights[i]

    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate local model updates into a new global model."
    )
    parser.add_argument(
        "--input-models",
        action="append",
        required=True,
        help="Local model checkpoint(s) (.pt)",
    )
    parser.add_argument("--prev-global", required=True, help="Previous global model (.pt)")
    parser.add_argument("--output-model", required=True, help="Output aggregated global model (.pt)")
    parser.add_argument("--output-metrics", required=True, help="Output aggregation metrics JSON")
    parser.add_argument(
        "--strategy",
        choices=["fedavg", "fedprox", "scaffold"],
        default="fedavg",
    )
    parser.add_argument(
        "--client-metrics",
        action="append",
        default=[],
        help="Client metrics JSON files for weight computation",
    )
    parser.add_argument("--round", type=int, default=0, help="FL round number for metrics")

    args = parser.parse_args()

    logger.info(f"Input models: {args.input_models}")
    logger.info(f"Prev global:  {args.prev_global}")
    logger.info(f"Strategy:     {args.strategy}")
    logger.info(f"Round:        {args.round}")

    model_dir = os.path.dirname(args.output_model)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    metrics_dir = os.path.dirname(args.output_metrics)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    # Compute weights from client metrics
    weights = None
    if args.client_metrics:
        weights = extract_client_weights(args.client_metrics)
        if weights:
            logger.info(f"Client weights: {weights}")

    try:
        import torch
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False

    if not HAS_TORCH:
        logger.error("torch not available; writing placeholder")
        metrics = {
            "round": args.round,
            "strategy": args.strategy,
            "status": "no_torch",
            "num_clients": len(args.input_models),
        }
        with open(args.output_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        import pickle
        with open(args.output_model, "wb") as f:
            pickle.dump({"status": "placeholder"}, f)
        logger.info("aggregate: wrote placeholders (no torch).")
        return

    # Load all local state dicts
    local_states = []
    for model_path in args.input_models:
        if not os.path.exists(model_path):
            logger.error(f"Local model not found: {model_path}")
            sys.exit(1)
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state" in state:
                logger.warning(f"Model {model_path} is a placeholder")
                continue
            local_states.append(state)
        except Exception as e:
            logger.error(f"Failed to load {model_path}: {e}")
            sys.exit(1)

    if not local_states:
        logger.error("No valid local models to aggregate")
        sys.exit(1)

    logger.info(f"Aggregating {len(local_states)} local models")

    # Validate state dict keys match
    first_keys = set(local_states[0].keys())
    for i, state in enumerate(local_states[1:], 1):
        if set(state.keys()) != first_keys:
            logger.error(f"State dict keys mismatch at model {i}")
            sys.exit(1)

    # Aggregate
    if args.strategy in ("fedavg", "fedprox"):
        aggregated = fedavg_aggregate(local_states, weights=weights)
    elif args.strategy == "scaffold":
        logger.warning("SCAFFOLD not yet implemented; falling back to FedAvg")
        aggregated = fedavg_aggregate(local_states, weights=weights)
    else:
        aggregated = fedavg_aggregate(local_states, weights=weights)

    # Save aggregated global model
    torch.save(aggregated, args.output_model)
    logger.info(f"Aggregated model saved: {args.output_model}")

    # Save metrics — sanitize weights to plain Python types for JSON
    safe_weights = None
    if weights is not None:
        safe_weights = {str(k): int(v) for k, v in weights.items()}
    metrics = {
        "round": args.round,
        "strategy": args.strategy,
        "status": "ok",
        "num_clients": len(local_states),
        "client_weights": safe_weights,
    }
    with open(args.output_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Aggregation metrics saved: {args.output_metrics}")
    logger.info("aggregate complete.")


if __name__ == "__main__":
    main()
