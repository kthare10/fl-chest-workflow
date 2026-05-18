#!/usr/bin/env python3

"""
Pegasus workflow generator for the Federated Learning Chest Imaging Workflow.

Stages:
1. Data Preparation (parallel per TCIA client): ingest → preprocess → stats
2. Model Initialization: create initial global model
3. Federated Learning Rounds (iterative):
   For r = 1..R:
     broadcast global model → local training (parallel per client)
     local models → aggregate → new global model
4. Final Evaluation (parallel): TCIA eval + NIH eval (optional)
5. Reporting: generate HTML/PDF report

Usage:
    ./workflow_generator.py --clients LIDC-IDRI NSCLC-Radiomics RIDER \
        --rounds 10 --output workflow.yml
    ./workflow_generator.py --clients RIDER --test --output test_workflow.yml
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from Pegasus.api import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Per-tool resource configuration (matches SPEC §5.2)
TOOL_CONFIGS = {
    "ingest_tcia": {"memory": "4 GB", "cores": 2, "gpus": 0},
    "preprocess": {"memory": "8 GB", "cores": 4, "gpus": 0},
    "compute_stats": {"memory": "2 GB", "cores": 1, "gpus": 0},
    "initialize_model": {"memory": "2 GB", "cores": 1, "gpus": 0},
    "train_client": {"memory": "15 GB", "cores": 4, "gpus": 1},
    "aggregate": {"memory": "15 GB", "cores": 8, "gpus": 0},
    "evaluate": {"memory": "8 GB", "cores": 2, "gpus": 1},
    "generate_report": {"memory": "4 GB", "cores": 2, "gpus": 0},
}


class FLChestWorkflow:
    """Generate Pegasus workflow for Federated Learning on chest imaging."""

    wf = None
    sc = None
    tc = None
    rc = None
    props = None

    dagfile = None
    wf_dir = None
    shared_scratch_dir = None
    local_storage_dir = None
    wf_name = "fl-chest-workflow"

    def __init__(self, dagfile="workflow.yml"):
        self.dagfile = dagfile
        self.wf_dir = str(Path(__file__).parent.resolve())
        self.shared_scratch_dir = os.path.join(self.wf_dir, "scratch")
        self.local_storage_dir = os.path.join(self.wf_dir, "output")

    def write(self):
        """Write all catalogs and workflow to files."""
        if self.sc is not None:
            self.sc.write()
        self.props.write()
        self.rc.write()
        self.tc.write()
        self.wf.write(file=self.dagfile)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    def create_pegasus_properties(self):
        self.props = Properties()
        self.props["pegasus.transfer.threads"] = "16"

    # ------------------------------------------------------------------
    # Site Catalog
    # ------------------------------------------------------------------
    def create_sites_catalog(self, exec_site_name="condorpool"):
        self.sc = SiteCatalog()

        local = Site("local").add_directories(
            Directory(
                Directory.SHARED_SCRATCH, self.shared_scratch_dir
            ).add_file_servers(
                FileServer("file://" + self.shared_scratch_dir, Operation.ALL)
            ),
            Directory(
                Directory.LOCAL_STORAGE, self.local_storage_dir
            ).add_file_servers(
                FileServer("file://" + self.local_storage_dir, Operation.ALL)
            ),
        )

        exec_site = (
            Site(exec_site_name)
            .add_condor_profile(universe="vanilla")
            .add_pegasus_profile(style="condor")
        )

        self.sc.add_sites(local, exec_site)

    # ------------------------------------------------------------------
    # Transformation Catalog
    # ------------------------------------------------------------------
    def create_transformation_catalog(self, exec_site_name="condorpool", use_container=True, container_image=None):
        self.tc = TransformationCatalog()

        fl_container = None
        if use_container:
            if container_image is None:
                container_image = "docker://kthare10/fl-chest:latest"
            fl_container = Container(
                "fl_chest_container",
                container_type=Container.SINGULARITY,
                image=container_image,
                image_site="docker_hub",
                arguments="--nv",
            )
            self.tc.add_containers(fl_container)

        transformations = []
        for tool_name, config in TOOL_CONFIGS.items():
            profiles = {
                "memory": config["memory"],
                "cores": config.get("cores", 1),
            }
            if config.get("gpus", 0) > 0:
                profiles["gpus"] = config["gpus"]

            tx = Transformation(
                tool_name,
                site=exec_site_name,
                pfn=os.path.join(self.wf_dir, f"bin/{tool_name}.py"),
                is_stageable=True,
            ).add_pegasus_profile(**profiles)

            if use_container and fl_container is not None:
                tx.container = fl_container

            transformations.append(tx)

        self.tc.add_transformations(*transformations)

    # ------------------------------------------------------------------
    # Replica Catalog
    # ------------------------------------------------------------------
    def create_replica_catalog(self, args):
        self.rc = ReplicaCatalog()

        # Register model architecture config if provided.
        model_arch_path = os.path.join(self.wf_dir, "configs", "model_arch.json")
        if os.path.exists(model_arch_path):
            self.rc.add_replica(
                "local",
                "input_model_arch.json",
                "file://" + os.path.abspath(model_arch_path),
            )

    # ------------------------------------------------------------------
    # Workflow DAG
    # ------------------------------------------------------------------
    def create_workflow(self, args):
        self.wf = Workflow(self.wf_name, infer_dependencies=True)

        # Stage 1: Data Preparation (parallel per client)
        preprocessed_dirs = {}
        client_stats_files = []
        for client in args.clients:
            pp_dir, stats_file = self._add_client_data_prep(client, args)
            preprocessed_dirs[client] = pp_dir
            client_stats_files.append(stats_file)

        # Stage 2: Initialize global model
        global_model, model_arch_file = self._add_initialize_model_job(args)

        # Stage 3: Federated Learning Rounds (iterative, compacted via SubWorkflows)
        current_global = global_model
        all_round_metrics = []
        for r in range(1, args.rounds + 1):
            subwf_file, new_global, agg_metrics = self._write_round_subworkflow(
                r, args.clients, current_global, preprocessed_dirs, model_arch_file, args
            )
            subwf_file_obj = File(subwf_file)
            subwf_job = SubWorkflow(
                subwf_file_obj,
                is_planned=False,
                _id=f"round_{r}_subwf",
                node_label=f"round_{r}_subwf",
            )
            subwf_job.add_inputs(current_global)
            for client in args.clients:
                subwf_job.add_inputs(preprocessed_dirs[client])
            if model_arch_file is not None:
                subwf_job.add_inputs(model_arch_file)
            # Inner jobs keep outputs in subworkflow scratch (stage_out=False).
            # SubWorkflow boundary stages them to parent (stage_out=True).
            subwf_job.add_outputs(new_global, stage_out=True, register_replica=False)
            subwf_job.add_outputs(agg_metrics, stage_out=True, register_replica=False)
            self.wf.add_jobs(subwf_job)
            all_round_metrics.append(agg_metrics)
            current_global = new_global

        # Stage 4: Evaluation (parallel TCIA + NIH)
        final_eval_files = []
        final_tcia = self._add_evaluate_job(
            "tcia", current_global, preprocessed_dirs, model_arch_file, args
        )
        final_eval_files.append(final_tcia)

        if not args.skip_nih_eval and args.nih_data_dir:
            final_nih = self._add_evaluate_job(
                "nih", current_global, args.nih_data_dir, model_arch_file, args
            )
            final_eval_files.append(final_nih)

        # Stage 5: Generate Report
        self._add_report_job(all_round_metrics, final_eval_files, client_stats_files, args)

    def _add_client_data_prep(self, client, args):
        """Add Stage 1 jobs for one client: ingest -> preprocess -> stats.

        Returns:
            (preprocessed_labels_file, client_stats_file) for downstream dependencies.
        """
        safe_client = client.replace(" ", "_")

        # --- Job 1: ingest_tcia ---
        raw_dir = f"data/{safe_client}"
        ingest_manifest = File(f"{safe_client}_manifest.json")
        tar_file = None

        ingest_job = (
            Job("ingest_tcia", _id=f"ingest_{safe_client}", node_label=f"ingest_{safe_client}")
            .add_args("--collection-name", client)
            .add_args("--output-dir", raw_dir)
            .add_args("--output-manifest", ingest_manifest)
            .add_outputs(ingest_manifest, stage_out=False, register_replica=False)
            .add_pegasus_profile(label=client)
        )

        # If pre-staged tar.gz is provided, register it as a Pegasus input and pass to job
        if args.tcia_data_dir:
            tar_gz_path = os.path.join(args.tcia_data_dir, f"{client}.tar.gz")
            if os.path.exists(tar_gz_path):
                tar_lfn = f"{safe_client}_data.tar.gz"
                # Register in replica catalog so Pegasus stages it via condor I/O
                self.rc.add_replica("local", tar_lfn, "file://" + os.path.abspath(tar_gz_path))
                tar_file = File(tar_lfn)
                ingest_job.add_inputs(tar_file)
                ingest_job.add_args("--input-tar", tar_file)
                logger.info(f"Registered {tar_lfn} for client {client}")
            else:
                logger.warning(f"Tar file not found for {client}: {tar_gz_path}")

        self.wf.add_jobs(ingest_job)

        # --- Job 2: preprocess ---
        pp_dir = f"preprocessed/{safe_client}"
        pp_labels = File(f"{safe_client}_labels.json")

        preprocess_job = (
            Job("preprocess", _id=f"preprocess_{safe_client}", node_label=f"preprocess_{safe_client}")
            .add_args("--input-dir", raw_dir)
            .add_args("--output-dir", pp_dir)
            .add_args("--output-labels", pp_labels)
            .add_args("--manifest", ingest_manifest)
            .add_inputs(ingest_manifest)
            .add_outputs(pp_labels, stage_out=False, register_replica=False)
            .add_pegasus_profile(label=client)
        )
        if tar_file is not None:
            preprocess_job.add_inputs(tar_file)
            preprocess_job.add_args("--input-tar", tar_file)
        self.wf.add_jobs(preprocess_job)

        # --- Job 3: compute_stats ---
        stats_file = File(f"{safe_client}_stats.json")

        stats_job = (
            Job("compute_stats", _id=f"stats_{safe_client}", node_label=f"stats_{safe_client}")
            .add_args("--input-dir", pp_dir)
            .add_args("--output-json", stats_file)
            .add_args("--manifest", ingest_manifest)
            .add_inputs(pp_labels, ingest_manifest)
            .add_outputs(stats_file, stage_out=True, register_replica=False)
            .add_pegasus_profile(label=client)
        )
        self.wf.add_jobs(stats_job)

        return pp_labels, stats_file

    def _add_initialize_model_job(self, args):
        """Add Stage 2 job: initialize global model.

        Returns:
            (global_model_file, model_arch_file)
        """
        global_model = File("global_model_initial.pt")
        model_arch_file = None

        init_job = (
            Job("initialize_model", _id="initialize_model", node_label="initialize_model")
            .add_args("--output-model", global_model)
            .add_outputs(global_model, stage_out=False, register_replica=False)
        )

        # Optionally use provided architecture config from replica catalog.
        # Input LFN is "input_model_arch.json"; output LFN is "model_arch.json".
        model_arch_path = os.path.join(self.wf_dir, "configs", "model_arch.json")
        if os.path.exists(model_arch_path):
            input_arch_file = File("input_model_arch.json")
            model_arch_file = File("model_arch.json")
            init_job.add_inputs(input_arch_file)
            init_job.add_args("--arch-config", input_arch_file)
            init_job.add_args("--output-config", model_arch_file)
            init_job.add_outputs(model_arch_file, stage_out=False, register_replica=False)

        self.wf.add_jobs(init_job)
        return global_model, model_arch_file

    def _write_round_subworkflow(self, round_num, clients, prev_global, pp_labels_map, model_arch_file, args):
        """Create and write a separate workflow for one FL round (DAG compaction).

        The subworkflow contains parallel training jobs for all clients,
        followed by an aggregation job. It is referenced from the parent
        workflow via a SubWorkflow job.

        Inner job outputs use stage_out=False so they stay in the subworkflow
        scratch area. The SubWorkflow boundary in the parent DAG handles
        staging them out to the parent's output directory.

        Returns:
            (subworkflow_lfn, new_global_model_file, agg_metrics_file)
        """
        subwf_name = f"round_{round_num}_subwf"
        subwf_file = os.path.join(self.wf_dir, f"{subwf_name}.yml")

        subwf = Workflow(subwf_name, infer_dependencies=True)

        # Training jobs (parallel per client)
        local_models = []
        local_metrics = []
        for client in clients:
            safe_client = client.replace(" ", "_")
            pp_dir = f"preprocessed/{safe_client}"

            local_model = File(f"{safe_client}_local_r{round_num}.pt")
            local_metric = File(f"{safe_client}_metrics_r{round_num}.json")

            train_job = (
                Job("train_client", _id=f"train_{safe_client}_r{round_num}",
                    node_label=f"train_{safe_client}_r{round_num}")
                .add_args("--client-id", client)
                .add_args("--data-dir", pp_dir)
                .add_args("--global-model", prev_global)
                .add_args("--output-model", local_model)
                .add_args("--output-metrics", local_metric)
                .add_args("--round", str(round_num))
                .add_args("--epochs", str(args.local_epochs))
                .add_args("--batch-size", str(args.batch_size))
                .add_args("--lr", str(args.lr))
                .add_args("--fedprox-mu", str(args.fedprox_mu))
                .add_inputs(prev_global, pp_labels_map[client])
                .add_outputs(local_model, stage_out=False, register_replica=False)
                .add_outputs(local_metric, stage_out=False, register_replica=False)
                .add_pegasus_profile(label=client)
            )
            if model_arch_file is not None:
                train_job.add_args("--arch-config", model_arch_file)
                train_job.add_inputs(model_arch_file)
            train_job.add_args("--device", "cuda")
            subwf.add_jobs(train_job)

            local_models.append(local_model)
            local_metrics.append(local_metric)

        # Aggregation job
        new_global = File(f"global_model_r{round_num}.pt")
        agg_metrics = File(f"round_{round_num}_aggregation.json")

        agg_job = (
            Job("aggregate", _id=f"aggregate_r{round_num}", node_label=f"aggregate_r{round_num}")
            .add_args("--prev-global", prev_global)
            .add_args("--output-model", new_global)
            .add_args("--output-metrics", agg_metrics)
            .add_args("--strategy", args.strategy)
            .add_args("--round", str(round_num))
            .add_inputs(prev_global)
        )
        for lm in local_models:
            agg_job.add_args("--input-models", lm)
            agg_job.add_inputs(lm)
        for lmet in local_metrics:
            agg_job.add_args("--client-metrics", lmet)
            agg_job.add_inputs(lmet)

        # Keep outputs in subworkflow scratch (stage_out=False).
        # The parent SubWorkflow job handles staging them to the parent
        # output directory (stage_out=True on lines 210-211).
        agg_job.add_outputs(new_global, stage_out=False, register_replica=False)
        agg_job.add_outputs(agg_metrics, stage_out=False, register_replica=False)
        subwf.add_jobs(agg_job)

        # Write subworkflow file
        subwf.write(file=subwf_file)
        logger.info(f"Wrote round {round_num} subworkflow to {subwf_file}")

        # Register subworkflow file in ReplicaCatalog so Pegasus can stage it
        subwf_lfn = os.path.basename(subwf_file)
        self.rc.add_replica(
            "local",
            subwf_lfn,
            "file://" + os.path.abspath(subwf_file),
        )
        logger.info(f"Registered subworkflow replica: {subwf_lfn} -> {subwf_file}")

        return subwf_lfn, new_global, agg_metrics

    def _add_evaluate_job(self, dataset_name, global_model, data_ref, model_arch_file, args):
        """Add Stage 4 job: evaluate global model on a dataset."""
        metrics_file = File(f"final_{dataset_name}_metrics.json")

        eval_job = (
            Job("evaluate",
                _id=f"evaluate_{dataset_name}",
                node_label=f"evaluate_{dataset_name}")
            .add_args("--model", global_model)
            .add_args("--output-metrics", metrics_file)
            .add_args("--dataset-name", dataset_name)
            .add_inputs(global_model)
            .add_outputs(metrics_file, stage_out=True, register_replica=False)
        )

        if model_arch_file is not None:
            eval_job.add_args("--arch-config", model_arch_file)
            eval_job.add_inputs(model_arch_file)

        if dataset_name == "tcia":
            pp_dir = "preprocessed"
            eval_job.add_args("--data-dir", pp_dir)
            for client in args.clients:
                safe_client = client.replace(" ", "_")
                pp_labels = data_ref[client]
                eval_job.add_inputs(pp_labels)
        else:
            eval_job.add_args("--data-dir", data_ref)

        eval_job.add_args("--device", "cuda")

        self.wf.add_jobs(eval_job)
        return metrics_file

    def _add_report_job(self, round_metrics, final_metrics, client_stats, args):
        """Add Stage 5 job: generate HTML/PDF report."""
        html_report = File("FL_report.html")
        pdf_report = File("FL_report.pdf")

        report_job = (
            Job("generate_report", _id="generate_report", node_label="generate_report")
            .add_args("--output-html", html_report)
            .add_args("--output-pdf", pdf_report)
        )

        for rm in round_metrics:
            report_job.add_args("--round-metrics", rm)
            report_job.add_inputs(rm)

        for fm in final_metrics:
            report_job.add_args("--final-metrics", fm)
            report_job.add_inputs(fm)

        for cs in client_stats:
            report_job.add_args("--client-stats", cs)
            report_job.add_inputs(cs)

        report_job.add_outputs(html_report, stage_out=True, register_replica=False)
        report_job.add_outputs(pdf_report, stage_out=True, register_replica=False)

        self.wf.add_jobs(report_job)


# ======================================================================
# main() — CLI argument parsing
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="FL Chest Imaging Workflow Generator for Pegasus WMS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --clients LIDC-IDRI NSCLC-Radiomics --rounds 10 --output workflow.yml
  %(prog)s --clients RIDER --test --output test_workflow.yml
  %(prog)s --clients LIDC-IDRI NSCLC-Radiomics --strategy fedprox --fedprox-mu 0.01
  %(prog)s --clients LIDC-IDRI --tcia-data-dir data/ --rounds 5
""",
    )

    # --- Standard Pegasus arguments ---
    parser.add_argument(
        "-s", "--skip-sites-catalog",
        action="store_true",
        help="Skip site catalog creation",
    )
    parser.add_argument(
        "-e", "--execution-site-name",
        metavar="STR",
        type=str,
        default="condorpool",
        help="Execution site name (default: condorpool)",
    )
    parser.add_argument(
        "-o", "--output",
        metavar="STR",
        type=str,
        default="workflow.yml",
        help="Output workflow file (default: workflow.yml)",
    )

    # --- FL-specific arguments ---
    parser.add_argument(
        "--clients",
        nargs="+",
        required=True,
        help="TCIA collection names to use as FL clients",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of FL training rounds (default: 10)",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=5,
        help="Local epochs per round (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--strategy",
        choices=["fedavg", "fedprox", "scaffold"],
        default="fedavg",
        help="Aggregation strategy (default: fedavg)",
    )
    parser.add_argument(
        "--fedprox-mu",
        type=float,
        default=0.0,
        help="FedProx mu parameter (default: 0.0)",
    )
    parser.add_argument(
        "--tcia-data-dir",
        type=str,
        default=None,
        help="Path to PRE-STAGED TCIA data directory (run download_tcia.py first)",
    )
    parser.add_argument(
        "--nih-data-dir",
        type=str,
        default=None,
        help="Path to NIH Chest X-Ray14 preprocessed data for evaluation",
    )
    parser.add_argument(
        "--skip-nih-eval",
        action="store_true",
        help="Skip NIH evaluation",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip ingestion/preprocessing (reuse existing)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode with a single small TCIA collection (2 rounds, 1 epoch)",
    )

    args = parser.parse_args()

    # --- Deduplicate clients ---
    seen = set()
    unique_clients = []
    for c in args.clients:
        safe = c.replace(" ", "_")
        if safe not in seen:
            seen.add(safe)
            unique_clients.append(c)
        else:
            logger.warning(f"Duplicate client detected and skipped: '{c}' (normalizes to '{safe}')")
    if len(unique_clients) < len(args.clients):
        logger.warning(f"Deduplicated clients: {len(args.clients)} -> {len(unique_clients)}")
    args.clients = unique_clients

    # --- Parameter consistency ---
    if args.tcia_data_dir and not os.path.isdir(args.tcia_data_dir):
        logger.warning(f"--tcia-data-dir does not exist: {args.tcia_data_dir}")

    # --- Test mode overrides ---
    if args.test:
        if len(args.clients) > 1:
            logger.info("Test mode: using first client only")
            args.clients = [args.clients[0]]
        args.rounds = 2
        args.local_epochs = 1
        args.skip_nih_eval = True
        logger.info("Test mode enabled: rounds=2, local_epochs=1, skip_nih_eval=True")

    logger.info("=" * 70)
    logger.info("FL CHEST IMAGING WORKFLOW GENERATOR")
    logger.info("=" * 70)
    logger.info(f"Clients:          {args.clients}")
    logger.info(f"Rounds:           {args.rounds}")
    logger.info(f"Local epochs:     {args.local_epochs}")
    logger.info(f"Batch size:       {args.batch_size}")
    logger.info(f"LR:               {args.lr}")
    logger.info(f"Strategy:         {args.strategy}")
    logger.info(f"FedProx mu:       {args.fedprox_mu}")
    logger.info(f"Skip NIH eval:    {args.skip_nih_eval}")
    logger.info(f"NIH data dir:     {args.nih_data_dir}")
    logger.info(f"TCIA data dir:    {args.tcia_data_dir}")
    logger.info(f"Execution site:   {args.execution_site_name}")
    logger.info(f"Output file:      {args.output}")
    logger.info("=" * 70)

    try:
        workflow = FLChestWorkflow(dagfile=args.output)

        workflow.create_pegasus_properties()

        if not args.skip_sites_catalog:
            workflow.create_sites_catalog(
                exec_site_name=args.execution_site_name
            )

        workflow.create_transformation_catalog(
            exec_site_name=args.execution_site_name,
            use_container=True,
        )
        workflow.create_replica_catalog(args)
        workflow.create_workflow(args)
        workflow.write()

        logger.info(f"\nWorkflow written to {args.output}")
        logger.info(
            f"Submit: pegasus-plan --submit "
            f"-s {args.execution_site_name} -o local {args.output}"
        )

    except Exception as e:
        logger.error(f"Failed to generate workflow: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
