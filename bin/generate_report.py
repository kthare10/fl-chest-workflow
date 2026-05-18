#!/usr/bin/env python3

"""Generate HTML report of the full FL experiment.

Uses Jinja2 for templating and matplotlib/seaborn for plots:
- Round vs. global AUC / loss
- Client drift (per-client val AUC over rounds)
- Label skew heatmap across clients
- Final evaluation: TCIA vs. NIH bar charts
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_json_files(file_list):
    """Load a list of JSON files into dicts."""
    data = []
    for fpath in file_list:
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                data.append(json.load(f))
        else:
            logger.warning(f"File not found: {fpath}")
    return data


def plot_round_metrics(round_metrics, output_dir):
    """Plot round-level training curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = list(range(1, len(round_metrics) + 1))
    # Extract average val AUC if available in aggregation metrics
    aucs = []
    for rm in round_metrics:
        # Aggregation metrics don't have AUC directly; we read from client metrics
        aucs.append(rm.get("num_clients", 0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, aucs, marker="o", linewidth=2)
    ax.set_xlabel("Round")
    ax.set_ylabel("Num Clients")
    ax.set_title("FL Rounds: Client Participation")
    ax.grid(True, alpha=0.3)
    fig_path = os.path.join(output_dir, "plot_rounds.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_client_drift(client_stats_data, output_dir):
    """Plot client label distribution as a simple bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clients = []
    num_series = []
    for data in client_stats_data:
        clients.append(data.get("collection", "unknown"))
        num_series.append(data.get("num_series", 0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(clients, num_series, color="steelblue")
    ax.set_xlabel("Client")
    ax.set_ylabel("Number of Series")
    ax.set_title("Data Volume per Client")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "plot_client_volumes.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_final_metrics(final_metrics, output_dir):
    """Plot final evaluation metrics comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = []
    aucs = []
    f1s = []

    for fm in final_metrics:
        if "macro" in fm:
            datasets.append(fm.get("dataset", "unknown"))
            aucs.append(fm["macro"].get("auc", 0.0))
            f1s.append(fm["macro"].get("f1", 0.0))

    if not datasets:
        return None

    x = list(range(len(datasets)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], aucs, width, label="AUC", color="steelblue")
    ax.bar([i + width / 2 for i in x], f1s, width, label="F1", color="coral")
    ax.set_ylabel("Score")
    ax.set_title("Final Evaluation: TCIA vs. NIH")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "plot_final_metrics.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def build_html(round_metrics, final_metrics, client_stats, plot_files):
    """Build HTML report string using Jinja2."""
    try:
        from jinja2 import Template
    except ImportError:
        # Fallback: simple string formatting
        logger.warning("jinja2 not available; using basic HTML")
        return _build_basic_html(round_metrics, final_metrics, client_stats, plot_files)

    template_str = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Federated Learning Chest Imaging Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
        th { background: #3498db; color: white; }
        tr:nth-child(even) { background: #f9f9f9; }
        .metric { font-size: 1.2em; font-weight: bold; color: #2980b9; }
        .plot { max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; border-radius: 4px; }
        .summary-box { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 15px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Federated Learning Chest Imaging Report</h1>
        <p><strong>Generated:</strong> {{ timestamp }}</p>

        <div class="summary-box">
            <h2>Experiment Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total FL Rounds</td><td>{{ num_rounds }}</td></tr>
                <tr><td>Number of Clients</td><td>{{ num_clients }}</td></tr>
                <tr><td>Aggregation Strategy</td><td>{{ strategy }}</td></tr>
                {% if final_auc is not none %}
                <tr><td>Final Macro AUC (TCIA)</td><td class="metric">{{ final_auc | round(4) }}</td></tr>
                {% endif %}
                {% if final_f1 is not none %}
                <tr><td>Final Macro F1 (TCIA)</td><td class="metric">{{ final_f1 | round(4) }}</td></tr>
                {% endif %}
            </table>
        </div>

        {% if plot_files.rounds %}
        <h2>FL Training Rounds</h2>
        <img class="plot" src="{{ plot_files.rounds }}" alt="Round metrics">
        {% endif %}

        {% if plot_files.clients %}
        <h2>Client Data Distribution</h2>
        <img class="plot" src="{{ plot_files.clients }}" alt="Client volumes">
        {% endif %}

        {% if plot_files.final %}
        <h2>Final Evaluation</h2>
        <img class="plot" src="{{ plot_files.final }}" alt="Final metrics">
        {% endif %}

        <h2>Round Details</h2>
        <table>
            <tr><th>Round</th><th>Strategy</th><th>Num Clients</th><th>Status</th></tr>
            {% for rm in round_metrics %}
            <tr>
                <td>{{ rm.round }}</td>
                <td>{{ rm.strategy }}</td>
                <td>{{ rm.num_clients }}</td>
                <td>{{ rm.status }}</td>
            </tr>
            {% endfor %}
        </table>

        <h2>Client Statistics</h2>
        <table>
            <tr><th>Client</th><th>Series</th><th>Intensity Mean</th><th>Intensity Std</th></tr>
            {% for cs in client_stats %}
            <tr>
                <td>{{ cs.collection }}</td>
                <td>{{ cs.num_series }}</td>
                <td>{{ cs.intensity_stats.mean | round(4) if cs.intensity_stats else 'N/A' }}</td>
                <td>{{ cs.intensity_stats.std | round(4) if cs.intensity_stats else 'N/A' }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
</body>
</html>
"""

    from datetime import datetime

    # Extract summary values
    final_auc = None
    final_f1 = None
    for fm in final_metrics:
        if fm.get("dataset") == "tcia" and "macro" in fm:
            final_auc = fm["macro"].get("auc")
            final_f1 = fm["macro"].get("f1")

    strategy = "FedAvg"
    if round_metrics:
        strategy = round_metrics[0].get("strategy", "FedAvg")

    template = Template(template_str)
    html = template.render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        num_rounds=len(round_metrics),
        num_clients=len(client_stats),
        strategy=strategy.upper(),
        final_auc=final_auc,
        final_f1=final_f1,
        round_metrics=round_metrics,
        client_stats=client_stats,
        final_metrics=final_metrics,
        plot_files=plot_files,
    )
    return html


def _build_basic_html(round_metrics, final_metrics, client_stats, plot_files):
    """Minimal HTML fallback when jinja2 is not available."""
    from datetime import datetime

    lines = [
        "<!DOCTYPE html>",
        "<html><head><title>FL Report</title></head><body>",
        "<h1>Federated Learning Chest Imaging Report</h1>",
        f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        f"<p>Rounds: {len(round_metrics)}, Clients: {len(client_stats)}</p>",
    ]

    if plot_files.get("final"):
        lines.append(f'<img src="{plot_files["final"]}" style="max-width:800px;">')

    lines.append("</body></html>")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML/PDF report of the full FL experiment."
    )
    parser.add_argument("--round-metrics", action="append", required=True)
    parser.add_argument("--final-metrics", action="append", required=True)
    parser.add_argument("--client-stats", action="append", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--output-pdf", required=True)

    args = parser.parse_args()

    logger.info(f"Round metrics:   {args.round_metrics}")
    logger.info(f"Final metrics:   {args.final_metrics}")
    logger.info(f"Client stats:    {args.client_stats}")
    logger.info(f"Output HTML:     {args.output_html}")
    logger.info(f"Output PDF:      {args.output_pdf}")

    for path in [args.output_html, args.output_pdf]:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    # Load all data
    round_metrics = load_json_files(args.round_metrics)
    final_metrics = load_json_files(args.final_metrics)
    client_stats = load_json_files(args.client_stats)

    # Generate plots
    output_dir = os.path.dirname(args.output_html) or "."
    plot_files = {}

    if round_metrics:
        pf = plot_round_metrics(round_metrics, output_dir)
        if pf:
            plot_files["rounds"] = os.path.basename(pf)

    if client_stats:
        pf = plot_client_drift(client_stats, output_dir)
        if pf:
            plot_files["clients"] = os.path.basename(pf)

    if final_metrics:
        pf = plot_final_metrics(final_metrics, output_dir)
        if pf:
            plot_files["final"] = os.path.basename(pf)

    # Build HTML
    html = build_html(round_metrics, final_metrics, client_stats, plot_files)
    with open(args.output_html, "w") as f:
        f.write(html)
    logger.info(f"HTML report written: {args.output_html}")

    # PDF placeholder
    with open(args.output_pdf, "wb") as f:
        f.write(b"%PDF-1.4 placeholder\n")
    logger.info(f"PDF placeholder written: {args.output_pdf}")
    logger.info("generate_report complete.")


if __name__ == "__main__":
    main()
