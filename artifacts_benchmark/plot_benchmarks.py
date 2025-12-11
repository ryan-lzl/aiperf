"""Plot output token throughput for four benchmark scenarios under artifacts_benchmark.
Outputs a two-panel bar chart image: artifacts_benchmark/output_token_throughput_<model>_<backend>_c<concurrency>.png
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


WORKLOAD_DESCRIPTIONS = {
    "A": "Workload A (prefill-heavy): input ~900 tokens, output ~160 tokens; useful to see prefill disaggregation effects.",
    "B": "Workload B (decode-heavy): input ~180 tokens, output ~600 tokens; stresses decode, batching, and router/speculation benefits.",
}


def load_throughput(base_dir: Path, rel_path: str):
    path = base_dir / rel_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open() as f:
        data = json.load(f)
    try:
        throughput = float(data["output_token_throughput"]["avg"])
        ttft_ms = float(data["time_to_first_token"]["avg"])
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not read throughput/TTFT from {path}") from exc
    error_summary = data.get("error_summary") or {}
    has_errors = False
    if isinstance(error_summary, dict):
        has_errors = any(error_summary.values())
    elif isinstance(error_summary, list):
        has_errors = len(error_summary) > 0
    else:
        has_errors = bool(error_summary)
    return throughput, ttft_ms, has_errors, path


def infer_backend_label(rel_path: str) -> str:
    """Infer backend name from the directory portion of the path."""
    dir_name = Path(rel_path).parts[0] if Path(rel_path).parts else rel_path
    tokens = dir_name.split("-")
    mapping = {
        "vllm": "vLLM",
        "trtllm": "TensorRT-LLM",
        "tensorrt": "TensorRT-LLM",
        "sglang": "SGLang",
    }
    for tok in tokens:
        key = tok.lower()
        if key in mapping:
            return mapping[key]
    return dir_name


def normalize_backend_flag(raw: str) -> tuple[str, str]:
    key = raw.strip().lower().replace("_", "-")
    mapping = {
        "vllm": ("vLLM", "vllm"),
        "trt": ("TensorRT-LLM", "trtllm"),
        "trt-llm": ("TensorRT-LLM", "trtllm"),
        "trtllm": ("TensorRT-LLM", "trtllm"),
        "tensorrt": ("TensorRT-LLM", "trtllm"),
        "sglang": ("SGLang", "sglang"),
        "sg-lang": ("SGLang", "sglang"),
    }
    if key in mapping:
        return mapping[key]
    raise ValueError(f"Unsupported backend '{raw}'. Expected one of: vllm, trtllm, sglang.")


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\\/]+", "-", text)
    text = re.sub(r"[^a-z0-9-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "value"


def model_to_path_token(model: str) -> str:
    token = model.strip()
    token = token.replace("/", "_")
    token = re.sub(r"\s+", "_", token)
    # Keep dots for model versioning (e.g., 0.6B) to match folder names.
    token = re.sub(r"[^A-Za-z0-9_.-]", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "model"


def find_disagg_dir(
    base_dir: Path, model_token: str, backend_token: str, workload_key: str, concurrency: int, require_balanced: bool
) -> str:
    """Find a single disagg directory with worker-balancing suffix (e.g., ...-3P-1D-24C), filtered by 2P-2D or non-2P-2D."""
    prefix = f"{model_token}-disagg-router-{backend_token}-workload-{workload_key}-"
    candidates = []
    pattern = re.compile(rf"^{re.escape(prefix)}\d+P-\d+D-{concurrency}C$")
    for entry in base_dir.iterdir():
        if entry.is_dir() and pattern.match(entry.name):
            profile = entry / "profile_export_aiperf.json"
            if not profile.is_file():
                continue
            prefill, decode = parse_worker_counts(entry.name)
            if prefill is None or decode is None:
                continue
            if require_balanced and (prefill, decode) != (2, 2):
                continue
            if not require_balanced and (prefill, decode) == (2, 2):
                continue
            candidates.append(entry)
    if not candidates:
        expectation = "2P-2D" if require_balanced else "non-2P-2D"
        raise FileNotFoundError(
            f"No disagg directory found for workload {workload_key} with prefix '{prefix}<n>P-<m>D-{concurrency}C' "
            f"and worker balance '{expectation}' containing profile_export_aiperf.json."
        )
    if len(candidates) > 1:
        names = ", ".join(e.name for e in candidates)
        expectation = "2P-2D" if require_balanced else "non-2P-2D"
        raise ValueError(f"Multiple disagg directories match {prefix}* with {expectation}: {names}.")
    return candidates[0].name


def parse_worker_counts(dir_name: str) -> tuple[int | None, int | None]:
    """Extract prefill/decode counts from a suffix like -3P-1D or -3P-1D-24C; returns (prefill, decode)."""
    match = re.search(r"-(\d+)P-(\d+)D(?:-\d+C)?$", dir_name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def main():
    parser = argparse.ArgumentParser(description="Plot throughput and TTFT for aggregation vs disaggregation+router runs.")
    parser.add_argument("--model", required=True, help="Model name to annotate in the chart title.")
    parser.add_argument(
        "--inference-backend",
        dest="inference_backend",
        required=True,
        help="Inference backend label to annotate in the chart title (e.g., vLLM, TensorRT-LLM).",
    )
    parser.add_argument(
        "--worker-balancing",
        action="store_true",
        help="Use only 2P-2D disagg folders; when omitted, use only non-2P-2D disagg folders.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=24,
        help="Concurrency used in the benchmark folder names (e.g., 24C).",
    )
    args = parser.parse_args()

    model_name = args.model
    try:
        backend_label, backend_path_token = normalize_backend_flag(args.inference_backend)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    concurrency = args.concurrency
    if concurrency <= 0:
        print("Error: --concurrency must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    model_path_token = model_to_path_token(model_name)
    base_dir = Path(__file__).resolve().parent

    try:
        disagg_dir_a = find_disagg_dir(
            base_dir, model_path_token, backend_path_token, "A", concurrency, require_balanced=args.worker_balancing
        )
        disagg_dir_b = find_disagg_dir(
            base_dir, model_path_token, backend_path_token, "B", concurrency, require_balanced=args.worker_balancing
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    scenarios = [
        ("Agg", "A", f"{model_path_token}-agg-{backend_path_token}-workload-A-{concurrency}C/profile_export_aiperf.json"),
        ("Disagg + Router", "A", f"{disagg_dir_a}/profile_export_aiperf.json"),
        ("Agg", "B", f"{model_path_token}-agg-{backend_path_token}-workload-B-{concurrency}C/profile_export_aiperf.json"),
        ("Disagg + Router", "B", f"{disagg_dir_b}/profile_export_aiperf.json"),
    ]
    results = []
    for label, workload_key, rel_path in scenarios:
        throughput, ttft_ms, has_errors, full_path = load_throughput(base_dir, rel_path)
        results.append((label, workload_key, throughput, ttft_ms, has_errors, full_path))

    inferred_backend = infer_backend_label(scenarios[0][2]) if scenarios else "Backend"
    if backend_label != inferred_backend:
        print(f"Note: using backend from flag ({backend_label}); inferred backend from paths is {inferred_backend}.")

    # Prepare plot data grouped by workload
    workload_to_entries = {"A": [], "B": []}
    for label, workload_key, throughput, ttft_ms, has_errors, full_path in results:
        workload_to_entries[workload_key].append((label, throughput, ttft_ms, has_errors, full_path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    fig.suptitle(
        f"{model_name} on {backend_label} with Dynamo (Concurrency {concurrency}) - Throughput & TTFT (Time to First Token)",
        fontsize=14,
    )

    throughput_color = "#4C78A8"
    ttft_color = "#F58518"

    def plot_workload(ax, workload_key: str, entries):
        labels = [e[0] for e in entries]
        vals = [e[1] for e in entries]
        ttfts = [e[2] for e in entries]
        ttft_max = max(ttfts) if ttfts else 0.0
        ttft_min = min(ttfts) if ttfts else 0.0
        positions = list(range(len(labels)))
        ax.plot(positions, vals, color=throughput_color, marker="o", linewidth=2, label="Throughput (tokens/sec)")
        ax.set_title(f"Workload {workload_key} ({'prefill-heavy' if workload_key=='A' else 'decode-heavy'})")
        ax.set_ylabel("Output Token Throughput (tokens/sec)", color=throughput_color)
        ax.tick_params(axis="y", labelcolor=throughput_color)
        ax.spines["left"].set_edgecolor(throughput_color)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        # Add headroom
        ymax = max(vals) * 1.25
        ax.set_ylim(0, ymax)
        for x, val in zip(positions, vals):
            ax.text(x, val, f"{val:.1f}", ha="center", va="bottom", fontsize=9, color=throughput_color)
        # TTFT line on secondary axis
        ax2 = ax.twinx()
        ax2.plot(positions, ttfts, color=ttft_color, marker="s", linestyle="-", linewidth=2, label="TTFT (ms)")
        ax2.set_ylabel("TTFT (ms)", color=ttft_color)
        ax2_ylim_top = ttft_max * 1.45 if ttft_max else 1.0
        ax2.set_ylim(0, ax2_ylim_top)
        ax2.tick_params(axis="y", labelsize=9, colors=ttft_color)
        ax2.spines["right"].set_edgecolor(ttft_color)
        # Show TTFT values above markers
        label_offset = 0.04 * ax2_ylim_top
        for x, tval in zip(positions, ttfts):
            ax2.text(x, tval + label_offset, f"{tval:.1f}", ha="center", va="bottom", fontsize=9, color=ttft_color)
        # Combined legend
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc="lower left", fontsize=9)
        # Annotate percentage delta vs agg if applicable
        base = None
        alt = None
        base_ttft = None
        alt_ttft = None
        for label, val, tval in zip(labels, vals, ttfts):
            if "Agg" in label:
                base = val
                base_ttft = tval
            if "Disagg" in label:
                alt = val
                alt_ttft = tval
        if base and alt:
            pct = (alt - base) / base * 100.0
            ax.text(
                0.5,
                0.9,
                f"Throughput Δ vs Agg: {pct:+.1f}%",
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                color="#2E6F40" if pct >= 0 else "#B22222",
                transform=ax.transAxes,
            )
        if base_ttft and alt_ttft:
            pct_ttft = (alt_ttft - base_ttft) / base_ttft * 100.0
            # Place TTFT delta under the yellow TTFT line with a small buffer
            ttft_range = ttft_max - ttft_min if ttft_max != ttft_min else ttft_max or 1.0
            y_text = ttft_min - 0.08 * ttft_range
            y_text = max(0.05 * ax2_ylim_top, y_text)
            ttft_delta_color = "#2E6F40" if pct_ttft < 0 else "#B22222"
            ax2.text(
                (positions[0] + positions[-1]) / 2 if positions else 0.5,
                y_text,
                f"TTFT Δ vs Agg: {pct_ttft:+.1f}% (lower is better)",
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                color=ttft_delta_color,
                transform=ax2.transData,
            )

    plot_workload(axes[0], "A", workload_to_entries["A"])
    plot_workload(axes[1], "B", workload_to_entries["B"])

    # Build footer notes
    disagg_a_counts = parse_worker_counts(disagg_dir_a)
    disagg_b_counts = parse_worker_counts(disagg_dir_b)

    def fmt_counts(counts: tuple[int | None, int | None]) -> str:
        prefill, decode = counts
        if prefill is None or decode is None:
            return "unknown prefill / decode"
        return f"{prefill} prefill / {decode} decode"

    note3_text = (
        f"3) Disagg + Router mode uses {fmt_counts(disagg_a_counts)} worker nodes for Workload A; "
        f"{fmt_counts(disagg_b_counts)} for Workload B. Each worker node requires 1 GPU."
    )

    # Add workload explanations below the plots, left-aligned with numbering
    fig.text(0.02, 0.11, f"1) {WORKLOAD_DESCRIPTIONS['A']}", ha="left", fontsize=9, wrap=True)
    fig.text(0.02, 0.08, f"2) {WORKLOAD_DESCRIPTIONS['B']}", ha="left", fontsize=9, wrap=True)
    fig.text(0.02, 0.05, note3_text, ha="left", fontsize=9, wrap=True)
    fig.text(0.02, 0.02, "4) Agg mode uses 4 worker nodes and round-robin for load balancing; each worker node requires 1 GPU.", ha="left", fontsize=9, wrap=True)

    fig.tight_layout(rect=(0, 0.18, 1, 0.93))

    slug_model = slugify(model_name)
    slug_backend = slugify(backend_label)
    slug_concurrency = f"c{concurrency}"
    out_path = base_dir / f"output_token_throughput_{slug_model}_{slug_backend}_{slug_concurrency}.png"
    fig.savefig(out_path, dpi=150)

    print("Output Token Throughput (tokens/sec) and TTFT (ms)")
    print("---------------------------------------------------")
    for label, workload_key, throughput, ttft_ms, has_errors, full_path in results:
        warn = "WARN" if has_errors else ""
        print(f"{label:18s} W{workload_key}  TPS:{throughput:8.2f}  TTFT:{ttft_ms:7.2f} {warn:4s} ({full_path})")
    print(f"\nSaved chart: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
