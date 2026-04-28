from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

from ..gds.centrality import CentralityService


def run_centrality_batch(
    metrics: list[str] | None = None,
    recreate_projection: bool = False,
    betweenness_sampling_size: int = 10000,
) -> dict[str, Any]:
    selected_metrics = metrics or ["pagerank", "degree"]
    run_id = f"centrality-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    service = CentralityService()
    results: dict[str, Any] = {"run_id": run_id, "metrics": selected_metrics}
    try:
        results["projection"] = service.ensure_projection(recreate=recreate_projection)
        if "pagerank" in selected_metrics:
            results["pagerank"] = service.write_pagerank()
        if "degree" in selected_metrics:
            results["degree"] = service.write_degree()
        if "betweenness" in selected_metrics:
            results["betweenness"] = service.write_betweenness(sampling_size=betweenness_sampling_size)
        results["promoted"] = service.promote_next_properties()
        results["status"] = service.write_status("success", run_id=run_id, metrics=selected_metrics)
        return results
    except Exception as exc:
        service.write_status("failed", run_id=run_id, metrics=selected_metrics, error=str(exc))
        raise
    finally:
        service.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run centrality batch job for persona graph")
    parser.add_argument(
        "--metrics",
        default="pagerank,degree",
        help="Comma-separated metrics: pagerank,degree,betweenness",
    )
    parser.add_argument("--recreate-projection", action="store_true")
    parser.add_argument("--betweenness-sampling-size", type=int, default=10000)
    args = parser.parse_args()
    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    result = run_centrality_batch(
        metrics=metrics,
        recreate_projection=args.recreate_projection,
        betweenness_sampling_size=args.betweenness_sampling_size,
    )
    print(result)


if __name__ == "__main__":
    main()
