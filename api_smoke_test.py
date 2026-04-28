import json
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"


def request_json(method: str, path: str, payload: dict | None = None, query: dict | None = None) -> tuple[int, dict]:
    url = f"{BASE_URL}{path}"
    if query:
        url = f"{url}?{urlencode(query)}"
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=120) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except HTTPError as exc:
        body = exc.read().decode("utf-8")
        return exc.code, json.loads(body) if body else {}
    except URLError as exc:
        raise RuntimeError(f"Request failed for {method} {path}: {exc}") from exc


def assert_ok(name: str, status: int, condition: bool, detail: str) -> None:
    if status != 200 or not condition:
        raise AssertionError(f"{name} failed: status={status}, detail={detail}")
    print(f"PASS {name}: {detail}")


def main() -> None:
    status, stats = request_json("GET", "/api/stats")
    assert_ok("stats overview", status, stats.get("total_personas") == 10000, f"total={stats.get('total_personas')}")

    status, hobby_stats = request_json("GET", "/api/stats/hobby", query={"limit": 3})
    assert_ok("stats hobby", status, len(hobby_stats.get("distribution", [])) > 0, f"items={len(hobby_stats.get('distribution', []))}")

    status, search = request_json("GET", "/api/search", query={"page_size": 2})
    results = search.get("results", [])
    assert_ok("search", status, len(results) >= 2, f"total={search.get('total_count')}, returned={len(results)}")
    uuid1 = results[0]["uuid"]
    uuid2 = results[1]["uuid"]

    status, profile = request_json("GET", f"/api/persona/{uuid1}")
    assert_ok("persona profile", status, profile.get("uuid") == uuid1, f"uuid={profile.get('uuid')}")

    status, subgraph = request_json("GET", f"/api/graph/subgraph/{uuid1}", query={"depth": 1, "max_nodes": 10})
    assert_ok("subgraph", status, subgraph.get("node_count", 0) >= 1, f"nodes={subgraph.get('node_count')}, edges={subgraph.get('edge_count')}")

    status, similar = request_json("POST", f"/api/similar/{uuid1}", payload={"top_k": 3})
    assert_ok("similar", status, len(similar.get("similar_personas", [])) > 0, f"returned={len(similar.get('similar_personas', []))}")

    status, communities = request_json("GET", "/api/communities", query={"min_size": 10})
    assert_ok("communities", status, len(communities.get("communities", [])) > 0, f"returned={len(communities.get('communities', []))}")

    status, path = request_json("GET", f"/api/path/{uuid1}/{uuid2}")
    assert_ok("path", status, "path_found" in path, f"path_found={path.get('path_found')}")

    compare_payload = {
        "segment_a": {"label": "서울", "filters": {"province": "서울"}},
        "segment_b": {"label": "부산", "filters": {"province": "부산"}},
        "dimensions": ["hobby"],
        "top_k": 3,
    }
    status, compare = request_json("POST", "/api/compare/segments", payload=compare_payload)
    assert_ok("compare", status, "hobby" in compare.get("comparisons", {}), f"ai_analysis_len={len(compare.get('ai_analysis', ''))}")

    print("ALL_SMOKE_TESTS_PASSED")


if __name__ == "__main__":
    main()
