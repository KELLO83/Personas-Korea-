from __future__ import annotations

from types import SimpleNamespace

from experiments.hobby_recommender_ml.scripts import evaluate_ranker


def test_evaluate_default_cpu_threads_follow_laptop_policy(monkeypatch) -> None:
    monkeypatch.setattr(evaluate_ranker.os, "cpu_count", lambda: 22)
    monkeypatch.setattr(evaluate_ranker, "_query_system_memory_mb", lambda: (0, 0))
    monkeypatch.setattr(evaluate_ranker, "_query_gpu_memory_mb", lambda: (0, 0, 0))

    plan = evaluate_ranker._resolve_system_resource_plan(SimpleNamespace(cpu_thread_count=0))

    assert plan["default_cpu_threads"] == 18
    assert plan["cpu_threads"] == 18


def test_evaluate_requested_cpu_threads_are_respected(monkeypatch) -> None:
    monkeypatch.setattr(evaluate_ranker.os, "cpu_count", lambda: 22)
    monkeypatch.setattr(evaluate_ranker, "_query_system_memory_mb", lambda: (0, 0))
    monkeypatch.setattr(evaluate_ranker, "_query_gpu_memory_mb", lambda: (0, 0, 0))

    plan = evaluate_ranker._resolve_system_resource_plan(SimpleNamespace(cpu_thread_count=10))

    assert plan["cpu_threads"] == 10
