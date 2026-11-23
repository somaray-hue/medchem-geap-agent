"""Tests for Category 5: Infrastructure & CI/CD (Golden Eval Suite, IaC, Secret Management, API)."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from eval.run_eval import run_deterministic_golden_suite
from medchem_agent.config.secrets import SecretNotFoundError, SecretProvider
from medchem_agent.config.settings import AppSettings
from medchem_agent.main import fastapi_app

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_golden_eval_suite_passes_completely() -> None:
    """Verify ADK EvalSet schema and all deterministic golden regression checks pass."""
    summary = run_deterministic_golden_suite()
    assert summary["all_passed"] is True
    assert summary["passed_checks"] == summary["total_checks"]
    assert summary["total_checks"] >= 5


def test_secret_provider_uses_secret_manager_with_ttl_cache_and_no_hardcoded_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify GCP Secret Manager injection, caching, env fallback, and zero hardcoded secrets."""
    calls: list[str] = []

    class _MockSecretClient:
        def access_secret_version(self, *, request: dict[str, str]) -> SimpleNamespace:
            calls.append(request["name"])
            return SimpleNamespace(payload=SimpleNamespace(data=b"sm-injected-token-value"))

    cfg = AppSettings(
        gcp_project_id="demo-pharma-proj",
        enable_secret_manager=True,
    )
    provider = SecretProvider(settings=cfg, ttl_seconds=60, client=_MockSecretClient())

    val_1 = provider.get_secret("eln-api-token", required=True)
    val_2 = provider.get_secret("eln-api-token", required=True)
    assert val_1 == "sm-injected-token-value"
    assert val_2 == "sm-injected-token-value"
    assert len(calls) == 1  # Proven TTL cache hit

    # Missing required secret fails cleanly without crashing silently
    cfg_local = AppSettings(enable_secret_manager=False)
    local_provider = SecretProvider(settings=cfg_local)
    monkeypatch.delenv("NONEXISTENT_TEST_SECRET", raising=False)
    with pytest.raises(SecretNotFoundError):
        local_provider.get_secret("missing-secret", env_var="NONEXISTENT_TEST_SECRET", required=True)

    # Static check: no real API keys committed in medchem_agent/
    secret_pattern = re.compile(r"(AIza[0-9A-Za-z_\-]{30,}|sk-[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,})")
    for py_file in (REPO_ROOT / "medchem_agent").rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert not secret_pattern.search(content), f"Hardcoded credential found in {py_file}"


def test_iac_terraform_and_cicd_artifacts_exist() -> None:
    """Verify Terraform IaC, Cloud Build, GitHub Actions CI, Dockerfile, and GEAP deploy script exist."""
    expected_files = [
        "terraform/versions.tf",
        "terraform/variables.tf",
        "terraform/main.tf",
        "terraform/iam.tf",
        "terraform/secrets.tf",
        "terraform/observability.tf",
        "terraform/outputs.tf",
        "Dockerfile",
        "cloudbuild.yaml",
        ".github/workflows/ci.yml",
        "deploy/deploy_geap.sh",
        "eval/medchem_golden.evalset.json",
        "eval/test_config.json",
    ]
    for rel in expected_files:
        assert (REPO_ROOT / rel).exists(), f"Missing infrastructure artifact: {rel}"

    tf_main = (REPO_ROOT / "terraform/main.tf").read_text(encoding="utf-8")
    assert "google_vertex_ai_reasoning_engine" in tf_main
    assert "google_cloud_run_v2_service" in tf_main


@pytest.mark.asyncio
async def test_fastapi_health_and_schemas_endpoints() -> None:
    """Verify production FastAPI /health and /schemas endpoints respond cleanly."""
    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "healthy"

        schemas = await client.get("/schemas")
        assert schemas.status_code == 200
        assert "QsarModelOutput" in schemas.json()
