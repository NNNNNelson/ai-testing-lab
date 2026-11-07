import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "data" / "corpus.txt"


def pytest_collection_modifyitems(config, items):
    """Real-model tests never silently pass: they skip, loudly."""
    real = os.environ.get("ARK_API_KEY") and os.environ.get("EVAL_PROVIDER") != "mock"
    if real:
        return
    skip_real = pytest.mark.skip(reason="no real model configured (set ARK_API_KEY); mock scores are not evidence")
    for item in items:
        if "requires_real_model" in item.keywords:
            item.add_marker(skip_real)


@pytest.fixture(scope="session")
def corpus_path():
    if not CORPUS.exists():
        pytest.skip(f"corpus not built: run python scripts/build_corpus.py ({CORPUS})")
    return CORPUS


@pytest.fixture(scope="session")
def provider():
    from evals import llm
    return llm.get_provider(allow_mock=True)


@pytest.fixture(scope="session")
def golden_rag():
    import yaml
    path = ROOT / "evals" / "golden_rag.yaml"
    if not path.exists():
        pytest.skip("golden_rag.yaml missing")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def golden_agent():
    import yaml
    path = ROOT / "evals" / "golden_agent.yaml"
    if not path.exists():
        pytest.skip("golden_agent.yaml missing")
    return yaml.safe_load(path.read_text(encoding="utf-8"))
