import importlib
import os
import py_compile
from pathlib import Path

import pytest

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

MODULES = [
    "intent_agent",
    "intent_agent.config",
    "intent_agent.llm",
    "intent_agent.agent.classify",
    "intent_agent.memory.store",
    "intent_agent.optimizer.opro",
    "intent_agent.evaluation.evaluate",
    "intent_agent.training.train",
    "intent_agent.tools",
    "intent_agent.api.app",
]

UI_SCRIPT = Path(__file__).resolve().parents[1] / "src" / "intent_agent" / "ui" / "streamlit_app.py"


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)


def test_streamlit_app_compiles():
    # Importing the UI would execute the whole Streamlit page, so only compile it.
    py_compile.compile(str(UI_SCRIPT), doraise=True)


def test_paths_point_at_project_files():
    from intent_agent import config

    assert config.TRAIN_FILE.is_file()
    assert config.TEST_FILE.is_file()
    assert config.DEMO_FILE.is_file()
    assert config.MEMORY_DB_DIR.parent == config.STORAGE_DIR
    assert config.RESULTS_FILE.parent == config.STORAGE_DIR
