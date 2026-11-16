import json
from types import SimpleNamespace

import pytest

from intent_agent.memory import store
from intent_agent.training import train


@pytest.mark.parametrize(
    ("debug", "expected_output"),
    [(False, ""), (True, "Round 1: score 0.75\n")],
)
def test_training_progress_respects_debug_flag(
    debug, expected_output, monkeypatch, tmp_path, capsys
):
    train_file = tmp_path / "train.json"
    test_file = tmp_path / "test.json"
    train_file.write_text(json.dumps([]))
    test_file.write_text(json.dumps([]))

    monkeypatch.setattr(train, "get_settings", lambda: SimpleNamespace(debug=debug))
    monkeypatch.setattr(train, "TRAIN_FILE", train_file)
    monkeypatch.setattr(train, "TEST_FILE", test_file)
    monkeypatch.setattr(train, "RESULTS_FILE", tmp_path / "results.json")
    monkeypatch.setattr(train, "evaluate", lambda *_args, **_kwargs: (0.75, []))
    monkeypatch.setattr(train.memory, "all_lessons", lambda: [])
    monkeypatch.setattr(train.memory, "cleanup", lambda: None)
    monkeypatch.setattr(train, "new_prompt", lambda history, mistakes: history[-1][0])

    train.run_training(rounds=1)

    assert capsys.readouterr().out == expected_output


@pytest.mark.parametrize("debug", [False, True])
def test_memory_diagnostics_respect_debug_flag(debug, monkeypatch, capsys):
    class FakeCollection:
        def count(self):
            return 0

        def add(self, **_kwargs):
            pass

        def get(self):
            return {"documents": ["lesson"], "metadatas": [{"score": 0}]}

    monkeypatch.setattr(store, "get_settings", lambda: SimpleNamespace(debug=debug))
    monkeypatch.setattr(store, "col", FakeCollection())

    store.add("lesson")
    store.all_lessons()

    output = capsys.readouterr().out
    if debug:
        assert output == "saved: score=0  lesson\nscore=0  lesson\n"
    else:
        assert output == ""
