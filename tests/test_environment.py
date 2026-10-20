from __future__ import annotations

from app.environment import load_local_environment


def test_local_environment_precedence_is_deterministic(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "BASE_ONLY=base\n"
        "FILE_SHARED=base\n"
        "PROCESS_SHARED=base\n"
        "EMPTY_OVERRIDE=base\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.local").write_text(
        "LOCAL_ONLY=local\n"
        "FILE_SHARED=local\n"
        "PROCESS_SHARED=local\n"
        "EMPTY_OVERRIDE=\n",
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    environ = {
        "PROCESS_SHARED": "process",
        "PROCESS_ONLY": "process-only",
    }

    loaded = load_local_environment(root=tmp_path, environ=environ)

    assert loaded == (tmp_path / ".env.local", tmp_path / ".env")
    assert environ == {
        "PROCESS_SHARED": "process",
        "PROCESS_ONLY": "process-only",
        "LOCAL_ONLY": "local",
        "FILE_SHARED": "local",
        "EMPTY_OVERRIDE": "",
        "BASE_ONLY": "base",
    }


def test_local_environment_loading_can_be_disabled(tmp_path):
    (tmp_path / ".env").write_text("SHOULD_NOT_LOAD=value\n", encoding="utf-8")
    environ = {"MEMORY_LOAD_LOCAL_ENV": "false"}

    loaded = load_local_environment(root=tmp_path, environ=environ)

    assert loaded == ()
    assert environ == {"MEMORY_LOAD_LOCAL_ENV": "false"}


def test_local_environment_skips_bare_and_malformed_names(tmp_path):
    (tmp_path / ".env").write_text(
        "BARE\n"
        "VALID_NAME=value\n"
        "DASH-NAME=ignored\n"
        "LITERAL=${UNRELATED_SHELL_VALUE}\n",
        encoding="utf-8",
    )
    environ = {"UNRELATED_SHELL_VALUE": "ambient"}

    load_local_environment(root=tmp_path, environ=environ)

    assert "BARE" not in environ
    assert "DASH-NAME" not in environ
    assert environ["VALID_NAME"] == "value"
    assert environ["LITERAL"] == "${UNRELATED_SHELL_VALUE}"
