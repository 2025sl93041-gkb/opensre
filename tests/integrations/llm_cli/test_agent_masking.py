"""Masking across the CLI agent's JSON and plain-text response boundaries."""

from __future__ import annotations

import json
import re
from unittest.mock import MagicMock, patch

import pytest

from config.constants.masking import (
    OPENSRE_MASK_ENABLED_ENV,
    OPENSRE_MASK_EXTRA_REGEX_ENV,
    OPENSRE_MASK_KINDS_ENV,
)
from core.llm.transports.sdk.agent_clients import CLIBackedAgentClient
from integrations.llm_cli.base import CLIInvocation, CLIProbe


@pytest.mark.parametrize("envelope", ["input", "arguments", "final"])
def test_agent_restores_masked_values_after_parsing(
    monkeypatch: pytest.MonkeyPatch, envelope: str
) -> None:
    value = 'ab"cd\\path'
    placeholder = "<QUOTED_VAL_0>"
    monkeypatch.setenv(OPENSRE_MASK_ENABLED_ENV, "true")
    monkeypatch.setenv(OPENSRE_MASK_KINDS_ENV, "ip_address")
    monkeypatch.setenv(OPENSRE_MASK_EXTRA_REGEX_ENV, json.dumps({"QUOTED_VAL": re.escape(value)}))
    arguments = {"values": [placeholder]}
    if envelope == "final":
        stdout = f"Found {placeholder}"
    else:
        payload = arguments if envelope == "input" else json.dumps(arguments)
        stdout = json.dumps({"tool_calls": [{"id": "c1", "name": "lookup", envelope: payload}]})

    captured: list[str] = []

    def build(*, prompt: str, **_kwargs: object) -> CLIInvocation:
        captured.append(prompt)
        return CLIInvocation(
            argv=("mock-cli",), stdin=prompt, cwd="/tmp", env=None, timeout_sec=30.0
        )

    adapter = MagicMock()
    adapter.name = "mock-cli"
    adapter.detect.return_value = CLIProbe(
        installed=True, bin_path="/usr/bin/mock-cli", version=None, logged_in=True, detail="ok"
    )
    adapter.build.side_effect = build
    adapter.parse.side_effect = lambda *, stdout, **_kwargs: stdout
    with (
        patch("infrastructure.safety.guardrails.evaluator.get_guardrail_evaluator") as guardrails,
        patch("integrations.llm_cli.runner.subprocess.run") as run,
    ):
        guardrails.return_value.is_active = False
        run.return_value = MagicMock(returncode=0, stdout=stdout, stderr="")
        client = CLIBackedAgentClient(adapter)
        result = client.invoke([{"role": "user", "content": f"Look up {value}"}])

    assert len(captured) == 1
    assert value not in captured[0]
    assert placeholder in captured[0]
    if envelope == "final":
        assert result.tool_calls == []
        assert result.content == f"Found {value}"
    else:
        assert result.content == ""
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "lookup"
        assert result.tool_calls[0].input == {"values": [value]}
