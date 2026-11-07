from pydantic import BaseModel

from text_signals.claude_code import clean_env, command, parse


class Out(BaseModel):
    n: int


def test_clean_env_drops_api_key_and_parent_session() -> None:
    env = clean_env(
        {"PATH": "/bin", "ANTHROPIC_API_KEY": "k", "CLAUDE_CODE_SESSION_ID": "s", "HOME": "/h"}
    )
    assert env == {"PATH": "/bin", "HOME": "/h"}


def test_command_disables_tools_and_pins_model_and_effort() -> None:
    cmd = command("claude-opus-5-5", "high", "SYSTEM", {"type": "object"})
    assert cmd[:2] == ["claude", "-p"]
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--model") + 1] == "claude-opus-5-5"
    assert cmd[cmd.index("--effort") + 1] == "high"
    assert cmd[cmd.index("--system-prompt") + 1] == "SYSTEM"
    assert "--strict-mcp-config" in cmd and "--no-session-persistence" in cmd


def test_parse_validates_the_structured_output() -> None:
    assert parse({"structured_output": {"n": 7}}, Out) == ({"n": 7}, None)
    out, err = parse({"structured_output": {"n": "x"}}, Out)
    assert out is None and err is not None
    assert parse({"is_error": True, "subtype": "error_max_turns"}, Out)[0] is None
    assert parse({}, Out) == (None, "no structured output")
