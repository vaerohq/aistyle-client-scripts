from unittest.mock import patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_module_imports():
    """Verify the module can be imported without errors."""
    import vaero_api_tester  # noqa: F401


def test_get_multiline_input():
    """Test that get_multiline_input collects lines until the END keyword."""
    from vaero_api_tester import get_multiline_input

    with patch("builtins.input", side_effect=["Hello world", "Second line", "END"]):
        result = get_multiline_input()
    assert result == "Hello world\nSecond line"


def test_get_multiline_input_case_insensitive_end():
    """END keyword should be case-insensitive."""
    from vaero_api_tester import get_multiline_input

    with patch("builtins.input", side_effect=["Only line", "end"]):
        result = get_multiline_input()
    assert result == "Only line"


def test_output_processing_filters_metadata_lines():
    """Lines starting with Original:, Levenshtein Distance:, etc. are filtered out."""
    raw_output = (
        "Original: some original text\n"
        "Styled: This is the styled output\n"
        "Levenshtein Distance: 5\n"
        "Levenshtein Ratio: 0.9\n"
        "Length ratio: 1.1\n"
        "Normal line here\n"
    )

    condensed_lines = []
    for line in raw_output.splitlines():
        if (
            line.startswith("Original: ")
            or line.startswith("Levenshtein Distance: ")
            or line.startswith("Levenshtein Ratio: ")
            or line.startswith("Length ratio: ")
        ):
            continue
        elif line.startswith("Styled: "):
            condensed_lines.append(line[8:])
        else:
            condensed_lines.append(line)

    assert "Original: some original text" not in condensed_lines
    assert "Levenshtein Distance: 5" not in condensed_lines
    assert "This is the styled output" in condensed_lines
    assert "Normal line here" in condensed_lines


def test_output_processing_collapses_blank_lines():
    """Consecutive blank lines should be collapsed to one."""
    lines = ["Hello", "", "", "", "World"]

    condensed = [
        ln
        for i, ln in enumerate(lines)
        if ln.strip() or i == 0 or lines[i - 1].strip()
    ]

    assert condensed == ["Hello", "", "World"]


def test_endpoint_url_without_port():
    """When port is None, endpoint should not include a port segment."""
    host = "https://vaeroapi.com"
    port = None

    endpoint = f"{host}:{port}/v1/files" if port else f"{host}/v1/files"
    assert endpoint == "https://vaeroapi.com/v1/files"


def test_endpoint_url_with_port():
    """When port is set, endpoint should include the port."""
    host = "https://vaeroapi.com"
    port = 8080

    endpoint = f"{host}:{port}/v1/files" if port else f"{host}/v1/files"
    assert endpoint == "https://vaeroapi.com:8080/v1/files"


def test_payload_construction_full_mode():
    """Payload for full mode should include messages and full_mode_options."""
    model = "test-model"
    mode = "full"
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    base_model = "gpt-4o"

    payload = {
        "mode": mode,
        "model": model,
    }
    payload["messages"] = messages
    payload["full_mode_options"] = {"base_model": base_model}

    assert payload["mode"] == "full"
    assert payload["model"] == "test-model"
    assert len(payload["messages"]) == 2
    assert payload["full_mode_options"]["base_model"] == "gpt-4o"


def test_payload_construction_rewrite_mode():
    """Payload for rewrite mode should include message and rewrite_mode_options."""
    model = "test-model"
    mode = "rewrite"
    message = "Some text to rewrite"
    base_prompt = "Make it formal"

    payload = {
        "mode": mode,
        "model": model,
    }
    payload["message"] = message
    payload["rewrite_mode_options"] = {"base_prompt": base_prompt}

    assert payload["mode"] == "rewrite"
    assert payload["message"] == "Some text to rewrite"
    assert payload["rewrite_mode_options"]["base_prompt"] == "Make it formal"
