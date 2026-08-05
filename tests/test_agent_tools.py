"""CPU tests for the agent's tools and its response parsing."""

from __future__ import annotations

import json

import pytest

from agent.loop import extract_json, run_tool
from agent.tools.calculator import CalculatorError, calculate
from agent.tools.doc_search import document_count, search
from agent.tools.mock_api import call as mock_api_call

# -- calculator ------------------------------------------------------------


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("2 + 2", "4"),
        ("1240 * 12", "14880"),
        ("3150 * 3", "9450"),
        ("875 * 30 * 0.88", "23100"),
        ("(1690 * 12) + 45", "20325"),
        ("100 / 8", "12.5"),
        ("-5 + 3", "-2"),
        ("2 ** 10", "1024"),
        ("abs(-7)", "7"),
        ("round(3.14159, 2)", "3.14"),
        ("max(3, 9)", "9"),
        ("sqrt(144)", "12"),
    ],
)
def test_calculator_evaluates_arithmetic(expression, expected):
    assert calculate(expression) == expected


def test_integral_results_print_without_a_decimal_point():
    """The scorer matches strings; '9450.0' would fail against '9450'."""
    assert calculate("18900 / 2") == "9450"
    assert "." not in calculate("4800 * 3")


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "open('/etc/passwd').read()",
        "().__class__.__bases__",
        "[x for x in range(10)]",
        "lambda: 1",
        "exec('1')",
        "print(1)",
        "a = 1",
        "1 if True else 2",
        "{'a': 1}",
    ],
)
def test_calculator_refuses_anything_that_is_not_arithmetic(expression):
    with pytest.raises(CalculatorError):
        calculate(expression)


def test_calculator_rejects_unknown_names_and_functions():
    with pytest.raises(CalculatorError, match="unknown name"):
        calculate("total * 2")
    with pytest.raises(CalculatorError, match="whitelisted"):
        calculate("eval('1')")


def test_calculator_refuses_division_by_zero_and_huge_exponents():
    with pytest.raises(CalculatorError, match="division by zero"):
        calculate("1 / 0")
    with pytest.raises(CalculatorError, match="too large"):
        calculate("2 ** 100000")


def test_calculator_rejects_empty_and_unparseable_input():
    with pytest.raises(CalculatorError, match="empty"):
        calculate("   ")
    with pytest.raises(CalculatorError, match="could not parse"):
        calculate("2 +")


def test_calculator_errors_are_returned_to_the_model_not_raised():
    """A bad expression is a recoverable observation, not a run-ending failure."""
    observation = run_tool("calculator", "1 / 0")
    assert observation.startswith("Calculator error:")


# -- doc search ------------------------------------------------------------


def test_corpus_is_present():
    assert document_count() == 10


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Kestrel K3 price", "1,690 euro"),
        ("Vantage V1 cost per unit", "3,150 euro"),
        ("Torvald T4 price", "875 euro"),
        ("Gold support response time", "one business hour"),
        ("warranty length hardware", "24-month warranty"),
        ("restocking fee opened hardware", "15 percent"),
        ("express shipping European Union cost", "130 euro"),
        ("volume discount 25 units", "12 percent"),
        ("recalibration cost per unit", "210 euro"),
        ("chief executive", "Marit Osland"),
    ],
)
def test_search_surfaces_the_document_holding_the_fact(query, expected):
    assert expected in search(query)


def test_search_returns_at_most_three_documents():
    assert search("euro price cost").count("] ") <= 3


def test_search_handles_an_empty_or_unmatched_query():
    assert "No query given" in search("")
    assert "No documents matched" in search("zzzz qqqq xxxx")


# -- mock api --------------------------------------------------------------


def test_get_order_returns_the_record():
    payload = json.loads(mock_api_call("get_order(A-1001)"))
    assert payload["customer"] == "Nordkapp Marine"
    assert payload["items"][0] == {"product": "Kestrel K3", "quantity": 12}


def test_get_order_is_case_and_quote_tolerant():
    for form in ("get_order(a-1001)", "get_order('A-1001')", "get_order A-1001"):
        assert json.loads(mock_api_call(form))["order_id"] == "A-1001"


def test_get_weather_returns_the_record():
    payload = json.loads(mock_api_call("get_weather(Tromso)"))
    assert payload["temperature_c"] == -6
    assert payload["conditions"] == "snow"


def test_unknown_ids_return_an_error_listing_valid_options():
    order = json.loads(mock_api_call("get_order(Z-9999)"))
    assert order["error"] == "order not found"
    assert "A-1001" in order["known_orders"]

    weather = json.loads(mock_api_call("get_weather(Atlantis)"))
    assert weather["error"] == "city not found"


def test_unknown_function_is_reported_with_usage():
    payload = json.loads(mock_api_call("delete_everything()"))
    assert payload["error"] == "unknown function"
    assert any("get_order" in u for u in payload["usage"])


# -- action parsing --------------------------------------------------------


def test_parses_a_clean_action():
    action = extract_json('{"thought": "look it up", "action": "doc_search", '
                          '"action_input": "K3 price"}')
    assert action == {
        "thought": "look it up",
        "action": "doc_search",
        "action_input": "K3 price",
    }


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"thought": "t", "action": "final", "action_input": "42"}\n```',
        '```\n{"thought": "t", "action": "final", "action_input": "42"}\n```',
        'Sure! {"thought": "t", "action": "final", "action_input": "42"} Hope that helps.',
        '  {"thought": "t", "action": "final", "action_input": "42"}  ',
    ],
)
def test_parses_around_the_formatting_habits_of_small_models(text):
    """Tolerating wrappers avoids scoring formatting as a task failure."""
    assert extract_json(text)["action_input"] == "42"


def test_numeric_action_input_is_coerced_to_a_string():
    assert extract_json('{"thought": "t", "action": "final", "action_input": 42}')[
        "action_input"
    ] == "42"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "I think the answer is 42.",
        '{"thought": "t", "action": "sudo", "action_input": "x"}',
        '{"thought": "t", "action_input": "x"}',
        '{"thought": "t", "action": "final"',
        '["final", "42"]',
        '{"thought": "t", "action": "final", "action_input": {"a": 1}}',
    ],
)
def test_rejects_responses_that_are_not_a_valid_action(text):
    with pytest.raises(ValueError):
        extract_json(text)


def test_rejection_messages_name_the_problem():
    with pytest.raises(ValueError, match="action must be one of"):
        extract_json('{"thought": "t", "action": "browse", "action_input": "x"}')
