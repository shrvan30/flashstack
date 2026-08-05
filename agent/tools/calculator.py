"""Arithmetic evaluation over an AST whitelist.

`eval` is not used and must not be: the agent's input is model-generated text,
so anything that reaches a real interpreter is an arbitrary-code-execution path
with a language model on the other end of it. Parsing to an AST and walking only
the node types arithmetic needs makes the unsupported cases structurally
unreachable rather than merely filtered.
"""

from __future__ import annotations

import ast
import math
import operator

BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
}

CONSTANTS = {"pi": math.pi, "e": math.e}

# An exponent large enough to hang the process is a denial of service even
# without any code execution, so the one unbounded primitive is bounded.
MAX_EXPONENT = 1000


class CalculatorError(ValueError):
    """Raised when an expression is unsupported or cannot be evaluated."""


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculatorError(f"unsupported constant: {node.value!r}")
        return node.value

    if isinstance(node, ast.Name):
        if node.id not in CONSTANTS:
            raise CalculatorError(f"unknown name: {node.id!r}")
        return CONSTANTS[node.id]

    if isinstance(node, ast.BinOp):
        handler = BINARY_OPERATORS.get(type(node.op))
        if handler is None:
            raise CalculatorError(f"unsupported operator: {type(node.op).__name__}")
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError(f"exponent {right} is too large")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise CalculatorError("division by zero")
        return handler(left, right)

    if isinstance(node, ast.UnaryOp):
        handler = UNARY_OPERATORS.get(type(node.op))
        if handler is None:
            raise CalculatorError(f"unsupported unary operator: {type(node.op).__name__}")
        return handler(_evaluate(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            raise CalculatorError("only whitelisted functions may be called")
        if node.keywords:
            raise CalculatorError("keyword arguments are not supported")
        return FUNCTIONS[node.func.id](*(_evaluate(arg) for arg in node.args))

    if isinstance(node, (ast.List, ast.Tuple)):
        return [_evaluate(element) for element in node.elts]

    raise CalculatorError(f"unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression, returning a display string.

    Integral results print without a trailing `.0`, because the task suite scores
    string matches and "42.0" failing to match "42" would be a scorer artefact
    rather than a model error.
    """
    text = (expression or "").strip()
    if not text:
        raise CalculatorError("empty expression")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"could not parse {text!r}: {exc.msg}") from exc

    result = _evaluate(tree)

    if isinstance(result, list):
        raise CalculatorError("expression produced a list, not a number")
    if isinstance(result, float):
        if math.isnan(result) or math.isinf(result):
            raise CalculatorError(f"result is not finite: {result}")
        if result == int(result) and abs(result) < 1e15:
            return str(int(result))
        return f"{round(result, 6):g}"
    return str(result)
