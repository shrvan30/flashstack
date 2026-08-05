"""Canned deterministic JSON for `get_order` and `get_weather`.

Deterministic on purpose: the harness compares three backends on an identical
task suite, so any tool that returned live or randomised data would inject
variance that has nothing to do with the thing being measured.
"""

from __future__ import annotations

import json

ORDERS: dict[str, dict] = {
    "A-1001": {
        "order_id": "A-1001",
        "customer": "Nordkapp Marine",
        "status": "shipped",
        "placed": "2026-01-14",
        "items": [{"product": "Kestrel K3", "quantity": 12}],
        "shipping": "standard",
        "destination_region": "EU",
    },
    "A-1002": {
        "order_id": "A-1002",
        "customer": "Baltic Drilling",
        "status": "processing",
        "placed": "2026-02-03",
        "items": [{"product": "Torvald T4", "quantity": 30}],
        "shipping": "express",
        "destination_region": "EU",
    },
    "A-1003": {
        "order_id": "A-1003",
        "customer": "Cape Breton Energy",
        "status": "delivered",
        "placed": "2025-11-20",
        "items": [
            {"product": "Vantage V1", "quantity": 2},
            {"product": "Kestrel K2", "quantity": 4},
        ],
        "shipping": "standard",
        "destination_region": "North America",
    },
    "A-1004": {
        "order_id": "A-1004",
        "customer": "Gdansk Shipyard",
        "status": "cancelled",
        "placed": "2026-01-02",
        "items": [{"product": "Kestrel K2", "quantity": 60}],
        "shipping": "standard",
        "destination_region": "EU",
    },
}

WEATHER: dict[str, dict] = {
    "tromso": {"city": "Tromso", "temperature_c": -6, "conditions": "snow", "wind_kph": 22},
    "gdansk": {"city": "Gdansk", "temperature_c": 3, "conditions": "overcast", "wind_kph": 14},
    "halifax": {"city": "Halifax", "temperature_c": -1, "conditions": "clear", "wind_kph": 9},
    "oslo": {"city": "Oslo", "temperature_c": -2, "conditions": "light snow", "wind_kph": 11},
}


def get_order(order_id: str) -> str:
    key = (order_id or "").strip().upper()
    if key not in ORDERS:
        return json.dumps(
            {"error": "order not found", "order_id": key, "known_orders": sorted(ORDERS)}
        )
    return json.dumps(ORDERS[key])


def get_weather(city: str) -> str:
    key = (city or "").strip().lower()
    if key not in WEATHER:
        return json.dumps(
            {"error": "city not found", "city": city, "known_cities": sorted(WEATHER)}
        )
    return json.dumps(WEATHER[key])


def call(action_input: str) -> str:
    """Dispatch `get_order(A-1001)` / `get_weather(Tromso)` written as a single string.

    The agent protocol gives every tool one string argument, so the function name
    and its argument arrive together and are split here rather than in the loop.
    """
    text = (action_input or "").strip()
    lowered = text.lower()

    for name, handler in (("get_order", get_order), ("get_weather", get_weather)):
        if lowered.startswith(name):
            argument = text[len(name) :].strip()
            if argument.startswith("(") and argument.endswith(")"):
                argument = argument[1:-1]
            return handler(argument.strip().strip("'\""))

    return json.dumps(
        {
            "error": "unknown function",
            "received": text,
            "usage": ["get_order(<order_id>)", "get_weather(<city>)"],
        }
    )
