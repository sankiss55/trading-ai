"""Deterministic simulation adapters (sec. 8.6, 45.2, 49.5).

No wall clock, no network and no unseeded randomness: every adapter here is driven by
injected data and an injected clock.
"""

from adapters.simulation.historical_feed import HistoricalFeed, ScriptedFeed, load_bars_csv
from adapters.simulation.scripted_ai_filter import ScriptedAIFilter
from adapters.simulation.sim_clock import FixedClock, SimClock
from adapters.simulation.simulated_broker import SimulatedBroker
from adapters.simulation.static_calendar import StaticCalendar, build_regular_sessions

__all__ = [
    "FixedClock",
    "HistoricalFeed",
    "ScriptedAIFilter",
    "ScriptedFeed",
    "SimClock",
    "SimulatedBroker",
    "StaticCalendar",
    "build_regular_sessions",
    "load_bars_csv",
]
