"""Alpaca adapters (alpaca-py, the official SDK; never ``alpaca-trade-api``).

Implemented so far (declared early pull-forward of part of Phase 2, for the Phase 1
backtest on real data): historical bars (:class:`AlpacaMarketData`) and the trading
calendar (:class:`AlpacaCalendar`). Orders (``AlpacaBroker``), the live clock and the
real-time streams remain Phase 2/3.
"""

from adapters.alpaca._http import AlpacaCredentials, HttpTimeouts
from adapters.alpaca._retry import RequestPacer, RetryPolicy
from adapters.alpaca.calendar import AlpacaCalendar
from adapters.alpaca.market_data import AlpacaMarketData, BarAdjustment

__all__ = [
    "AlpacaCalendar",
    "AlpacaCredentials",
    "AlpacaMarketData",
    "BarAdjustment",
    "HttpTimeouts",
    "RequestPacer",
    "RetryPolicy",
]
