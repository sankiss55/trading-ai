"""Static (mypy-only) proof that the Alpaca adapters satisfy their ports (sec. 8.5).

Nothing here runs: the assignments are type-checked by ``mypy adapters/alpaca``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient

    from adapters.alpaca.broker import AlpacaBroker, TradingApiClient
    from adapters.alpaca.calendar import AlpacaCalendar, CalendarClient
    from adapters.alpaca.market_data import AlpacaMarketData, StockBarsClient
    from domain.ports import IBroker, IMarketCalendar, IMarketData

    def _check_conformance(
        market_data: AlpacaMarketData,
        calendar: AlpacaCalendar,
        broker: AlpacaBroker,
        bars_client: StockHistoricalDataClient,
        trading_client: TradingClient,
    ) -> None:
        _md: IMarketData = market_data
        _cal: IMarketCalendar = calendar
        _broker: IBroker = broker
        _bars: StockBarsClient = bars_client
        _trading: CalendarClient = trading_client
        _orders: TradingApiClient = trading_client
