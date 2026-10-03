"""Risk gate: assembles the check-library context from the ports (sec. 19, 20).

The check library (``domain.guards.checks``) is pure; this module collects what it needs
at one instant and returns a frozen :class:`~domain.guards.checks.CheckContext`:

* broker state (source of truth, sec. 3.3): account, positions, open orders (``IBroker``);
* liquidity daily bars of ``universe.liquidity_feed`` (``IMarketData.get_daily_bars``),
  only sessions BEFORE the decision session (no lookahead), cached per symbol and day;
* the latest quote when the spread filter is enforced (``IMarketData.get_latest_quote``);
* runtime safety facts (``system_control`` + STOP file, circuit breaker, system mode,
  reconciliation, pending OWNER_DECISIONs, tradable assets, broker clock, feed status)
  from a :class:`GuardEnvironment` provided by the composition root.

The same gate serves the pre-AI step (``application.market_flow``) and, in Phase 5, the
Execution Guard, which builds its context from freshly refreshed state and calls
``run_execution_checks``. Missing sources fail closed: no environment means every
runtime check fails with its ``*_UNAVAILABLE`` code; an unreachable liquidity or quote
source (``RetryableError``) means ``LIQUIDITY_UNAVAILABLE`` / ``QUOTE_UNAVAILABLE``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import Annotated, Protocol, Self

from pydantic import Field

from domain.errors import RetryableError
from domain.guards.checks import (
    CheckContext,
    CheckParams,
    EntryTiming,
    MarketFacts,
    PortfolioFacts,
    RuntimeFacts,
    SpreadFilter,
)
from domain.models import (
    AccountState,
    Bar,
    BrokerOrder,
    DataFeed,
    DomainModel,
    Money,
    NonNegativeDecimal,
    Position,
    Price,
    ProposedTrade,
    Quote,
    Signal,
    Symbol,
)
from domain.ports import IBroker, IMarketData
from domain.risk.risk_engine import PendingEntry, PositionStop, RiskParams

__all__ = [
    "BrokerState",
    "GuardEnvironment",
    "GuardSettings",
    "RiskGate",
    "liquidity_fetch_range",
]


class GuardSettings(DomainModel):
    """Check settings that come from configuration (``None`` = OWNER_DECISION pending).

    The strategy- and risk-derived parameters (risk limits, cooldown, minimum minutes per
    bar, minimum take-profit distance) are added by the caller that owns them
    (:meth:`RiskGate.check_params`), so they are never configured twice.

    Attributes:
        spread_filter: ``ENFORCED`` where a quote source exists (live), ``NOT_APPLIED``
            where none does (backtest on stored bars).
        max_reconcile_age_seconds: Maximum age of the last clean reconciliation
            (``STATE_RECONCILED``).
    """

    whitelist: tuple[Symbol, ...] | None
    blacklist: tuple[Symbol, ...]
    min_price: Price | None
    max_price: Price | None
    min_avg_daily_volume: Annotated[int, Field(ge=0)] | None
    avg_volume_lookback_days: Annotated[int, Field(ge=1)] | None
    liquidity_feed: DataFeed | None
    max_spread_bps: NonNegativeDecimal | None
    spread_filter: SpreadFilter
    max_bar_age_seconds: Annotated[int, Field(ge=1)] | None
    max_reconcile_age_seconds: Annotated[int, Field(ge=1)] | None

    @classmethod
    def unconfigured(cls) -> Self:
        """No setting known: every check that needs one fails closed (``PARAM_PENDING``)."""
        return cls(
            whitelist=None,
            blacklist=(),
            min_price=None,
            max_price=None,
            min_avg_daily_volume=None,
            avg_volume_lookback_days=None,
            liquidity_feed=None,
            max_spread_bps=None,
            spread_filter=SpreadFilter.ENFORCED,
            max_bar_age_seconds=None,
            max_reconcile_age_seconds=None,
        )


class GuardEnvironment(Protocol):
    """Source of the runtime safety facts (composition root: DB, STOP file, breaker...)."""

    async def runtime_facts(self) -> RuntimeFacts:
        """Current control state, breaker, mode, reconciliation, config, assets, clock, feed."""
        ...


class BrokerState(DomainModel):
    """Account, positions and open orders read from the broker at one instant."""

    account: AccountState
    positions: tuple[Position, ...]
    open_orders: tuple[BrokerOrder, ...]


def liquidity_fetch_range(as_of_utc: datetime, lookback_days: int) -> tuple[date, date]:
    """Calendar-date range of the daily bars fetched for a decision at ``as_of_utc``.

    The range ends the day BEFORE the decision day, so the decision session's own daily
    bar is never read (no lookahead). It spans ``2 * lookback_days + 10`` calendar days
    (weekends and holidays included), the same margin the downloader uses for its daily
    warm-up; fewer bars than ``lookback_days`` fail closed in the check.
    """
    end = as_of_utc.date() - timedelta(days=1)
    return end - timedelta(days=2 * lookback_days + 10), end


class RiskGate:
    """Builds :class:`~domain.guards.checks.CheckContext` objects from the ports.

    Args:
        settings: Configuration-derived check settings.
        environment: Runtime facts source; ``None`` = unknown (fail closed).
        liquidity_data: Daily bars of ``universe.liquidity_feed``; ``None`` = unavailable.
        quote_data: Quote source for the spread filter; ``None`` = no quote.
    """

    def __init__(
        self,
        *,
        settings: GuardSettings,
        environment: GuardEnvironment | None,
        liquidity_data: IMarketData | None,
        quote_data: IMarketData | None,
    ) -> None:
        self._settings = settings
        self._environment = environment
        self._liquidity = liquidity_data
        self._quotes = quote_data
        self._daily: dict[tuple[str, date], tuple[Bar, ...]] = {}

    @classmethod
    def unconfigured(cls) -> RiskGate:
        """A gate with nothing configured or connected: every entry fails closed."""
        return cls(
            settings=GuardSettings.unconfigured(),
            environment=None,
            liquidity_data=None,
            quote_data=None,
        )

    @property
    def settings(self) -> GuardSettings:
        """The configuration-derived settings."""
        return self._settings

    def check_params(
        self,
        *,
        risk: RiskParams,
        min_tp_distance_ticks: int | None,
        cooldown_bars: int | None,
        min_minutes_per_bar: int | None,
    ) -> CheckParams:
        """Complete check parameters: the settings plus the caller's strategy/risk values."""
        settings = self._settings
        return CheckParams(
            whitelist=settings.whitelist,
            blacklist=settings.blacklist,
            min_price=settings.min_price,
            max_price=settings.max_price,
            min_avg_daily_volume=settings.min_avg_daily_volume,
            avg_volume_lookback_days=settings.avg_volume_lookback_days,
            liquidity_feed=settings.liquidity_feed,
            max_spread_bps=settings.max_spread_bps,
            spread_filter=settings.spread_filter,
            max_bar_age_seconds=settings.max_bar_age_seconds,
            max_reconcile_age_seconds=settings.max_reconcile_age_seconds,
            cooldown_bars=cooldown_bars,
            min_minutes_per_bar=min_minutes_per_bar,
            min_tp_distance_ticks=min_tp_distance_ticks,
            risk=risk,
        )

    async def broker_state(self, broker: IBroker) -> BrokerState:
        """Read account, positions and open orders (broker errors propagate)."""
        account = await broker.get_account()
        positions = tuple(await broker.get_positions())
        open_orders = tuple(await broker.get_open_orders())
        return BrokerState(account=account, positions=positions, open_orders=open_orders)

    async def runtime_facts(self) -> RuntimeFacts:
        """Runtime facts from the environment, or all unknown without one."""
        if self._environment is None:
            return RuntimeFacts.unavailable()
        return await self._environment.runtime_facts()

    async def daily_bars(self, symbol: Symbol, as_of_utc: datetime) -> tuple[Bar, ...] | None:
        """Liquidity daily bars of ``symbol`` for a decision at ``as_of_utc`` (cached per day).

        ``None`` when there is no source, no lookback setting, or the source is
        temporarily unreachable (``RetryableError``; not cached, retried next time).
        """
        lookback = self._settings.avg_volume_lookback_days
        if self._liquidity is None or lookback is None:
            return None
        key = (symbol, as_of_utc.date())
        cached = self._daily.get(key)
        if cached is not None:
            return cached
        start, end = liquidity_fetch_range(as_of_utc, lookback)
        try:
            bars = tuple(await self._liquidity.get_daily_bars(symbol, start, end))
        except RetryableError:
            return None
        self._daily = {k: v for k, v in self._daily.items() if k[0] != symbol}
        self._daily[key] = bars
        return bars

    async def quote(self, symbol: Symbol) -> Quote | None:
        """Latest quote when the spread filter is enforced (``None`` otherwise / unreachable)."""
        if self._settings.spread_filter is not SpreadFilter.ENFORCED or self._quotes is None:
            return None
        try:
            return await self._quotes.get_latest_quote(symbol)
        except RetryableError:
            return None

    async def context(
        self,
        *,
        now_utc: datetime,
        signal: Signal,
        signal_bar: Bar,
        bars_since_last_exit: int | None,
        trade: ProposedTrade | None,
        params: CheckParams,
        timing: EntryTiming,
        broker: BrokerState,
        pending_entries: Sequence[PendingEntry],
        position_stops: Sequence[PositionStop],
        week_start_equity: Money,
        peak_equity: Money,
        executed_signal_ids: frozenset[str] | None,
    ) -> CheckContext:
        """Assemble the frozen context of one signal at ``now_utc``."""
        symbol = signal.symbol
        return CheckContext(
            now_utc=now_utc,
            signal=signal,
            signal_bar=signal_bar,
            bars_since_last_exit=bars_since_last_exit,
            trade=trade,
            params=params,
            timing=timing,
            runtime=await self.runtime_facts(),
            market=MarketFacts(
                daily_bars=await self.daily_bars(symbol, signal_bar.bar_start_utc),
                quote=await self.quote(symbol),
            ),
            portfolio=PortfolioFacts(
                account=broker.account,
                positions=broker.positions,
                open_orders=broker.open_orders,
                pending_entries=tuple(pending_entries),
                position_stops=tuple(position_stops),
                week_start_equity=week_start_equity,
                peak_equity=peak_equity,
                executed_signal_ids=executed_signal_ids,
            ),
        )
