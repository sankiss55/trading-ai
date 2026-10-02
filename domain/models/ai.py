"""AI veto filter models: input snapshot and response contract (sec. 17.2, 18).

Snapshot rules (sec. 17.2): no secrets, account ids, absolute equity or buying power,
file paths or internal configuration. The models below have no fields for those, and
``extra="forbid"`` prevents adding them ad hoc. Building the snapshot, its canonical
serialization and its SHA-256 hash belong to ``domain/ai/snapshot.py``.

Response rules (sec. 18): ``AIVerdict`` mirrors the JSON Schema of sec. 18.1 (types,
enums, required fields, no extra fields). The semantic checks of sec. 18.3 (stop reason,
identity, confidence range, rationale length, verdict/reason coherence, unique flags)
belong to ``domain/ai/verdict.py`` and are not duplicated here.
"""

from typing import Literal, Self

from pydantic import ConfigDict, Field, StrictStr, model_validator

from domain.models.base import (
    DomainModel,
    Money,
    NonEmptyStr,
    Price,
    Symbol,
    UtcDatetime,
)
from domain.models.enums import (
    AIReasonCode,
    AIRiskFlag,
    AIValidity,
    AIVerdictKind,
    BarStatus,
    ExitReason,
    OrderSide,
    Timeframe,
)
from domain.models.strategy import RuleResult

__all__ = [
    "AIUsage",
    "AIVerdict",
    "AIVerdictResult",
    "Snapshot",
    "SnapshotBar",
    "SnapshotHeadline",
    "SnapshotMarketContext",
    "SnapshotNews",
    "SnapshotPayload",
    "SnapshotPortfolioContext",
    "SnapshotProposedTrade",
    "SnapshotRecentOutcome",
    "SnapshotSession",
    "SnapshotTimeframe",
]


# --------------------------------------------------------------------------- snapshot


class SnapshotSession(DomainModel):
    """Session position of the signal (``session`` block of sec. 17.2)."""

    minutes_since_open: int
    minutes_to_close: int
    is_early_close_day: bool


class SnapshotBar(DomainModel):
    """Compact bar inside the snapshot. Prices are ``None`` only for EMPTY bars."""

    t: UtcDatetime
    o: Price | None
    h: Price | None
    l: Price | None  # noqa: E741 - field name fixed by the snapshot contract (sec. 17.2)
    c: Price | None
    v: int = Field(ge=0)
    status: BarStatus


class SnapshotTimeframe(DomainModel):
    """Bars and indicator values of one timeframe."""

    timeframe: Timeframe
    bars: tuple[SnapshotBar, ...]
    indicators: dict[str, Money | None] = Field(default_factory=dict)


class SnapshotProposedTrade(DomainModel):
    """Proposed trade as shown to the model: relative risk only, no absolute amounts."""

    side: Literal[OrderSide.BUY] = OrderSide.BUY
    entry_ref: Price
    stop_price: Price
    take_profit_price: Price
    r_multiple: Money
    risk_pct_of_equity: Money


class SnapshotMarketContext(DomainModel):
    """Benchmark context (``market_context`` block of sec. 17.2)."""

    benchmark_symbol: Symbol
    benchmark_change_today_pct: Money | None
    benchmark_above_ema_fast: bool | None


class SnapshotRecentOutcome(DomainModel):
    """One recent closed trade of the same symbol, from the operational DB."""

    closed_at_utc: UtcDatetime
    result_r: Money
    exit_reason: ExitReason


class SnapshotPortfolioContext(DomainModel):
    """Portfolio context as percentages and counts only (sec. 17.2.3)."""

    open_positions: int = Field(ge=0)
    daily_pnl_pct: Money
    trades_today: int = Field(ge=0)
    recent_outcomes_symbol: tuple[SnapshotRecentOutcome, ...] = ()


class SnapshotHeadline(DomainModel):
    """One untrusted news headline (sec. 10.7, 43.2). Data, never instructions."""

    source: NonEmptyStr
    published_at_utc: UtcDatetime
    text: str


class SnapshotNews(DomainModel):
    """News block; disabled unless ``ai.include_news`` is true (OWNER_DECISION)."""

    enabled: bool = False
    untrusted_headlines: tuple[SnapshotHeadline, ...] = ()


class SnapshotPayload(DomainModel):
    """Full snapshot content of sec. 17.2 (everything that is hashed)."""

    snapshot_version: NonEmptyStr
    signal_id: NonEmptyStr
    symbol: Symbol
    timestamp_utc: UtcDatetime
    session: SnapshotSession
    primary_timeframe: SnapshotTimeframe
    confirmation_timeframe: SnapshotTimeframe | None
    rule_results: tuple[RuleResult, ...]
    proposed_trade: SnapshotProposedTrade
    market_context: SnapshotMarketContext
    portfolio_context: SnapshotPortfolioContext
    news: SnapshotNews = SnapshotNews()


class Snapshot(DomainModel):
    """Immutable AI input: the payload plus the SHA-256 hex digest of its canonical JSON.

    The hash is computed by ``domain/ai/snapshot.py``; this model only validates its
    format.
    """

    payload: SnapshotPayload
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


# --------------------------------------------------------------------------- response


class AIVerdict(DomainModel):
    """Parsed AI response, mirroring the JSON Schema of sec. 18.1.

    Types are strict (no string-to-number coercion), unknown fields are rejected.
    Passing this model does NOT make a response valid: sec. 18.3 checks in
    ``domain/ai/verdict.py`` are mandatory before reporting ``AIValidity.VALID``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    signal_id: StrictStr
    verdict: AIVerdictKind
    reason_code: AIReasonCode
    risk_flags: tuple[AIRiskFlag, ...]
    confidence: float
    """JSON number; integers are accepted and converted, strings and booleans rejected."""
    rationale: StrictStr


class AIUsage(DomainModel):
    """Token usage of one model call (sec. 34.1). Field names VERIFICAR against the SDK."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)


class AIVerdictResult(DomainModel):
    """Outcome of ``IAIFilter.evaluate``: always returned, never raised (sec. 8.5).

    ``VALID`` requires a verdict and no invalid reason; ``INVALID`` / ``UNAVAILABLE``
    require an invalid reason and no verdict.
    """

    validity: AIValidity
    verdict: AIVerdict | None = None
    invalid_reason: str | None = None
    raw_response: str | None = None
    usage: AIUsage | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    stop_reason: str | None = None
    model_id: str | None = None

    @model_validator(mode="after")
    def _check_validity(self) -> Self:
        if self.validity is AIValidity.VALID:
            if self.verdict is None:
                raise ValueError("VALID results require a verdict")
            if self.invalid_reason is not None:
                raise ValueError("VALID results cannot carry an invalid_reason")
        else:
            if self.verdict is not None:
                raise ValueError(f"{self.validity} results cannot carry a verdict")
            if not self.invalid_reason:
                raise ValueError(f"{self.validity} results require an invalid_reason")
        return self
