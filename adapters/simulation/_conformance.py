"""Static (mypy-only) proof that every simulation adapter satisfies its port (sec. 8.5).

Nothing here runs: the assignments are type-checked by ``mypy adapters/simulation``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adapters.simulation.historical_feed import HistoricalFeed, ScriptedFeed
    from adapters.simulation.scripted_ai_filter import ScriptedAIFilter
    from adapters.simulation.sim_clock import AdvanceableClock, FixedClock, SimClock
    from adapters.simulation.simulated_broker import SimulatedBroker
    from adapters.simulation.static_calendar import StaticCalendar
    from domain.ports import IAIFilter, IBroker, IClock, IMarketCalendar, IMarketData

    def _check_conformance(
        sim_clock: SimClock,
        fixed_clock: FixedClock,
        calendar: StaticCalendar,
        historical: HistoricalFeed,
        scripted: ScriptedFeed,
        broker: SimulatedBroker,
        ai_filter: ScriptedAIFilter,
    ) -> None:
        _sim: IClock = sim_clock
        _fixed: IClock = fixed_clock
        _adv_sim: AdvanceableClock = sim_clock
        _adv_fixed: AdvanceableClock = fixed_clock
        _cal: IMarketCalendar = calendar
        _hist: IMarketData = historical
        _scr: IMarketData = scripted
        _brk: IBroker = broker
        _ai: IAIFilter = ai_filter
