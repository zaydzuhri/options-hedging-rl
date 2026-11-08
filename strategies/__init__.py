"""Registry: the name used in config files -> Strategy class. Add new strategies here."""

from strategies.baselines import DeltaBand, DeltaHedge, NoHedge

STRATEGIES = {
    "no_hedge": NoHedge,
    "delta_hedge": DeltaHedge,
    "delta_band": DeltaBand,
}
