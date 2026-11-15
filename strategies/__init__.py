"""Registry: the name used in config files -> Strategy class. Add new strategies here."""

from strategies.baselines import DeltaBand, DeltaHedge, NoHedge, WhalleyWilmott
from strategies.learned import DeepHedge

STRATEGIES = {
    "no_hedge": NoHedge,
    "delta_hedge": DeltaHedge,
    "delta_band": DeltaBand,
    "whalley_wilmott": WhalleyWilmott,
    "deep_hedge": DeepHedge,
}
