"""Registry: the name used in config files -> Strategy class. Add new strategies here."""

from strategies.baselines import DeltaBand, DeltaHedge, NoHedge, WhalleyWilmott
from strategies.learned import DeepHedge
from strategies.optimal import DPOptimal

STRATEGIES = {
    "no_hedge": NoHedge,
    "delta_hedge": DeltaHedge,
    "delta_band": DeltaBand,
    "whalley_wilmott": WhalleyWilmott,
    "dp_optimal": DPOptimal,
    "deep_hedge": DeepHedge,
}
