"""Registry: the name used in config files -> Strategy class. Add new strategies here."""

from strategies.baselines import DeltaBand, DeltaHedge, NoHedge, WhalleyWilmott
from strategies.dp import DPHedge
from strategies.learned import DeepHedge
from strategies.mpc import MPCHedge
from strategies.optimal import DPOptimal

STRATEGIES = {
    "no_hedge": NoHedge,
    "delta_hedge": DeltaHedge,
    "delta_band": DeltaBand,
    "ww_band": WhalleyWilmott,
    "whalley_wilmott": WhalleyWilmott,   # same class, two names (both branches' configs use theirs)
    "mpc": MPCHedge,
    "dp": DPHedge,                       # DP for any model in strategies/models.py, solved on the fly
    "dp_optimal": DPOptimal,             # DP for the Black-Scholes world, solved by dp_optimal.py
    "deep_hedge": DeepHedge,
}
