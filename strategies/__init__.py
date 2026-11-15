"""Registry: the name used in config files -> Strategy class. Add new strategies here."""

from strategies.baselines import DeltaBand, DeltaHedge, NoHedge, WhalleyWilmott
from strategies.dp import DPHedge
from strategies.ilqg import ILQGHedge
from strategies.mpc import MPCHedge

STRATEGIES = {
    "no_hedge": NoHedge,
    "delta_hedge": DeltaHedge,
    "delta_band": DeltaBand,
    "ww_band": WhalleyWilmott,
    "mpc": MPCHedge,
    "dp": DPHedge,
    "ilqg": ILQGHedge,
}
