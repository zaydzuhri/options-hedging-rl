"""Learned hedging strategies: load a policy trained by deep_hedge.py and run it in the backtester."""

from pathlib import Path

from strategy import Observation, Strategy


class DeepHedge(Strategy):
    """A policy trained with pathwise gradients through whole episodes (see deep_hedge.py).

    params:
      model (str): checkpoint path; `uv run deep_hedge.py --config <config>` writes it.
      The other params of the config entry (arch, objective, steps, ...) are only read at training time.
    """

    def __init__(self, **params):
        super().__init__(**params)
        if not Path(params["model"]).exists():
            raise FileNotFoundError(f"{params['model']} not found: train it first with "
                                    "`uv run deep_hedge.py --config <this config>`")
        import deep_hedge   # imported here so the classical strategies don't need torch

        self._act = deep_hedge.act
        self.policy = deep_hedge.load_policy(params["model"])

    def act(self, obs: Observation) -> float:
        return self._act(self.policy, obs)
