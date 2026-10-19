"""Engine protocol (spec 3.6 / Technical Implementation 3.6): one protocol, interchangeable
implementations (simulated, real, cached), selectable task by task. The rest of the pipeline does not
know which one it is talking to."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel
from tle.core.rng import GameRng


@dataclass
class LabelContext:
    """Struct-of-arrays context of a batch of citizens in labels (6.5): what S1 would see.

    Label arrays hold S7 indices 0..6 (bipolar reading for T, C, I, O; unipolar for fear and risk).
    """

    ids: np.ndarray  # (m,) citizen ids
    age_band: np.ndarray  # (m,) 0 minor, 1 young, 2 adult, 3 elderly
    social_class: np.ndarray  # (m,) 0 low, 1 middle, 2 high
    trade: np.ndarray  # (m,) trade index
    circle: np.ndarray
    T: np.ndarray  # (m, 4)
    C: np.ndarray  # (m, 3)
    I: np.ndarray  # (m, 3)
    O: np.ndarray  # (m, 6)
    fear: np.ndarray  # (m,) unipolar index
    risk: np.ndarray  # (m,) redistributive risk label (upper class only; 0 otherwise)
    member: np.ndarray | None = None  # (m,) bool: belongs to the class/custodian group that acts

    def __len__(self) -> int:
        return len(self.ids)

    def row(self, i: int) -> dict[str, Any]:
        """One citizen as a plain dict (used to render prompts for real engines)."""
        return {
            "age_band": ("minor", "young", "adult", "elderly")[int(self.age_band[i])],
            "class": ("low", "middle", "high")[int(self.social_class[i])],
            "trade": int(self.trade[i]),
            "circle": int(self.circle[i]),
            "temperament": [int(x) - 3 for x in self.T[i]],
            "character": [int(x) - 3 for x in self.C[i]],
            "ideology": [int(x) - 3 for x in self.I[i]],
            "opinion": [int(x) - 3 for x in self.O[i]],
            "fear": int(self.fear[i]),
            "redistributive_risk": int(self.risk[i]),
        }


@dataclass
class ActorCtx:
    kind: str  # mayor | faction | influencer | class | custodian | exogenous | threat | system
    ideology: np.ndarray | None = None  # (3,) label indices of the actor's ideology (faction/influencer)


@dataclass
class OptionFeatures:
    """Numeric reading (P8 lookup) of the extracted features of the K options of one event."""

    t_I: np.ndarray  # (K, 3) bipolar value or NaN (mask)
    t_C: np.ndarray  # (K, 3)
    e_r: np.ndarray  # (K,) bipolar value or NaN
    F: np.ndarray  # (K,) unipolar value
    stance: np.ndarray  # (K, 8): mayor, pillar x5, actor, target (NaN = mask)

    @property
    def K(self) -> int:
        return len(self.F)

    @property
    def t_O(self) -> np.ndarray:
        """(K, 6) opinion targets: stance toward the mayor and acceptance of the erosion of each pillar."""
        return self.stance[:, :6]

    @property
    def stance_actor(self) -> np.ndarray:
        return self.stance[:, 6]

    @property
    def stance_target(self) -> np.ndarray:
        return self.stance[:, 7]


@dataclass
class ExecutionRecord:
    task: str
    calls: int = 0
    attempts: int = 0
    fallbacks: int = 0
    regenerations: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    latency_s: float = 0.0
    backend: str = "simulated"
    model: str = ""


@runtime_checkable
class Engine(Protocol):
    """Both S1 and S2 reach models only through this protocol."""

    name: str

    def call(self, task: str, inp: BaseModel, rng: GameRng) -> BaseModel: ...

    def choose_reaction_many(
        self,
        ctx: LabelContext,
        actor: ActorCtx,
        news: Any,
        options: list[Any],
        features: OptionFeatures,
        rng: GameRng,
    ) -> np.ndarray: ...

    def choose_activity_many(self, ctx: LabelContext, rng: GameRng) -> np.ndarray: ...

    def drain_execution(self) -> list[ExecutionRecord]: ...

    def drain_warnings(self) -> list[dict[str, Any]]: ...


@dataclass
class Engines:
    """The two engines of a game (S1: classifier/chooser, S2: narrator)."""

    s1: Engine
    s2: Engine

    def drain_execution(self) -> list[ExecutionRecord]:
        return [*self.s1.drain_execution(), *self.s2.drain_execution()]

    def drain_warnings(self) -> list[dict[str, Any]]:
        return [*self.s1.drain_warnings(), *self.s2.drain_warnings()]


@dataclass
class Warning_:
    task: str
    input: str
    response: str
    day: float = 0.0
    extra: dict = field(default_factory=dict)
