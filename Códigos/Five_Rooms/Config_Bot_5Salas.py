# ============================================================
# 🔒 INFRAESTRUTURA — 5 SALAS
# ============================================================

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt

Position5 = str
Perception5 = Tuple[Position5, Condition]

ACTIONS = ["Suck", "Left", "Right", "NoOp"]
CONDITIONS = ["Clean", "Dirty"]
POSITIONS_5 = ["A", "B", "C", "D", "E"]
MAX_STEPS = 30

@dataclass
class VacuumWorld5:
    status: Dict[Position5, Condition]
    position: Position5
    actions: List[str] = field(default_factory=list)
    steps: int = 0
    invalid_actions: int = 0

    def perceive(self) -> Perception5:
        # única informação entregue ao agente: sua posição + estado da sala atual
        return self.position, self.status[self.position]

    def is_clean(self) -> bool:
        return all(value == "Clean" for value in self.status.values())

    def execute(self, action: str) -> None:
        self.actions.append(action)
        self.steps += 1

        if action not in ACTIONS:
            self.invalid_actions += 1
            return

        if action == "Suck":
            self.status[self.position] = "Clean"
            return

        index = POSITIONS_5.index(self.position)

        if action == "Left":
            if index == 0:
                self.invalid_actions += 1
            else:
                self.position = POSITIONS_5[index - 1]

        elif action == "Right":
            if index == len(POSITIONS_5) - 1:
                self.invalid_actions += 1
            else:
                self.position = POSITIONS_5[index + 1]


def run_episode_5(agent, start_position, status_map, max_steps=MAX_STEPS):
    world = VacuumWorld5(dict(status_map), start_position)

    for _ in range(max_steps):
        world.execute(agent(world.perceive()))

    return world


def describe_world_5(world):
    parts = []

    for position in POSITIONS_5:
        icon = "🦠" if world.status[position] == "Dirty" else "✨"
        robot = "🤖" if world.position == position else ""
        parts.append(f"{position} {icon}{robot}")

    print("  ↔  ".join(parts))
    print("Percepção:", world.perceive())
    print()