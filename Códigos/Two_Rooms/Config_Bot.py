# =============================== #
# 🔒 INFRAESTRUTURA — 2 SALAS 🔒 #
# =============================== #

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt

Position = str
Condition = str
Perception = Tuple[Position, Condition]

POSITIONS = ["A", "B"]
CONDITIONS = ["Clean", "Dirty"]
ACTIONS = ["Suck", "Left", "Right", "NoOp"]
MAX_STEPS = 30

@dataclass
class VacuumWorld:
    # Estado de cada sala. Ex.: {"A": "Dirty", "B": "Clean"}
    status: Dict[Position, Condition]
    # Posição atual do agente
    position: Position
    # Registro da execução pelo simulador
    actions: List[str] = field(default_factory=list)
    steps: int = 0

    def perceive(self) -> Perception:
        # O agente percebe somente sua posição e o estado da sala atual
        return self.position, self.status[self.position]

    def is_clean(self) -> bool:
        return all(value == "Clean" for value in self.status.values())

    def execute(self, action: str) -> None:
        self.actions.append(action)
        self.steps += 1

        if action == "Suck":
            self.status[self.position] = "Clean"
        elif action == "Left" and self.position == "B":
            self.position = "A"
        elif action == "Right" and self.position == "A":
            self.position = "B"


def run_episode(agent, start_position, status_map, max_steps=MAX_STEPS):
    world = VacuumWorld(dict(status_map), start_position)

    for _ in range(max_steps):
        # 1. Perceber
        perception = world.perceive()

        # 2. Decidir
        action = agent(perception)

        # 3. Agir
        world.execute(action)

    return world


def all_states():
    # 2 posições iniciais × 2² configurações de Clean/Dirty = 8 estados.
    for start in POSITIONS:
        for values in product(CONDITIONS, repeat=2):
            yield start, dict(zip(POSITIONS, values))


print("✅ Vacuum World de 2 salas carregado.")