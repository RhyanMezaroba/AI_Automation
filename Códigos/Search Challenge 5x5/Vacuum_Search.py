"""
====================================================================
🤖 AULA 07 — MÉTODOS E ESTRATÉGIAS DE BUSCA (UCS)
🧹 VACUUM SEARCH — PLANEJAMENTO E OTIMIZAÇÃO DE CUSTOS DE LAÇO
====================================================================
Curso: Inteligência Artificial — UNOESC (Ciência da Computação)
Script: Vacuum_Search.py
Algoritmo: Busca de Custo Uniforme (Uniform Cost Search - UCS)

Objetivo:
    Planejar a sequência ótima de ações para um robô aspirador limpar
    as salas sujas do ambiente minimizando o custo total acumulado g(n)
    e eliminando custos de laço (ciclos redundantes e oscilações de movimento).

Módulos incluídos:
    1. Ambiente Linear da Aula 07 (5 salas: A, B, C, D, E)
    2. Algoritmo UCS com descarte de laços e fila de prioridade (heapq)
    3. Simulador passo a passo com representação gráfica dos estados
    4. Módulo de Desafio em Grade 5x5 (Search Challenge 5x5 com obstáculos)
====================================================================
"""

from typing import Tuple, List, Dict, Optional, Set, FrozenSet
from heapq import heappush, heappop
from dataclasses import dataclass, field
import time
import sys

# Garante suporte a emojis e caracteres UTF-8 no console do Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass



# ====================================================================
# 1. REPRESENTAÇÃO DO VACUUM WORLD (5 SALAS — AULA 07)
# ====================================================================

Position = str
Condition = str
# Estado: (posição_do_robô, status_A, status_B, status_C, status_D, status_E)
State = Tuple[Position, Condition, Condition, Condition, Condition, Condition]

POSITIONS = ["A", "B", "C", "D", "E"]
ACTIONS = ["Suck", "Left", "Right"]

# Tabela de custos padrão da Aula 07:
# Mover custa 1, Aspirar (Suck) custa 2.
DEFAULT_ACTION_COSTS: Dict[str, int] = {
    "Suck": 2,
    "Left": 1,
    "Right": 1
}

# Custos uniformes (todas as ações custam 1) para comparação
UNIFORM_ACTION_COSTS: Dict[str, int] = {
    "Suck": 1,
    "Left": 1,
    "Right": 1
}

# Estado inicial padrão trabalhado na Aula 07:
# A 🦠 ↔ B ✨ ↔ C 🤖🦠 ↔ D ✨ ↔ E 🦠
INITIAL_STATE_AULA_07: State = (
    "C",      # Robô começa na sala C
    "Dirty",  # Sala A suja
    "Clean",  # Sala B limpa
    "Dirty",  # Sala C suja
    "Clean",  # Sala D limpa
    "Dirty"   # Sala E suja
)


# ====================================================================
# 2. FUNÇÕES DO AMBIENTE (TRANSIÇÃO E SUCESSORES)
# ====================================================================

def describe_state(state: State) -> str:
    """Gera uma representação textual visual do estado com ícones."""
    position = state[0]
    rooms = state[1:]

    parts = []
    for room, condition in zip(POSITIONS, rooms):
        icon = "🦠" if condition == "Dirty" else "✨"
        robot = "🤖" if room == position else ""
        parts.append(f"[{room} {robot}{icon}]")

    return " ↔ ".join(parts)


def is_goal(state: State) -> bool:
    """
    Critério de parada: o objetivo é alcançado quando TODAS as salas
    estão limpas ('Clean'). A posição final do robô não importa.
    """
    rooms = state[1:]
    return all(condition == "Clean" for condition in rooms)


def apply_action(state: State, action: str) -> Optional[State]:
    """
    Aplica uma ação sobre o estado atual e retorna o novo estado.
    Se a ação for fisicamente impossível ou inválida, retorna None.
    
    Regras:
    - Suck: se a sala atual estiver 'Dirty', passa para 'Clean'.
    - Left: move para a esquerda se não estiver na borda esquerda (A).
    - Right: move para a direita se não estiver na borda direita (E).
    """
    position = state[0]
    rooms = list(state[1:])
    index = POSITIONS.index(position)

    if action == "Suck":
        if rooms[index] == "Dirty":
            rooms[index] = "Clean"
            return (position, *rooms)
        # Aspirar uma sala já limpa é ineficaz e descartado
        return None

    elif action == "Left":
        if index > 0:
            new_position = POSITIONS[index - 1]
            return (new_position, *rooms)
        return None

    elif action == "Right":
        if index < len(POSITIONS) - 1:
            new_position = POSITIONS[index + 1]
            return (new_position, *rooms)
        return None

    return None


def successors(state: State) -> List[Tuple[str, State]]:
    """Gera todos os pares válidos (ação, novo_estado) a partir do estado atual."""
    valid_successors = []
    for action in ACTIONS:
        new_state = apply_action(state, action)
        if new_state is not None:
            valid_successors.append((action, new_state))
    return valid_successors


# ====================================================================
# 3. ESTRUTURA DE RESULTADOS DA BUSCA
# ====================================================================

@dataclass
class SearchResult:
    """Armazena as métricas completas da execução da busca."""
    plan: Optional[List[str]]
    cost: Optional[int]
    expanded_nodes: int
    loops_avoided: int
    generated_nodes: int
    execution_time_ms: float


# ====================================================================
# 4. ALGORITMO UCS (UNIFORM COST SEARCH)
# ====================================================================

def ucs(initial_state: State, action_costs: Optional[Dict[str, int]] = None) -> SearchResult:
    """
    Executa a Busca de Custo Uniforme (UCS) para encontrar o plano com MENOR CUSTO.
    
    Como o UCS reduz os custos de laço (ciclos redundantes):
    ---------------------------------------------------------
    1. Fila de Prioridade (Min-Heap): Sempre expande o caminho com menor g(n).
       Caminhos que ficam em laços (ex: A -> B -> A) acumulam custo desnecessário
       e são naturalmente empurrados para o final da fila.
    2. Dicionário best_cost (Grafo Fechado): Se um estado gerado já foi visitado
       com custo menor ou igual, a nova transição é identificada como um laço/ciclo
       e é IMEDIATAMENTE PODADA (loops_avoided).
    3. Poda de Entradas Desatualizadas: Se um nó foi inserido com custo maior
       antes de descobrirmos um caminho melhor para ele, ele é ignorado no heappop.
    """
    if action_costs is None:
        action_costs = DEFAULT_ACTION_COSTS

    start_time = time.perf_counter()

    # 1. FRONTEIRA (Min-Heap com desempate por contador sequencial)
    # Tupla: (custo_acumulado_g, contador_desempate, estado, plano_de_ações)
    frontier = []
    counter = 0
    heappush(frontier, (0, counter, initial_state, []))

    # 2. MELHOR CUSTO CONHECIDO POR ESTADO (Elimina laços e redundâncias)
    best_cost: Dict[State, int] = {initial_state: 0}

    # Métricas de análise
    expanded = 0
    loops_avoided = 0
    generated = 1

    while frontier:
        cost, _, current_state, plan = heappop(frontier)

        # 3. VERIFICAÇÃO DE ENTRADA DESATUALIZADA
        # Se este estado já teve um caminho mais barato expandido, descarte
        if cost != best_cost.get(current_state):
            continue

        # 4. TESTE DE OBJETIVO
        # No UCS, o teste de objetivo é feito ao DESENFILEIRAR (garante otimalidade)
        if is_goal(current_state):
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return SearchResult(
                plan=plan,
                cost=cost,
                expanded_nodes=expanded,
                loops_avoided=loops_avoided,
                generated_nodes=generated,
                execution_time_ms=elapsed_ms
            )

        # 5. EXPANSÃO DO ESTADO
        expanded += 1

        # 6. GERAÇÃO DE SUCESSORES
        for action, new_state in successors(current_state):
            step_cost = action_costs.get(action, 1)
            new_cost = cost + step_cost
            generated += 1

            # 7. FILTRAGEM DE LAÇO: Só enfileira se for estritamente mais barato
            if new_cost < best_cost.get(new_state, float("inf")):
                best_cost[new_state] = new_cost
                counter += 1
                heappush(frontier, (new_cost, counter, new_state, plan + [action]))
            else:
                # Este ramo foi descartado por formar um laço/ciclo ou caminho mais caro
                loops_avoided += 1

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return SearchResult(
        plan=None,
        cost=None,
        expanded_nodes=expanded,
        loops_avoided=loops_avoided,
        generated_nodes=generated,
        execution_time_ms=elapsed_ms
    )


# ====================================================================
# 5. SIMULADOR PASSO A PASSO DA EXECUÇÃO DO ROBÔ
# ====================================================================

def simulate_execution(initial_state: State, plan: List[str], action_costs: Dict[str, int]) -> None:
    """Executa o plano gerado pelo UCS passo a passo, exibindo o robô no console."""
    print("\n" + "=" * 65)
    print("🚀 INICIANDO SIMULAÇÃO PASSO A PASSO DO ROBÔ")
    print("=" * 65)

    current_state = initial_state
    accumulated_cost = 0

    print(f"\n[Passo 00] Estado Inicial (Custo Acumulado: {accumulated_cost})")
    print(f"  Visual: {describe_state(current_state)}")
    print(f"  Salas sujas restantes: {sum(c == 'Dirty' for c in current_state[1:])}")

    for step, action in enumerate(plan, 1):
        step_cost = action_costs[action]
        accumulated_cost += step_cost
        next_state = apply_action(current_state, action)

        action_symbol = "🌀 ASPIRAR" if action == "Suck" else f"➡️ MOVER ({action})"
        print(f"\n[Passo {step:02d}] Ação: {action_symbol} | Custo da ação: +{step_cost} | Custo Total: {accumulated_cost}")
        
        if next_state is not None:
            current_state = next_state
        print(f"  Visual: {describe_state(current_state)}")
        dirty_count = sum(c == "Dirty" for c in current_state[1:])
        print(f"  Salas sujas restantes: {dirty_count}")

    print("\n" + "-" * 65)
    if is_goal(current_state):
        print("🎉 MISSÃO CUMPRIDA! Todas as salas estão completamente limpas ✨")
        print(f"Total de passos executados: {len(plan)}")
        print(f"Custo total final: {accumulated_cost}")
    else:
        print("⚠️ AVISO: O plano terminou mas restam salas sujas!")
    print("-" * 65)


# ====================================================================
# 6. MÓDULO ESTENDIDO — SEARCH CHALLENGE GRADE 5x5
# ====================================================================

@dataclass(frozen=True)
class GridState:
    """Estado do robô em uma grade 2D (5x5)."""
    robot_pos: Tuple[int, int]                  # (linha, coluna) de 0 a 4
    dirty_cells: FrozenSet[Tuple[int, int]]     # conjunto de coordenadas sujas


class GridVacuumWorld5x5:
    """
    Implementação do Desafio Vacuum World em Grade 5x5 com obstáculos.
    Permite movimentação em 4 direções (Up, Down, Left, Right) e Suck.
    """
    GRID_ROWS = 5
    GRID_COLS = 5
    ACTIONS = ["Up", "Down", "Left", "Right", "Suck"]

    DEFAULT_COSTS = {
        "Suck": 2,
        "Up": 1,
        "Down": 1,
        "Left": 1,
        "Right": 1
    }

    DELTAS = {
        "Up": (-1, 0),
        "Down": (1, 0),
        "Left": (0, -1),
        "Right": (0, 1)
    }

    def __init__(self, obstacles: Optional[Set[Tuple[int, int]]] = None):
        self.obstacles = obstacles or set()

    def is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.GRID_ROWS and 0 <= c < self.GRID_COLS and pos not in self.obstacles

    def apply_action(self, state: GridState, action: str) -> Optional[GridState]:
        r, c = state.robot_pos

        if action == "Suck":
            if state.robot_pos in state.dirty_cells:
                new_dirty = frozenset(cell for cell in state.dirty_cells if cell != state.robot_pos)
                return GridState(robot_pos=state.robot_pos, dirty_cells=new_dirty)
            return None

        if action in self.DELTAS:
            dr, dc = self.DELTAS[action]
            new_pos = (r + dr, c + dc)
            if self.is_valid_pos(new_pos):
                return GridState(robot_pos=new_pos, dirty_cells=state.dirty_cells)
            return None

        return None

    def successors(self, state: GridState) -> List[Tuple[str, GridState]]:
        res = []
        for action in self.ACTIONS:
            new_state = self.apply_action(state, action)
            if new_state is not None:
                res.append((action, new_state))
        return res

    def is_goal(self, state: GridState) -> bool:
        return len(state.dirty_cells) == 0

    def ucs(self, initial_state: GridState, action_costs: Optional[Dict[str, int]] = None) -> SearchResult:
        costs = action_costs or self.DEFAULT_COSTS
        start_time = time.perf_counter()

        frontier = []
        counter = 0
        heappush(frontier, (0, counter, initial_state, []))
        best_cost: Dict[GridState, int] = {initial_state: 0}

        expanded = 0
        loops_avoided = 0
        generated = 1

        while frontier:
            cost, _, current, plan = heappop(frontier)

            if cost != best_cost.get(current):
                continue

            if self.is_goal(current):
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return SearchResult(
                    plan=plan,
                    cost=cost,
                    expanded_nodes=expanded,
                    loops_avoided=loops_avoided,
                    generated_nodes=generated,
                    execution_time_ms=elapsed_ms
                )

            expanded += 1

            for action, new_state in self.successors(current):
                step_cost = costs.get(action, 1)
                new_cost = cost + step_cost
                generated += 1

                if new_cost < best_cost.get(new_state, float("inf")):
                    best_cost[new_state] = new_cost
                    counter += 1
                    heappush(frontier, (new_cost, counter, new_state, plan + [action]))
                else:
                    loops_avoided += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return SearchResult(None, None, expanded, loops_avoided, generated, elapsed_ms)

    def print_grid(self, state: GridState) -> None:
        """Renderiza a grade 5x5 no terminal."""
        print("  ┌" + "───┬" * (self.GRID_COLS - 1) + "───┐")
        for r in range(self.GRID_ROWS):
            row_str = f"{r} │"
            for c in range(self.GRID_COLS):
                pos = (r, c)
                if pos == state.robot_pos:
                    cell = "🤖"
                elif pos in self.obstacles:
                    cell = "🧱"
                elif pos in state.dirty_cells:
                    cell = "🦠"
                else:
                    cell = "✨"
                row_str += f"{cell} │"
            print(row_str)
            if r < self.GRID_ROWS - 1:
                print("  ├" + "───┼" * (self.GRID_COLS - 1) + "───┤")
        print("  └" + "───┴" * (self.GRID_COLS - 1) + "───┘")
        print("    " + "   ".join(str(c) for c in range(self.GRID_COLS)))


# ====================================================================
# 7. EXECUÇÃO PRINCIPAL E CENÁRIOS DE TESTE
# ====================================================================

def run_aula_07_scenarios():
    print("=" * 70)
    print("🤖 VACUUM WORLD (AULA 07) — BUSCA DE CUSTO UNIFORME (UCS)")
    print("=" * 70)
    print("Configuração:")
    print(f"Salas: {POSITIONS}")
    print(f"Custos diferenciados: {DEFAULT_ACTION_COSTS}")
    print(f"Estado Inicial: {describe_state(INITIAL_STATE_AULA_07)}")
    print("-" * 70)

    # 1. Execução com Custos Diferenciados da Aula 07 (Suck=2, Move=1)
    print("\n🔍 1. Executando UCS com CUSTOS DIFERENCIADOS (Suck=2, Mover=1)...")
    res_diff = ucs(INITIAL_STATE_AULA_07, DEFAULT_ACTION_COSTS)

    print(f"  • Status: {'✅ Solução encontrada' if res_diff.plan else '❌ Falha'}")
    print(f"  • Plano de Ações: {res_diff.plan}")
    print(f"  • Custo Total Ótimo: {res_diff.cost}")
    print(f"  • Passos no Plano: {len(res_diff.plan) if res_diff.plan else 0}")
    print(f"  • Estados Expandidos: {res_diff.expanded_nodes}")
    print(f"  • Laços/Ciclos Evitados (Poda): {res_diff.loops_avoided}")
    print(f"  • Estados Gerados: {res_diff.generated_nodes}")
    print(f"  • Tempo de Busca: {res_diff.execution_time_ms:.2f} ms")

    # 2. Execução com Custos Uniformes (Todas as ações custam 1)
    print("\n🔍 2. Executando UCS com CUSTOS UNIFORMES (Todas as ações = 1)...")
    res_unif = ucs(INITIAL_STATE_AULA_07, UNIFORM_ACTION_COSTS)

    print(f"  • Status: {'✅ Solução encontrada' if res_unif.plan else '❌ Falha'}")
    print(f"  • Plano de Ações: {res_unif.plan}")
    print(f"  • Custo Total: {res_unif.cost}")
    print(f"  • Estados Expandidos: {res_unif.expanded_nodes}")
    print(f"  • Laços/Ciclos Evitados (Poda): {res_unif.loops_avoided}")
    print(f"  • Tempo de Busca: {res_unif.execution_time_ms:.2f} ms")

    # 3. Simulação visual do plano ótimo
    if res_diff.plan:
        simulate_execution(INITIAL_STATE_AULA_07, res_diff.plan, DEFAULT_ACTION_COSTS)


def run_grid_5x5_demo():
    print("\n" + "=" * 70)
    print("🗺️ BÔNUS — SEARCH CHALLENGE: VACUUM WORLD EM GRADE 5x5 COM OBSTÁCULOS")
    print("=" * 70)

    # Obstáculos (paredes)
    obstacles = {(1, 1), (1, 2), (2, 3), (3, 1)}
    # Células sujas na grade
    dirty_cells = frozenset({(0, 4), (2, 2), (4, 0), (4, 4)})
    # Robô começa em (0, 0)
    initial_grid_state = GridState(robot_pos=(0, 0), dirty_cells=dirty_cells)

    world5x5 = GridVacuumWorld5x5(obstacles=obstacles)

    print("\nEstado Inicial da Grade 5x5:")
    print("Legenda: 🤖 Robô | 🧱 Obstáculo | 🦠 Sujeira | ✨ Limpo")
    world5x5.print_grid(initial_grid_state)

    print("\n🔍 Executando UCS na grade 5x5...")
    res = world5x5.ucs(initial_grid_state)

    if res.plan:
        print(f"\n✅ Solução Ótima Encontrada!")
        print(f"  • Custo total: {res.cost}")
        print(f"  • Quantidade de ações: {len(res.plan)}")
        print(f"  • Estados expandidos: {res.expanded_nodes}")
        print(f"  • Laços redundantes podados: {res.loops_avoided}")
        print(f"  • Tempo: {res.execution_time_ms:.2f} ms")
        print(f"  • Primeiras 10 ações: {res.plan[:10]} ...")
    else:
        print("❌ Não foi possível encontrar solução.")


if __name__ == "__main__":
    # Executa o ambiente linear da Aula 07
    run_aula_07_scenarios()

    # Executa a demonstração do desafio em grade 5x5
    run_grid_5x5_demo()
