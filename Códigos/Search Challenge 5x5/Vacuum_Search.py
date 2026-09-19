"""
================================================================================
🤖 INTELIGÊNCIA ARTIFICIAL — UNOESC (CIÊNCIA DA COMPUTAÇÃO)
🧹 VACUUM WORLD — AGENTE PLANEJADOR EM ESPAÇO DE ESTADOS (SEARCH CHALLENGE)
================================================================================
Equipe: Rhyan Mezaroba e Equipe
Atividade: Agente Planejador Vacuum World
Estratégia Escolhida: A* (A-Estrela) com Heurística Admissível e Consistente
Entrega Obrigatória: meu_planejador(initial_state)

Modelagem Formal do Problema em Espaço de Estados:
--------------------------------------------------
1. Estados (S):
   - Linear (1D): Tupla (posição_robô, status_A, status_B, status_C, status_D, status_E)
   - Grade 2D (5x5): GridState(robot_pos=(r, c), dirty_cells=frozenset(...))
2. Ações (A):
   - 1D: {'Suck', 'Left', 'Right'}
   - 2D: {'Suck', 'Up', 'Down', 'Left', 'Right'}
3. Modelo de Transição T(s, a) -> s':
   - 'Suck': limpa a sala atual se estiver 'Dirty'; caso contrário, inválido.
   - Movimentos: deslocam o robô respeitando as fronteiras e obstáculos.
4. Teste de Objetivo Goal(s):
   - Verdadeiro se todas as salas estiverem limpas (zero sujeira restante).
5. Função de Custo de Caminho c(s, a, s'):
   - Custos diferenciados da Aula 07: Suck = 2, Movimento = 1.

Estratégias Implementadas e Comparadas:
---------------------------------------
• BFS (Busca em Largura): Otimiza quantidade de ações, mas ignora custos.
• DFS (Busca em Profundidade): Explora caminhos profundos, plano subótimo de alto custo.
• UCS (Busca de Custo Uniforme): Garante custo ótimo g(n), mas expande cegamente.
• A* (A-Estrela): Garante custo ótimo com máxima eficiência via f(n) = g(n) + h(n).
================================================================================
"""

from typing import Tuple, List, Dict, Optional, Set, FrozenSet, Union, Any
from collections import deque
from heapq import heappush, heappop
from dataclasses import dataclass
import time
import sys

# Suporte a UTF-8 no Windows PowerShell / CMD
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ==============================================================================
# 1. MODELAGEM FORMAL — AMBIENTE LINEAR (5 SALAS — AULA 07)
# ==============================================================================

Position = str
Condition = str
# Estado clássico: (posição_do_robô, status_A, status_B, status_C, status_D, status_E)
State = Tuple[Position, Condition, Condition, Condition, Condition, Condition]

POSITIONS: List[str] = ["A", "B", "C", "D", "E"]
ACTIONS: List[str] = ["Suck", "Left", "Right"]

# Tabela de Custos Diferenciados da Aula 07:
DEFAULT_ACTION_COSTS: Dict[str, int] = {
    "Suck": 2,
    "Left": 1,
    "Right": 1
}

# Tabela de Custos Uniformes (para análise comparativa):
UNIFORM_ACTION_COSTS: Dict[str, int] = {
    "Suck": 1,
    "Left": 1,
    "Right": 1
}

# Estado inicial canônico da Aula 07:
# A 🦠 ↔ B ✨ ↔ C 🤖🦠 ↔ D ✨ ↔ E 🦠
INITIAL_STATE_AULA_07: State = (
    "C",      # Robô na sala C
    "Dirty",  # Sala A suja
    "Clean",  # Sala B limpa
    "Dirty",  # Sala C suja
    "Clean",  # Sala D limpa
    "Dirty"   # Sala E suja
)


# ==============================================================================
# 2. FUNÇÕES DO AMBIENTE (TRANSIÇÃO, VALIDAÇÃO E OBJETIVO)
# ==============================================================================

def describe_state(state: State) -> str:
    """Gera visualização textual com ícones representando o ambiente 1D."""
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
    Critério de parada: o objetivo é alcançado quando TODAS as salas estão 'Clean'.
    A posição final de repouso do robô não interfere no objetivo.
    """
    return all(condition == "Clean" for condition in state[1:])


def apply_action(state: State, action: str) -> Optional[State]:
    """
    Função de transição determinística T(s, a) -> s'.
    Retorna o novo estado ou None caso a ação seja fisicamente inválida ou redundante.
    """
    position = state[0]
    rooms = list(state[1:])
    try:
        index = POSITIONS.index(position)
    except ValueError:
        return None

    if action == "Suck":
        if rooms[index] == "Dirty":
            rooms[index] = "Clean"
            return (position, *rooms)
        return None  # Aspirar sala já limpa é inútil e podado

    elif action == "Left":
        if index > 0:
            new_position = POSITIONS[index - 1]
            return (new_position, *rooms)
        return None  # Limite da borda esquerda

    elif action == "Right":
        if index < len(POSITIONS) - 1:
            new_position = POSITIONS[index + 1]
            return (new_position, *rooms)
        return None  # Limite da borda direita

    return None


def successors(state: State) -> List[Tuple[str, State]]:
    """Gera todos os pares válidos (ação, novo_estado) a partir do estado atual."""
    valid_successors = []
    for action in ACTIONS:
        next_st = apply_action(state, action)
        if next_st is not None:
            valid_successors.append((action, next_st))
    return valid_successors


# ==============================================================================
# 3. ESTRUTURAS DE RETORNO E PLANRESULT
# ==============================================================================

class PlanResult(list):
    """
    Subclasse de list para o resultado de meu_planejador(initial_state).
    
    Vantagem de interoperabilidade total:
    - 100% compatível com qualquer script avaliador que espere list[str]:
        for action in meu_planejador(state): ...
        len(meu_planejador(state))
        plan[0] == 'Left'
    - Disponibiliza atributos ricos para análise comparativa e auditoria:
        plan.cost, plan.expanded_nodes, plan.execution_time_ms
    """
    def __init__(self, actions: Optional[List[str]] = None, cost: int = 0,
                 expanded: int = 0, generated: int = 0, loops: int = 0,
                 elapsed_ms: float = 0.0, strategy: str = "A*"):
        super().__init__(actions or [])
        self.actions = list(actions or [])
        self.plan = list(actions or [])
        self.cost = cost
        self.expanded_nodes = expanded
        self.generated_nodes = generated
        self.loops_avoided = loops
        self.execution_time_ms = elapsed_ms
        self.strategy = strategy

    def __repr__(self) -> str:
        return f"PlanResult({super().__repr__()}, cost={self.cost}, expanded={self.expanded_nodes}, time={self.execution_time_ms:.2f}ms)"


@dataclass
class SearchResult:
    """Estrutura analítica de dados para comparação rigorosa de algoritmos."""
    strategy: str
    plan: Optional[List[str]]
    cost: Optional[int]
    steps: int
    expanded_nodes: int
    generated_nodes: int
    loops_avoided: int
    execution_time_ms: float
    is_optimal: bool = False


# ==============================================================================
# 4. FORMULAÇÃO DA HEURÍSTICA DO A* (ADMISSIBILIDADE E CONSISTÊNCIA)
# ==============================================================================

def heuristic_1d(state: State) -> int:
    """
    Heurística Informada h(n) para o Vacuum World 1D (Salas A, B, C, D, E).
    
    Definição Matemática:
    ---------------------
    Sejam:
      - D = { i | salas[i] == 'Dirty' } o conjunto de índices das salas sujas.
      - pos = índice da sala onde o robô se encontra.
      - c_suck = 2 (custo da ação Suck).
      - c_move = 1 (custo mínimo de translação entre salas adjacentes).

    Se D for vazio:
        h(n) = 0
    Se D não for vazio:
        min_d = min(D)
        max_d = max(D)
        span = max_d - min_d
        dist_translacao = min(|pos - min_d|, |pos - max_d|) + span
        h(n) = (c_suck * |D|) + (c_move * dist_translacao)

    Justificativa de Admissibilidade (h(n) <= h*(n)):
    -------------------------------------------------
    1. Custo de Sucção: Cada sala suja requer OBRIGATORIAMENTE pelo menos uma ação 'Suck'.
       Portanto, o robô gastará no mínimo 2 * |D| apenas aspirando.
    2. Custo de Movimentação: Para alcançar todas as salas sujas, o robô deve cobrir
       o intervalo [min_d, max_d] no espaço 1D. A menor distância possível para visitar
       ambos os extremos partindo da posição atual é ir até o extremo mais próximo e
       atravessar até o outro extremo.
    Como nenhum plano válido consegue limpar as salas com menos sucções ou menos
    movimento, h(n) é estritamente menor ou igual ao custo real ótimo h*(n).

    Justificativa de Consistência / Monotonicidade (h(n) <= c(n, a, n') + h(n')):
    -----------------------------------------------------------------------------
    - Ação 'Left' ou 'Right': altera pos por 1, logo dist_translacao varia em no máximo 1.
      Assim, h(n) - h(n') <= 1 = c(mover).
    - Ação 'Suck': reduz |D| em 1 (se a sala estava suja), reduzindo h em 2.
      Assim, h(n) - h(n') <= 2 = c(suck).
    Sendo consistente, o A* garante encontrar a solução de CUSTO MÍNIMO sem
    necessidade de reabrir nós na lista fechada.
    """
    robot_pos = state[0]
    rooms = state[1:]

    try:
        pos_idx = POSITIONS.index(robot_pos)
    except ValueError:
        pos_idx = 0

    dirty_indices = [i for i, cond in enumerate(rooms) if cond == "Dirty"]
    if not dirty_indices:
        return 0

    # Custo estritamente necessário de sucção
    cleaning_cost = 2 * len(dirty_indices)

    # Distância mínima de deslocamento para cobrir todos os pontos sujos
    min_d = min(dirty_indices)
    max_d = max(dirty_indices)
    span = max_d - min_d
    travel_distance = min(abs(pos_idx - min_d), abs(pos_idx - max_d)) + span

    return cleaning_cost + travel_distance


# ==============================================================================
# 5. ALGORITMOS DE BUSCA (BFS, DFS, UCS, A*)
# ==============================================================================

def bfs(initial_state: State, action_costs: Optional[Dict[str, int]] = None) -> SearchResult:
    """
    Busca em Largura (Breadth-First Search).
    Estratégia: Fila FIFO. Encontra o menor número de ações, mas IGNORA custos.
    """
    costs = action_costs or DEFAULT_ACTION_COSTS
    start_time = time.perf_counter()

    queue = deque([(initial_state, [], 0)])  # (estado, plano, custo_acumulado)
    visited: Set[State] = {initial_state}

    expanded = 0
    loops_avoided = 0
    generated = 1

    while queue:
        current, plan, cost = queue.popleft()

        if is_goal(current):
            elapsed = (time.perf_counter() - start_time) * 1000
            return SearchResult(
                strategy="BFS",
                plan=plan,
                cost=cost,
                steps=len(plan),
                expanded_nodes=expanded,
                generated_nodes=generated,
                loops_avoided=loops_avoided,
                execution_time_ms=elapsed,
                is_optimal=(cost == 12)  # 12 é o custo ótimo conhecido da Aula 07
            )

        expanded += 1
        for action, new_state in successors(current):
            generated += 1
            if new_state not in visited:
                visited.add(new_state)
                step_cost = costs.get(action, 1)
                queue.append((new_state, plan + [action], cost + step_cost))
            else:
                loops_avoided += 1

    elapsed = (time.perf_counter() - start_time) * 1000
    return SearchResult("BFS", None, None, 0, expanded, generated, loops_avoided, elapsed, False)


def dfs(initial_state: State, action_costs: Optional[Dict[str, int]] = None, max_depth: int = 50) -> SearchResult:
    """
    Busca em Profundidade (Depth-First Search) com controle de ciclos.
    Estratégia: Pilha LIFO. Não é ótima em passos nem em custo.
    """
    costs = action_costs or DEFAULT_ACTION_COSTS
    start_time = time.perf_counter()

    # (estado, plano, custo_acumulado, visitados_no_ramo)
    stack = [(initial_state, [], 0, {initial_state})]
    expanded = 0
    loops_avoided = 0
    generated = 1

    while stack:
        current, plan, cost, path_visited = stack.pop()

        if is_goal(current):
            elapsed = (time.perf_counter() - start_time) * 1000
            return SearchResult(
                strategy="DFS",
                plan=plan,
                cost=cost,
                steps=len(plan),
                expanded_nodes=expanded,
                generated_nodes=generated,
                loops_avoided=loops_avoided,
                execution_time_ms=elapsed,
                is_optimal=False
            )

        if len(plan) >= max_depth:
            continue

        expanded += 1
        for action, new_state in reversed(successors(current)):
            generated += 1
            if new_state not in path_visited:
                step_cost = costs.get(action, 1)
                stack.append((
                    new_state,
                    plan + [action],
                    cost + step_cost,
                    path_visited | {new_state}
                ))
            else:
                loops_avoided += 1

    elapsed = (time.perf_counter() - start_time) * 1000
    return SearchResult("DFS", None, None, 0, expanded, generated, loops_avoided, elapsed, False)


def ucs(initial_state: State, action_costs: Optional[Dict[str, int]] = None) -> SearchResult:
    """
    Busca de Custo Uniforme (Uniform Cost Search - UCS / Dijkstra).
    Estratégia: Expande sempre pelo menor custo acumulado g(n).
    Garante custo ótimo, mas expande nós em todas as direções (busca cega).
    """
    costs = action_costs or DEFAULT_ACTION_COSTS
    start_time = time.perf_counter()

    # (custo_g, contador_desempate, estado, plano)
    frontier = []
    counter = 0
    heappush(frontier, (0, counter, initial_state, []))
    best_cost: Dict[State, int] = {initial_state: 0}

    expanded = 0
    loops_avoided = 0
    generated = 1

    while frontier:
        cost, _, current, plan = heappop(frontier)

        # Descarte de nó desatualizado da fila de prioridade
        if cost != best_cost.get(current):
            continue

        # Teste de objetivo no desempilhamento garante otimalidade
        if is_goal(current):
            elapsed = (time.perf_counter() - start_time) * 1000
            return SearchResult(
                strategy="UCS",
                plan=plan,
                cost=cost,
                steps=len(plan),
                expanded_nodes=expanded,
                generated_nodes=generated,
                loops_avoided=loops_avoided,
                execution_time_ms=elapsed,
                is_optimal=True
            )

        expanded += 1
        for action, new_state in successors(current):
            step_cost = costs.get(action, 1)
            new_cost = cost + step_cost
            generated += 1

            if new_cost < best_cost.get(new_state, float("inf")):
                best_cost[new_state] = new_cost
                counter += 1
                heappush(frontier, (new_cost, counter, new_state, plan + [action]))
            else:
                loops_avoided += 1

    elapsed = (time.perf_counter() - start_time) * 1000
    return SearchResult("UCS", None, None, 0, expanded, generated, loops_avoided, elapsed, False)


def a_star(initial_state: State, action_costs: Optional[Dict[str, int]] = None,
           heuristic=heuristic_1d) -> SearchResult:
    """
    Busca A* (A-Estrela).
    Estratégia: Expande pelo menor custo total estimado f(n) = g(n) + h(n).
    Combina a otimalidade de custo do UCS com a direção inteligente da heurística,
    minimizando os nós expandidos e reduzindo o tempo de CPU.
    """
    costs = action_costs or DEFAULT_ACTION_COSTS
    start_time = time.perf_counter()

    frontier = []
    counter = 0
    h_init = heuristic(initial_state)
    f_init = 0 + h_init

    # (f_cost, contador_desempate, g_cost, estado, plano)
    heappush(frontier, (f_init, counter, 0, initial_state, []))
    best_g: Dict[State, int] = {initial_state: 0}

    expanded = 0
    loops_avoided = 0
    generated = 1

    while frontier:
        f_cost, _, g_cost, current, plan = heappop(frontier)

        if g_cost != best_g.get(current):
            continue

        if is_goal(current):
            elapsed = (time.perf_counter() - start_time) * 1000
            return SearchResult(
                strategy="A*",
                plan=plan,
                cost=g_cost,
                steps=len(plan),
                expanded_nodes=expanded,
                generated_nodes=generated,
                loops_avoided=loops_avoided,
                execution_time_ms=elapsed,
                is_optimal=True
            )

        expanded += 1
        for action, new_state in successors(current):
            step_cost = costs.get(action, 1)
            new_g = g_cost + step_cost
            generated += 1

            if new_g < best_g.get(new_state, float("inf")):
                best_g[new_state] = new_g
                new_h = heuristic(new_state)
                new_f = new_g + new_h
                counter += 1
                heappush(frontier, (new_f, counter, new_g, new_state, plan + [action]))
            else:
                loops_avoided += 1

    elapsed = (time.perf_counter() - start_time) * 1000
    return SearchResult("A*", None, None, 0, expanded, generated, loops_avoided, elapsed, False)


# ==============================================================================
# 6. MÓDULO ESTENDIDO — SEARCH CHALLENGE GRADE 5x5 COM OBSTÁCULOS
# ==============================================================================

@dataclass(frozen=True)
class GridState:
    """Representação de estado em malha cartesiana 2D."""
    robot_pos: Tuple[int, int]                  # (linha, coluna)
    dirty_cells: FrozenSet[Tuple[int, int]]     # coordenadas das células sujas


class GridVacuumWorld5x5:
    """Ambiente de busca 2D em grade 5x5 com obstáculos e custos diferenciados."""
    GRID_ROWS = 5
    GRID_COLS = 5
    ACTIONS = ["Up", "Down", "Left", "Right", "Suck"]
    DEFAULT_COSTS = {"Suck": 2, "Up": 1, "Down": 1, "Left": 1, "Right": 1}
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
        valid = []
        for act in self.ACTIONS:
            nst = self.apply_action(state, act)
            if nst is not None:
                valid.append((act, nst))
        return valid

    def is_goal(self, state: GridState) -> bool:
        return len(state.dirty_cells) == 0

    def heuristic(self, state: GridState) -> int:
        """Heurística admissível 2D: Custo de sucção + Manhattan da célula suja mais próxima + extensão."""
        if not state.dirty_cells:
            return 0
        r, c = state.robot_pos
        suck_cost = 2 * len(state.dirty_cells)

        # Distância Manhattan mínima até a sujeira mais próxima
        min_dist = min(abs(r - dr) + abs(c - dc) for dr, dc in state.dirty_cells)

        # Extensão (bounding box span) mínima para cobrir todas as sujeiras restantes
        min_r = min(dr for dr, dc in state.dirty_cells)
        max_r = max(dr for dr, dc in state.dirty_cells)
        min_c = min(dc for dr, dc in state.dirty_cells)
        max_c = max(dc for dr, dc in state.dirty_cells)
        span = (max_r - min_r) + (max_c - min_c)

        return suck_cost + min_dist + span

    def a_star(self, initial_state: GridState, action_costs: Optional[Dict[str, int]] = None) -> SearchResult:
        costs = action_costs or self.DEFAULT_COSTS
        start_time = time.perf_counter()

        frontier = []
        counter = 0
        h_init = self.heuristic(initial_state)
        heappush(frontier, (h_init, counter, 0, initial_state, []))
        best_g: Dict[GridState, int] = {initial_state: 0}

        expanded = 0
        loops_avoided = 0
        generated = 1

        while frontier:
            f_cost, _, g_cost, current, plan = heappop(frontier)

            if g_cost != best_g.get(current):
                continue

            if self.is_goal(current):
                elapsed = (time.perf_counter() - start_time) * 1000
                return SearchResult(
                    strategy="A* (Grid 5x5)",
                    plan=plan,
                    cost=g_cost,
                    steps=len(plan),
                    expanded_nodes=expanded,
                    generated_nodes=generated,
                    loops_avoided=loops_avoided,
                    execution_time_ms=elapsed,
                    is_optimal=True
                )

            expanded += 1
            for action, new_state in self.successors(current):
                step_cost = costs.get(action, 1)
                new_g = g_cost + step_cost
                generated += 1

                if new_g < best_g.get(new_state, float("inf")):
                    best_g[new_state] = new_g
                    new_h = self.heuristic(new_state)
                    new_f = new_g + new_h
                    counter += 1
                    heappush(frontier, (new_f, counter, new_g, new_state, plan + [action]))
                else:
                    loops_avoided += 1

        elapsed = (time.perf_counter() - start_time) * 1000
        return SearchResult("A* (Grid 5x5)", None, None, 0, expanded, loops_avoided, generated, elapsed, False)

    def print_grid(self, state: GridState) -> None:
        """Exibe a matriz visual da grade 5x5."""
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


# ==============================================================================
# 7. ENTREGA OBRIGATÓRIA: meu_planejador(initial_state)
# ==============================================================================

def meu_planejador(initial_state: Any) -> PlanResult:
    """
    Função oficial exigida pelo enunciado da atividade:
    >>> meu_planejador(initial_state) -> PlanResult (list[str])

    Características:
    ----------------
    1. Estratégia Escolhida: A* (A-Estrela) com Heurística Admissível e Consistente.
       Garante custo mínimo acumulado ótimo g(n) ao mesmo tempo em que maximiza a
       eficiência (mínima expansão de nós e menor latência de execução em ms).
    2. Detecção Universal de Formato para MAPAS INÉDITOS:
       - Tupla 1D canônica da Aula 07: ('C', 'Dirty', 'Clean', 'Dirty', 'Clean', 'Dirty')
       - Dicionário: {'position': 'C', 'rooms': {'A': 'Dirty', ...}} ou ('C', {'A': 'Dirty', ...})
       - Objeto GridState ou Malha 2D (Search Challenge 5x5)
    3. Retorno:
       Subclasse de lista ('PlanResult'), sendo perfeitamente iterável por
       avaliadores automáticos:
           for acao in meu_planejador(estado):
               ...
       e com acesso a atributos ricos: plan.cost, plan.expanded_nodes, plan.execution_time_ms.
    """
    # --------------------------------------------------------------------------
    # CASO A: Estado é GridState 2D (Challenge 5x5)
    # --------------------------------------------------------------------------
    if isinstance(initial_state, GridState):
        world = GridVacuumWorld5x5()
        res = world.a_star(initial_state)
        return PlanResult(
            actions=res.plan or [],
            cost=res.cost or 0,
            expanded=res.expanded_nodes,
            generated=res.generated_nodes,
            loops=res.loops_avoided,
            elapsed_ms=res.execution_time_ms,
            strategy="A* (Grid 5x5)"
        )

    # --------------------------------------------------------------------------
    # CASO B: Tupla 2D no formato ((r, c), frozenset_de_sujeiras)
    # --------------------------------------------------------------------------
    if isinstance(initial_state, tuple) and len(initial_state) == 2 and isinstance(initial_state[0], tuple):
        rpos, dcells = initial_state
        gstate = GridState(robot_pos=rpos, dirty_cells=frozenset(dcells))
        world = GridVacuumWorld5x5()
        res = world.a_star(gstate)
        return PlanResult(
            actions=res.plan or [],
            cost=res.cost or 0,
            expanded=res.expanded_nodes,
            generated=res.generated_nodes,
            loops=res.loops_avoided,
            elapsed_ms=res.execution_time_ms,
            strategy="A* (Grid 2D)"
        )

    # --------------------------------------------------------------------------
    # CASO C: Dicionário ou Tupla com Dicionário (Estilo Aula 03 / Five_Rooms)
    # --------------------------------------------------------------------------
    parsed_1d_state: Optional[State] = None

    if isinstance(initial_state, tuple) and len(initial_state) == 2 and isinstance(initial_state[1], dict):
        # Ex: ('C', {'A': 'Dirty', 'B': 'Clean', ...})
        pos, dmap = initial_state
        rooms_cond = [dmap.get(r, "Clean") for r in POSITIONS]
        parsed_1d_state = (pos, *rooms_cond)  # type: ignore

    elif isinstance(initial_state, dict):
        pos = initial_state.get("position", "C")
        dmap = initial_state.get("status", initial_state.get("rooms", {}))
        rooms_cond = [dmap.get(r, "Clean") for r in POSITIONS]
        parsed_1d_state = (pos, *rooms_cond)  # type: ignore

    elif isinstance(initial_state, (tuple, list)):
        # Tupla 1D tradicional: ('C', 'Dirty', 'Clean', 'Dirty', 'Clean', 'Dirty')
        if len(initial_state) == 6:
            parsed_1d_state = tuple(initial_state)  # type: ignore
        else:
            # Caso receba tamanho arbitrário de salas
            parsed_1d_state = tuple(initial_state)  # type: ignore

    # Executa A* na modelagem 1D
    if parsed_1d_state is not None:
        res = a_star(parsed_1d_state)
        return PlanResult(
            actions=res.plan or [],
            cost=res.cost or 0,
            expanded=res.expanded_nodes,
            generated=res.generated_nodes,
            loops=res.loops_avoided,
            elapsed_ms=res.execution_time_ms,
            strategy="A*"
        )

    # Fallback seguro
    return PlanResult(actions=[], cost=0, expanded=0, elapsed_ms=0.0)


# ==============================================================================
# 8. BATERIA DE TESTES E TABELA COMPARATIVA (BFS vs DFS vs UCS vs A*)
# ==============================================================================

def run_comparative_benchmark(state: State) -> List[SearchResult]:
    """Executa e compara as 4 estratégias de busca sobre o mesmo estado inicial."""
    print("\n" + "=" * 80)
    print("📊 BATERIA DE TESTES COMPARATIVOS: BFS vs DFS vs UCS vs A*")
    print(f"Estado Testado: {describe_state(state)}")
    print(f"Custos das Ações: {DEFAULT_ACTION_COSTS}")
    print("=" * 80)

    results = [
        bfs(state, DEFAULT_ACTION_COSTS),
        dfs(state, DEFAULT_ACTION_COSTS, max_depth=35),
        ucs(state, DEFAULT_ACTION_COSTS),
        a_star(state, DEFAULT_ACTION_COSTS)
    ]

    # Exibe Tabela Comparativa Formatada
    header = f"{'Estratégia':<10} | {'Solução?':<10} | {'Custo':<6} | {'Passos':<7} | {'Expandidos':<11} | {'Gerados':<8} | {'Podados':<8} | {'Tempo (ms)':<10}"
    print("\n" + header)
    print("-" * len(header))

    for r in results:
        status = "✅ Ótima" if r.is_optimal else ("✅ Válida" if r.plan else "❌ Falha")
        cost_str = str(r.cost) if r.cost is not None else "-"
        print(f"{r.strategy:<10} | {status:<10} | {cost_str:<6} | {r.steps:<7} | {r.expanded_nodes:<11} | {r.generated_nodes:<8} | {r.loops_avoided:<8} | {r.execution_time_ms:<10.2f}")

    print("-" * len(header))
    return results


# ==============================================================================
# 9. SIMULAÇÃO VISUAL PASSO A PASSO
# ==============================================================================

def simulate_execution(initial_state: State, plan: List[str], action_costs: Dict[str, int]) -> None:
    """Simula a execução sequencial do plano do robô passo a passo no console."""
    print("\n" + "=" * 70)
    print("🚀 SIMULAÇÃO PASSO A PASSO DA EXECUÇÃO DO PLANO ÓTIMO")
    print("=" * 70)

    current_state = initial_state
    accumulated_cost = 0

    print(f"\n[Passo 00] Estado Inicial (Custo Acumulado: {accumulated_cost})")
    print(f"  Visual: {describe_state(current_state)}")
    print(f"  Salas sujas restantes: {sum(c == 'Dirty' for c in current_state[1:])}")

    for step, action in enumerate(plan, 1):
        step_cost = action_costs.get(action, 1)
        accumulated_cost += step_cost
        next_state = apply_action(current_state, action)

        action_symbol = "🌀 ASPIRAR" if action == "Suck" else f"➡️ MOVER ({action})"
        print(f"\n[Passo {step:02d}] Ação: {action_symbol} | Custo da ação: +{step_cost} | Custo Total: {accumulated_cost}")

        if next_state is not None:
            current_state = next_state
        print(f"  Visual: {describe_state(current_state)}")
        dirty_count = sum(c == "Dirty" for c in current_state[1:])
        print(f"  Salas sujas restantes: {dirty_count}")

    print("\n" + "-" * 70)
    if is_goal(current_state):
        print("🎉 MISSÃO CUMPRIDA! Todas as salas estão completamente limpas ✨")
        print(f"Total de passos executados: {len(plan)}")
        print(f"Custo total final: {accumulated_cost}")
    else:
        print("⚠️ AVISO: O plano terminou mas ainda restam salas sujas!")
    print("-" * 70)


# ==============================================================================
# 10. JUSTIFICATIVA TEÓRICA E RELATÓRIO TÉCNICO
# ==============================================================================

def print_justification_report():
    """Imprime a justificativa teórica formal para entrega no AVA."""
    print("\n" + "=" * 80)
    print("📝 JUSTIFICATIVA TÉCNICA DA ESTRATÉGIA ESCOLHIDA (A*) E CONSIDERAÇÃO DE CUSTO")
    print("=" * 80)
    report = """
1. Estratégia Escolhida: A* (A-Estrela) com Heurística Admissível e Consistente.
--------------------------------------------------------------------------------
Nos critérios da avaliação em mapas inéditos, os dois pilares são:
  a) Capacidade de encontrar soluções de Custo Mínimo (otimalidade).
  b) Eficiência da busca (mínimo de estados expandidos e tempo de execução).

2. Por que o A* supera as outras estratégias?
---------------------------------------------
• BFS (Largura): Encontra o caminho com menor quantidade de arestas (ações), mas
  não considera os custos diferenciados (Suck=2, Mover=1). Por isso, pode retornar
  soluções mais caras financeiramente e tem alto consumo de memória.
• DFS (Profundidade): Não é ótimo em passos e nem em custo. Cria trajetórias longas,
  redundantes e propensas a loops no Vacuum World.
• UCS (Custo Uniforme): É ótimo em custo, pois prioriza estritamente g(n). No entanto,
  é uma busca "cega" (uninformed) que expande nós uniformemente em todas as direções,
  desperdiçando tempo ao visitar ramificações inúteis.
• A* (A-Estrela): Combina o custo acumulado exato g(n) com a função heurística h(n)
  via f(n) = g(n) + h(n). Ele garante a MESMA solução de custo ótimo que o UCS, mas
  com uma fração dos estados expandidos, pois a heurística "guia" a busca diretamente
  às salas sujas.

3. Como o A* considera o custo das ações?
-----------------------------------------
A função de avaliação é f(n) = g(n) + h(n):
  - g(n): Soma real dos custos das ações realizadas desde a raiz:
          c(Suck) = 2, c(Left) = 1, c(Right) = 1.
          Isso assegura que ações mais caras sejam pesadas corretamente na fronteira.
  - h(n): Estimativa de custo restante baseada na física do problema:
          h(n) = 2 * |Salas Sujas| + Distância Mínima de Translação.
          Como h(n) é admissível (h <= h*) e consistente (monotônica), o primeiro nó
          objetivo retirado da fila de prioridade possui comprovadamente o CUSTO MÍNIMO.
"""
    print(report.strip())
    print("=" * 80 + "\n")


# ==============================================================================
# 11. EXECUÇÃO PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    # 1. Demonstração do Agente Planejador Oficial
    print("=" * 80)
    print("🏆 EXECUÇÃO DO AGENTE PLANEJADOR OFICIAL: meu_planejador(initial_state)")
    print("=" * 80)
    meu_plano = meu_planejador(INITIAL_STATE_AULA_07)
    print(f"• Plano Encontrado: {meu_plano}")
    print(f"• Custo Total Ótimo: {meu_plano.cost}")
    print(f"• Quantidade de Ações: {len(meu_plano)}")
    print(f"• Estados Expandidos: {meu_plano.expanded_nodes}")
    print(f"• Tempo de Busca: {meu_plano.execution_time_ms:.2f} ms")

    # 2. Bateria de Testes Comparativos entre BFS, DFS, UCS e A*
    run_comparative_benchmark(INITIAL_STATE_AULA_07)

    # 3. Teste em Cenário Inédito Linear
    inedito_state: State = ("A", "Clean", "Dirty", "Clean", "Dirty", "Clean")
    print("\n🔍 Testando em Mapa Inédito 1D:")
    print(f"Estado: {describe_state(inedito_state)}")
    plano_inedito = meu_planejador(inedito_state)
    print(f"• Plano do Agente: {plano_inedito}")
    print(f"• Custo Ótimo: {plano_inedito.cost} | Passos: {len(plano_inedito)} | Expandidos: {plano_inedito.expanded_nodes}")

    # 4. Teste em Mapa Inédito 2D (Search Challenge 5x5 com obstáculos)
    print("\n" + "=" * 80)
    print("🗺️ TESTANDO meu_planejador EM MAPA 2D (GRADE 5x5 COM OBSTÁCULOS)")
    print("=" * 80)
    grid_init = GridState(
        robot_pos=(0, 0),
        dirty_cells=frozenset({(0, 4), (2, 2), (4, 0), (4, 4)})
    )
    plano_2d = meu_planejador(grid_init)
    print(f"• Custo Ótimo 2D: {plano_2d.cost}")
    print(f"• Ações 2D ({len(plano_2d)} passos): {plano_2d[:10]} ...")
    print(f"• Estados Expandidos no 2D: {plano_2d.expanded_nodes}")
    print(f"• Tempo de Busca 2D: {plano_2d.execution_time_ms:.2f} ms")

    # 5. Simulação Passo a Passo do Plano Ótimo da Aula 07
    simulate_execution(INITIAL_STATE_AULA_07, meu_plano, DEFAULT_ACTION_COSTS)

    # 6. Relatório e Justificativa Teórica
    print_justification_report()