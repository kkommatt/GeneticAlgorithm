"""Lab 4. Shortest paths from one vertex to every vertex.

The exact method is synchronous relaxation. A chromosome is a simple path
from the source to one target. The sentinel 1000 marks a missing edge and is
not traversable. Selection weight is 1/cost.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from labs.lab01_function_maximization import proportionate_index

MISSING = 1000
STAGNATION_LIMIT = 10
DEFAULT_POPULATION = 15
DEFAULT_GENERATIONS = 100
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1

# Rows and columns are vertices 1..7. Index 0 is unused.
# A stored 1000 means the edge does not exist. The diagonal is a non-edge.
_PAPER_ROWS = (
    (1000, 20, 20, 25, 1000, 1000, 1000),
    (20, 1000, 15, 1000, 14, 1000, 1000),
    (20, 15, 1000, 12, 11, 1000, 26),
    (25, 1000, 12, 1000, 1000, 13, 1000),
    (1000, 14, 11, 1000, 1000, 1000, 8),
    (1000, 1000, 1000, 13, 1000, 1000, 7),
    (1000, 1000, 26, 1000, 8, 7, 1000),
)

# Recorded routes. Several of them are longer than the minimum.
TABLE_1 = (
    ((1, 4, 6, 7), 45),
    ((1, 3, 7), 46),
    ((1, 2, 5, 7), 42),
    ((1, 2, 3, 7), 61),
    ((1, 2, 3, 5, 7), 54),
    ((1, 2, 3, 4, 6, 7), 67),
    ((1, 4, 3, 7), 63),
    ((1, 3, 4, 6, 7), 52),
    ((1, 4, 3, 5, 7), 56),
    ((1, 2, 5, 3, 7), 71),
    ((1, 2, 5, 3, 4, 6, 7), 77),
)


def network_matrix() -> list[list[int | None]]:
    """1-indexed matrix. None is a missing edge, including the diagonal."""
    size = 8
    matrix: list[list[int | None]] = [[None] * size for _ in range(size)]
    for i, row in enumerate(_PAPER_ROWS, start=1):
        for j, weight in enumerate(row, start=1):
            matrix[i][j] = None if weight == MISSING else weight
    return matrix


def graph_vertices(matrix: list[list[int | None]]) -> list[int]:
    """Every index that is the endpoint of at least one existing edge."""
    found: set[int] = set()
    for origin, row in enumerate(matrix):
        for destination, weight in enumerate(row):
            if weight is not None:
                found.add(origin)
                found.add(destination)
    return sorted(found)


def neighbors(matrix: list[list[int | None]], vertex: int) -> list[int]:
    return [dst for dst, weight in enumerate(matrix[vertex]) if weight is not None]


def path_cost(path: tuple[int, ...] | list[int], matrix: list[list[int | None]]) -> int:
    if len(path) < 2:
        raise ValueError("a path needs two vertices")
    total = 0
    for origin, destination in zip(path, path[1:]):
        weight = matrix[origin][destination]
        if weight is None:
            raise ValueError(f"no edge {origin}->{destination}")
        total += weight
    return total


def is_simple_path(
    path: tuple[int, ...] | list[int],
    matrix: list[list[int | None]],
    source: int,
    target: int,
) -> bool:
    if len(path) < 2 or path[0] != source or path[-1] != target:
        return False
    if len(path) != len(set(path)):
        return False
    for origin, destination in zip(path, path[1:]):
        if matrix[origin][destination] is None:
            return False
    return True


def relax(matrix: list[list[int | None]], source: int, rounds: int | None = None) -> list[float]:
    """Synchronous relaxation. After |V| rounds, D[v] is the shortest-path cost.

    Vertices are 1..n-1 when index 0 is unused, or 0..n-1 when every row is live.
    A row that is entirely missing is ignored, so the 7-vertex network (live
    vertices 1..7) and a compact 0-indexed matrix both work.
    """
    n = len(matrix)
    live = graph_vertices(matrix)
    if source not in live and source != 0:
        raise ValueError("source is not a vertex")
    if rounds is None:
        rounds = len(live)
    dist = [float("inf")] * n
    dist[source] = 0.0
    for _ in range(rounds):
        updated = list(dist)
        for vertex in range(n):
            incoming = [
                dist[origin] + weight
                for origin in range(n)
                if (weight := matrix[origin][vertex]) is not None and dist[origin] < float("inf")
            ]
            if incoming:
                updated[vertex] = min(dist[vertex], min(incoming))
        dist = updated
    return dist


def splice(parent_a: list[int], parent_b: list[int], node: int) -> tuple[list[int], list[int]]:
    """Exchange the suffixes that follow a shared vertex."""
    if node not in parent_a or node not in parent_b:
        raise ValueError("the splice vertex must lie on both paths")
    i = parent_a.index(node)
    j = parent_b.index(node)
    child_a = parent_a[: i + 1] + parent_b[j + 1 :]
    child_b = parent_b[: j + 1] + parent_a[i + 1 :]
    return child_a, child_b


def insert_vertex(path: list[int], vertex: int, index: int) -> list[int]:
    """Insert vertex so that it occupies `index`. The example uses index 3 on [1, 2, 3, 7]."""
    if not 0 <= index <= len(path):
        raise ValueError("insertion index out of range")
    return path[:index] + [vertex] + path[index:]


def random_simple_path(
    matrix: list[list[int | None]],
    source: int,
    target: int,
    rng: random.Random,
) -> list[int]:
    """Random walk from source to target. Backtrack when the walk cannot extend."""
    path = [source]
    tried: dict[int, set[int]] = {source: set()}
    while path[-1] != target:
        current = path[-1]
        candidates = [
            vertex
            for vertex in neighbors(matrix, current)
            if vertex not in path and vertex not in tried[current]
        ]
        if not candidates:
            if len(path) == 1:
                raise ValueError(f"no simple path from {source} to {target}")
            dead = path.pop()
            tried.pop(dead, None)
            tried[path[-1]].add(dead)
            continue
        nxt = rng.choice(candidates)
        path.append(nxt)
        tried[nxt] = set()
    return path


def _accept(path: list[int], matrix, source: int, target: int, fallback: list[int]) -> list[int]:
    if is_simple_path(path, matrix, source, target):
        return path
    return list(fallback)


def crossover_paths(
    parent_a: list[int],
    parent_b: list[int],
    matrix,
    source: int,
    target: int,
    rng: random.Random,
    rate: float = CROSSOVER_RATE,
) -> tuple[list[int], list[int]]:
    if rng.random() >= rate:
        return list(parent_a), list(parent_b)
    shared = [vertex for vertex in parent_a if vertex in set(parent_b)]
    rng.shuffle(shared)
    for node in shared:
        raw_a, raw_b = splice(parent_a, parent_b, node)
        child_a = raw_a if is_simple_path(raw_a, matrix, source, target) else list(parent_a)
        child_b = raw_b if is_simple_path(raw_b, matrix, source, target) else list(parent_b)
        if child_a != list(parent_a) or child_b != list(parent_b):
            return child_a, child_b
    return list(parent_a), list(parent_b)


def mutate_path(
    path: list[int],
    matrix,
    source: int,
    target: int,
    rng: random.Random,
    rate: float = MUTATION_RATE,
) -> list[int]:
    """Insert one vertex at an interior index. Keep the parent when the result is illegal."""
    if rng.random() >= rate:
        return list(path)
    vertices = graph_vertices(matrix)
    vertex = rng.choice(vertices)
    index = rng.randrange(1, len(path))
    return _accept(insert_vertex(path, vertex, index), matrix, source, target, path)


def cost_weights(costs: list[int]) -> list[float]:
    """Roulette weight is the reciprocal of the path cost. A shorter path gets a larger slice."""
    if not costs or any(cost <= 0 for cost in costs):
        raise ValueError("path costs must be positive")
    return [1.0 / cost for cost in costs]


def should_stop(generation: int, max_generations: int, stagnant: int, limit: int = STAGNATION_LIMIT) -> bool:
    """Either limit ends the run. generation is counted from 1."""
    return generation >= max_generations or stagnant >= limit


@dataclass(frozen=True)
class PathResult:
    path: tuple[int, ...]
    cost: int
    generations_run: int


def run_path_ga(
    matrix: list[list[int | None]],
    source: int,
    target: int,
    population_size: int = DEFAULT_POPULATION,
    max_generations: int = DEFAULT_GENERATIONS,
    seed: int = 1,
) -> PathResult:
    if population_size < 2:
        raise ValueError("population size must be at least 2")
    if max_generations < 1:
        raise ValueError("max generations must be at least 1")
    if source == target:
        raise ValueError("source and target must differ")
    rng = random.Random(seed)
    population = [random_simple_path(matrix, source, target, rng) for _ in range(population_size)]
    best = min(population, key=lambda path: path_cost(path, matrix))
    best_cost = path_cost(best, matrix)
    stagnant = 0
    generations_run = 0
    for generation in range(1, max_generations + 1):
        generations_run = generation
        costs = [path_cost(path, matrix) for path in population]
        weights = cost_weights(costs)
        nxt: list[list[int]] = [list(best)]
        while len(nxt) < population_size:
            first = population[proportionate_index(weights, rng)]
            second = population[proportionate_index(weights, rng)]
            child_a, child_b = crossover_paths(first, second, matrix, source, target, rng)
            child_a = mutate_path(child_a, matrix, source, target, rng)
            child_b = mutate_path(child_b, matrix, source, target, rng)
            nxt.append(child_a)
            if len(nxt) < population_size:
                nxt.append(child_b)
        population = nxt
        generation_best = min(population, key=lambda path: path_cost(path, matrix))
        generation_cost = path_cost(generation_best, matrix)
        if generation_cost < best_cost:
            best = list(generation_best)
            best_cost = generation_cost
            stagnant = 0
        else:
            stagnant += 1
        if should_stop(generation, max_generations, stagnant):
            break
    return PathResult(tuple(best), best_cost, generations_run)


def paths_from_source(
    matrix: list[list[int | None]] | None = None,
    source: int = 1,
    seed: int = 1,
) -> dict[int, PathResult]:
    """One genetic search per target on the 7-vertex network."""
    if matrix is None:
        matrix = network_matrix()
    live = [v for v in graph_vertices(matrix) if v != source]
    return {
        target: run_path_ga(matrix, source, target, seed=seed + target)
        for target in live
    }
