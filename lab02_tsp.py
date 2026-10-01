"""Lab 2. Genetic algorithm for the travelling salesman problem.

The instance is a complete undirected graph on n >= 3 cities numbered 0..n-1.
A chromosome is a permutation that starts at city 0. The tour returns to 0.
The reported objective is the distance sum. A smaller sum is better.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import factorial

DEFAULT_POPULATION = 30
DEFAULT_GENERATIONS = 300
DEFAULT_MUTATION_RATE = 0.1

# Five-city lab instance. This is not the untitled figure whose tours cost 33.
LAB_DISTANCE = (
    (0, 2, 9, 10, 7),
    (2, 0, 6, 4, 3),
    (9, 6, 0, 8, 5),
    (10, 4, 8, 0, 6),
    (7, 3, 5, 6, 0),
)


def chromosome_space(n: int) -> int:
    """Permutations of the cities after a fixed start: (n-1)!."""
    if n < 3:
        raise ValueError("the TSP statement requires n >= 3")
    return factorial(n - 1)


def undirected_tour_count(n: int) -> int:
    """Distinct undirected cycles: (n-1)!/2."""
    if n < 3:
        raise ValueError("the TSP statement requires n >= 3")
    return factorial(n - 1) // 2


def validate_distance(matrix: tuple[tuple[int, ...], ...] | list[list[int]]) -> int:
    n = len(matrix)
    if n < 3:
        raise ValueError("the TSP statement requires n >= 3")
    for i, row in enumerate(matrix):
        if len(row) != n:
            raise ValueError("distance matrix must be square")
        for j, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("distances must be integers")
            if value < 0:
                raise ValueError("distances must be nonnegative")
            if value != matrix[j][i]:
                raise ValueError("an undirected instance needs a symmetric matrix")
        if matrix[i][i] != 0:
            raise ValueError("the diagonal must be 0")
    return n


def tour_cost(tour: tuple[int, ...] | list[int], matrix) -> int:
    n = validate_distance(matrix)
    if sorted(tour) != list(range(n)):
        raise ValueError("a tour must visit each city exactly once")
    total = 0
    for origin, destination in zip(tour, list(tour)[1:]):
        total += matrix[origin][destination]
    total += matrix[tour[-1]][tour[0]]
    return total


def order_crossover(
    parent_a: list[int],
    parent_b: list[int],
    left: int,
    right: int,
) -> tuple[list[int], list[int]]:
    """Rewrite each cut segment in the relative order of the other parent.

    left and right are inclusive. The worked cut is left=2, right=5 on
    [7, 3, 6, 1, 0, 2, 5, 4] and [5, 4, 1, 7, 2, 3, 6, 0].
    """
    if len(parent_a) != len(parent_b) or len(parent_a) < 3:
        raise ValueError("parents must be equal-length tours of at least 3 cities")
    if sorted(parent_a) != sorted(parent_b):
        raise ValueError("parents must contain the same cities")
    n = len(parent_a)
    if not 0 <= left <= right < n:
        raise ValueError("require 0 <= left <= right < n")

    def recombine(parent: list[int], donor: list[int]) -> list[int]:
        wanted = set(parent[left : right + 1])
        ordered = [city for city in donor if city in wanted]
        child = list(parent)
        child[left : right + 1] = ordered
        return child

    return recombine(parent_a, parent_b), recombine(parent_b, parent_a)


def adjacent_swap(tour: list[int], index: int) -> list[int]:
    """Swap the city at index with the next city. The next of the last city is the first."""
    if len(tour) < 2:
        raise ValueError("tour is too short to swap")
    if not 0 <= index < len(tour):
        raise ValueError("index out of range")
    child = list(tour)
    nxt = (index + 1) % len(tour)
    child[index], child[nxt] = child[nxt], child[index]
    return child


def swap_cities(tour: list[int], i: int, j: int) -> list[int]:
    if not 0 <= i < len(tour) or not 0 <= j < len(tour):
        raise ValueError("index out of range")
    child = list(tour)
    child[i], child[j] = child[j], child[i]
    return child


def reverse_substring(tour: list[int], left: int, right: int) -> list[int]:
    """Reverse the inclusive segment [left, right]."""
    if not 0 <= left <= right < len(tour):
        raise ValueError("require 0 <= left <= right < n")
    child = list(tour)
    child[left : right + 1] = reversed(child[left : right + 1])
    return child


def canonicalize(tour: list[int]) -> tuple[int, ...]:
    """Rotate a cyclic tour so that city 0 is in front."""
    if 0 not in tour:
        raise ValueError("city 0 is the fixed start")
    start = tour.index(0)
    rotated = tour[start:] + tour[:start]
    return tuple(rotated)


def two_best_indices(costs: list[int]) -> tuple[int, int]:
    if len(costs) < 2:
        raise ValueError("need at least two individuals")
    order = sorted(range(len(costs)), key=lambda i: (costs[i], i))
    return order[0], order[1]


def two_worst_indices(costs: list[int]) -> tuple[int, int]:
    order = sorted(range(len(costs)), key=lambda i: (costs[i], i), reverse=True)
    return order[0], order[1]


def random_tour(n: int, rng: random.Random) -> tuple[int, ...]:
    cities = list(range(1, n))
    rng.shuffle(cities)
    return (0, *cities)


def _mutate(tour: list[int], rng: random.Random, mutation_rate: float) -> list[int]:
    if rng.random() >= mutation_rate:
        return tour
    return adjacent_swap(tour, rng.randrange(len(tour)))


@dataclass(frozen=True)
class TspResult:
    tour: tuple[int, ...]
    cost: int
    best_by_generation: tuple[int, ...]


def run_tsp(
    matrix=LAB_DISTANCE,
    population_size: int = DEFAULT_POPULATION,
    generations: int = DEFAULT_GENERATIONS,
    seed: int = 1,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
) -> TspResult:
    """Each generation crosses the two shortest tours and writes the offspring over the two longest.

    The two best parents stay in the population, so the next generation can select them again.
    The returned tour is the shortest seen, including the initial population.
    """
    n = validate_distance(matrix)
    if population_size < 2:
        raise ValueError("population size must be at least 2")
    if generations < 1:
        raise ValueError("generations must be at least 1")
    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation rate must lie in [0, 1]")
    rng = random.Random(seed)
    population = [list(random_tour(n, rng)) for _ in range(population_size)]
    history: list[int] = []
    best_tour = min(population, key=lambda tour: tour_cost(tour, matrix))
    best_cost = tour_cost(best_tour, matrix)
    for _ in range(generations):
        costs = [tour_cost(tour, matrix) for tour in population]
        history.append(min(costs))
        i_best, j_best = two_best_indices(costs)
        left = rng.randrange(0, n)
        right = rng.randrange(0, n)
        if left > right:
            left, right = right, left
        child_a, child_b = order_crossover(population[i_best], population[j_best], left, right)
        child_a = _mutate(child_a, rng, mutation_rate)
        child_b = _mutate(child_b, rng, mutation_rate)
        i_worst, j_worst = two_worst_indices(costs)
        population[i_worst] = child_a
        population[j_worst] = child_b
        for child in (child_a, child_b):
            cost = tour_cost(child, matrix)
            if cost < best_cost:
                best_tour = list(child)
                best_cost = cost
    costs = [tour_cost(tour, matrix) for tour in population]
    history.append(min(costs))
    incumbent = min(population, key=lambda tour: tour_cost(tour, matrix))
    if tour_cost(incumbent, matrix) < best_cost:
        best_tour = list(incumbent)
        best_cost = tour_cost(incumbent, matrix)
    return TspResult(canonicalize(best_tour), best_cost, tuple(history))


def brute_force_best(matrix=LAB_DISTANCE) -> TspResult:
    """Minimum tour among the (n-1)! chromosomes that start at city 0."""
    n = validate_distance(matrix)
    best_tour: tuple[int, ...] = (0, *range(1, n))
    best_cost = tour_cost(best_tour, matrix)

    def walk(prefix: list[int], remaining: set[int]) -> None:
        nonlocal best_tour, best_cost
        if not remaining:
            cost = tour_cost(prefix, matrix)
            if cost < best_cost:
                best_tour = tuple(prefix)
                best_cost = cost
            return
        for city in sorted(remaining):
            walk(prefix + [city], remaining - {city})

    walk([0], set(range(1, n)))
    return TspResult(best_tour, best_cost, ())


def cost_33_without_matrix() -> None:
    """Two tours are stated to cost 33, and no distance matrix is given for them."""
    raise LookupError("tours of cost 33 are stated without a distance matrix")
