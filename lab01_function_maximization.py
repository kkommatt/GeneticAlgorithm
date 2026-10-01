"""Lab 1. Maximize f(w, x, y, z) with a genetic algorithm.

A chromosome is 16 bits: four unsigned 4-bit fields, in the order w, x, y, z.
The domain of each variable is the integers 0 through 15.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

GENE_BITS = 4
CHROMOSOME_LENGTH = 16
DEFAULT_POPULATION = 41
DEFAULT_GENERATIONS = 200
BIT_MUTATION_PROBABILITY = 1 / 16


def objective(w: int, x: int, y: int, z: int) -> int:
    return (
        w**3
        + x**2
        - y**2
        - z**2
        + 2 * y * z
        - 3 * w * x
        + w * z
        - x * y
        + 2
    )


def decode(chromosome: str) -> tuple[int, int, int, int]:
    if len(chromosome) != CHROMOSOME_LENGTH or any(bit not in "01" for bit in chromosome):
        raise ValueError("a chromosome must be 16 characters, each '0' or '1'")
    values = [
        int(chromosome[i : i + GENE_BITS], 2)
        for i in range(0, CHROMOSOME_LENGTH, GENE_BITS)
    ]
    return values[0], values[1], values[2], values[3]


def encode(w: int, x: int, y: int, z: int) -> str:
    for name, value in (("w", w), ("x", x), ("y", y), ("z", z)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 15:
            raise ValueError(f"{name} must be an integer in 0..15")
    return "".join(f"{value:04b}" for value in (w, x, y, z))


def fitness(chromosome: str) -> int:
    return objective(*decode(chromosome))


def ordinal_ranks(values: list[int]) -> list[int]:
    """Rank 1 is the worst value. Equal values take successive ranks in input order."""
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0] * len(values)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks


def mean_evaluation(values: list[int]) -> float:
    if not values:
        raise ValueError("mean of an empty list is undefined")
    return sum(values) / len(values)


def _bit_parents(parent_a: str, parent_b: str) -> None:
    if not parent_a or len(parent_a) != len(parent_b):
        raise ValueError("parents must be non-empty and of equal length")
    if any(bit not in "01" for bit in parent_a + parent_b):
        raise ValueError("parents must be bit strings")


def one_point_crossover(parent_a: str, parent_b: str, cut: int) -> tuple[str, str]:
    """Cut lies in [0, l-1]. The prefix of that length stays with its parent."""
    _bit_parents(parent_a, parent_b)
    length = len(parent_a)
    if not 0 <= cut <= length - 1:
        raise ValueError(f"cut must lie in 0..{length - 1}")
    return parent_a[:cut] + parent_b[cut:], parent_b[:cut] + parent_a[cut:]


def two_point_crossover(parent_a: str, parent_b: str, left: int, right: int) -> tuple[str, str]:
    """Swap the half-open segment [left, right)."""
    _bit_parents(parent_a, parent_b)
    length = len(parent_a)
    if not 0 <= left < right <= length:
        raise ValueError("require 0 <= left < right <= length")
    child_a = parent_a[:left] + parent_b[left:right] + parent_a[right:]
    child_b = parent_b[:left] + parent_a[left:right] + parent_b[right:]
    return child_a, child_b


def n_point_crossover(parent_a: str, parent_b: str, cuts: list[int]) -> tuple[str, str]:
    """Alternate segments at the given cut positions. Cuts lie in 1..l-1."""
    _bit_parents(parent_a, parent_b)
    length = len(parent_a)
    points = sorted(set(cuts))
    if any(point <= 0 or point >= length for point in points):
        raise ValueError(f"each cut must lie in 1..{length - 1}")
    boundaries = [0, *points, length]
    parts_a: list[str] = []
    parts_b: list[str] = []
    for segment, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if segment % 2 == 0:
            parts_a.append(parent_a[start:end])
            parts_b.append(parent_b[start:end])
        else:
            parts_a.append(parent_b[start:end])
            parts_b.append(parent_a[start:end])
    return "".join(parts_a), "".join(parts_b)


def flip_bit(chromosome: str, index: int) -> str:
    if not chromosome or any(bit not in "01" for bit in chromosome):
        raise ValueError("chromosome must be a non-empty bit string")
    if not 0 <= index < len(chromosome):
        raise ValueError("bit index out of range")
    flipped = "1" if chromosome[index] == "0" else "0"
    return chromosome[:index] + flipped + chromosome[index + 1 :]


def mutate(
    chromosome: str,
    rng: random.Random,
    probability: float = BIT_MUTATION_PROBABILITY,
) -> str:
    chars = list(chromosome)
    for index, bit in enumerate(chars):
        if rng.random() < probability:
            chars[index] = "1" if bit == "0" else "0"
    return "".join(chars)


def proportionate_index(weights: list[float], rng: random.Random) -> int:
    """Roulette index. Every weight must be strictly positive."""
    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError("roulette weights must be positive")
    total = sum(weights)
    threshold = rng.random() * total
    running = 0.0
    for index, weight in enumerate(weights):
        running += weight
        if threshold < running:
            return index
    return len(weights) - 1


def selection_weights(fitnesses: list[int]) -> list[float]:
    """Shift fitness so the worst individual has weight 1.

    f takes negative values, and a raw roulette wheel is undefined on them.
    The gap above the worst individual is preserved, so a better individual
    still receives a larger slice.
    """
    if not fitnesses:
        raise ValueError("empty population")
    worst = min(fitnesses)
    return [float(value - worst + 1) for value in fitnesses]


def random_chromosome(rng: random.Random) -> str:
    return "".join(rng.choice("01") for _ in range(CHROMOSOME_LENGTH))


@dataclass(frozen=True)
class SearchResult:
    chromosome: str
    variables: tuple[int, int, int, int]
    fitness: int
    best_by_generation: tuple[int, ...]


def run_ga(
    population_size: int = DEFAULT_POPULATION,
    generations: int = DEFAULT_GENERATIONS,
    seed: int = 1,
    mutation_probability: float = BIT_MUTATION_PROBABILITY,
) -> SearchResult:
    """Fitness-proportionate selection, one-point crossover, bit mutation.

    The current best chromosome is copied unchanged into the next population,
    so the best fitness cannot fall.
    """
    if population_size < 2:
        raise ValueError("population size must be at least 2")
    if generations < 1:
        raise ValueError("generations must be at least 1")
    rng = random.Random(seed)
    population = [random_chromosome(rng) for _ in range(population_size)]
    history: list[int] = []
    for _ in range(generations):
        scores = [fitness(individual) for individual in population]
        history.append(max(scores))
        best = population[scores.index(max(scores))]
        weights = selection_weights(scores)
        nxt = [best]
        while len(nxt) < population_size:
            first = population[proportionate_index(weights, rng)]
            second = population[proportionate_index(weights, rng)]
            cut = rng.randrange(0, CHROMOSOME_LENGTH)
            child_a, child_b = one_point_crossover(first, second, cut)
            nxt.append(mutate(child_a, rng, mutation_probability))
            if len(nxt) < population_size:
                nxt.append(mutate(child_b, rng, mutation_probability))
        population = nxt
    scores = [fitness(individual) for individual in population]
    history.append(max(scores))
    best_index = max(range(population_size), key=lambda i: scores[i])
    best = population[best_index]
    return SearchResult(best, decode(best), scores[best_index], tuple(history))


def steepest_ascent(eval_budget: int, seed: int = 1) -> SearchResult:
    """One-bit steepest ascent with random restarts.

    From the current string, flip each bit from left to right. Move to the
    neighbor with the largest strict gain. When no neighbor improves, keep
    the string as a hilltop and start from a new random string. Stop when
    the evaluation budget is spent. Return the highest hilltop.
    """
    if eval_budget < 1:
        raise ValueError("evaluation budget must be at least 1")
    rng = random.Random(seed)
    evaluations = 0
    best_chromosome = ""
    best_score: int | None = None

    def evaluate(chromosome: str) -> int:
        nonlocal evaluations
        evaluations += 1
        return fitness(chromosome)

    while evaluations < eval_budget:
        current = random_chromosome(rng)
        current_score = evaluate(current)
        if best_score is None or current_score > best_score:
            best_chromosome = current
            best_score = current_score
        while evaluations < eval_budget:
            neighbor = ""
            neighbor_score = current_score
            gain = 0
            for index in range(CHROMOSOME_LENGTH):
                if evaluations >= eval_budget:
                    break
                candidate = flip_bit(current, index)
                candidate_score = evaluate(candidate)
                improvement = candidate_score - current_score
                if improvement > gain:
                    gain = improvement
                    neighbor = candidate
                    neighbor_score = candidate_score
            if gain <= 0:
                break
            current = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best_chromosome = current
                best_score = current_score
    if best_score is None:
        raise RuntimeError("budget produced no evaluation")
    return SearchResult(best_chromosome, decode(best_chromosome), best_score, ())


def brute_force_maximum() -> SearchResult:
    """Scan all 65536 chromosomes. Used as the exact reference for this domain."""
    best_chromosome = "0" * CHROMOSOME_LENGTH
    best_score = fitness(best_chromosome)
    for value in range(1, 2**CHROMOSOME_LENGTH):
        chromosome = f"{value:016b}"
        score = fitness(chromosome)
        if score > best_score:
            best_chromosome = chromosome
            best_score = score
    return SearchResult(best_chromosome, decode(best_chromosome), best_score, ())
