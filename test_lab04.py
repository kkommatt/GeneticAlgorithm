"""Tests for lab 4."""

import pytest

from labs.lab04_shortest_paths import (
    TABLE_1,
    cost_weights,
    insert_vertex,
    is_simple_path,
    network_matrix,
    path_cost,
    paths_from_source,
    random_simple_path,
    relax,
    run_path_ga,
    should_stop,
    splice,
)

# Independent enumeration, so the relaxation is not checked against itself.
def exhaustive_distances(matrix, source: int) -> dict[int, int]:
    live = [v for v in range(len(matrix)) if any(w is not None for w in matrix[v])]
    best = {vertex: None for vertex in live}
    best[source] = 0

    def walk(path, cost):
        for nxt in live:
            weight = matrix[path[-1]][nxt]
            if weight is None or nxt in path:
                continue
            nxt_cost = cost + weight
            if best[nxt] is None or nxt_cost < best[nxt]:
                best[nxt] = nxt_cost
            walk(path + [nxt], nxt_cost)

    walk([source], 0)
    return {vertex: cost for vertex, cost in best.items() if cost is not None}


EXACT_FROM_1 = {1: 0, 2: 20, 3: 20, 4: 25, 5: 31, 6: 38, 7: 39}


def test_table_1_costs_match_the_matrix():
    matrix = network_matrix()
    for route, expected in TABLE_1:
        assert path_cost(route, matrix) == expected


def test_relaxation_matches_exhaustive_search():
    matrix = network_matrix()
    dist = relax(matrix, source=1)
    exhaustive = exhaustive_distances(matrix, 1)
    for vertex, cost in EXACT_FROM_1.items():
        assert dist[vertex] == cost
        assert exhaustive[vertex] == cost
    assert path_cost((1, 3, 5, 7), matrix) == 39
    assert path_cost((1, 2, 5, 7), matrix) == 42


def test_directed_relaxation_prefers_the_two_edge_path():
    matrix = [
        [None, 1, 4],
        [None, None, 1],
        [None, None, None],
    ]
    dist = relax(matrix, source=0)
    assert dist == [0, 1, 2]


def test_initialization_and_one_round():
    matrix = network_matrix()
    after_zero = relax(matrix, source=1, rounds=0)
    assert after_zero[1] == 0
    assert after_zero[2] == float("inf")
    after_one = relax(matrix, source=1, rounds=1)
    assert after_one[2] == 20
    assert after_one[5] == float("inf")


def test_splice_at_the_shared_vertex_3():
    child_a, child_b = splice([1, 2, 3, 5, 7], [1, 3, 4, 6, 7], 3)
    assert child_a == [1, 2, 3, 4, 6, 7]
    assert child_b == [1, 3, 5, 7]
    matrix = network_matrix()
    assert path_cost(child_a, matrix) == 67
    assert path_cost(child_b, matrix) == 39


def test_insertion_mutation_example():
    child = insert_vertex([1, 2, 3, 7], 5, 3)
    assert child == [1, 2, 3, 5, 7]
    assert path_cost(child, network_matrix()) == 54


def test_invalid_paths_are_rejected():
    matrix = network_matrix()
    assert not is_simple_path([1, 2, 2, 7], matrix, 1, 7)
    assert not is_simple_path([1, 2, 7], matrix, 1, 7)
    assert not is_simple_path([2, 5, 7], matrix, 1, 7)
    assert not is_simple_path([1, 2, 5], matrix, 1, 7)
    with pytest.raises(ValueError):
        path_cost([1, 2, 7], matrix)
    with pytest.raises(ValueError):
        splice([1, 2, 5, 7], [1, 3, 4, 6, 7], 4)


def test_shorter_path_gets_the_larger_roulette_weight():
    weights = cost_weights([39, 42, 46])
    assert weights[0] > weights[1] > weights[2]
    assert weights[0] == pytest.approx(1 / 39)
    with pytest.raises(ValueError):
        cost_weights([0])


def test_random_walk_returns_a_simple_path():
    import random

    matrix = network_matrix()
    path = random_simple_path(matrix, 1, 7, random.Random(1))
    assert is_simple_path(path, matrix, 1, 7)


def test_stop_on_either_limit():
    assert should_stop(generation=10, max_generations=100, stagnant=10)
    assert should_stop(generation=100, max_generations=100, stagnant=0)
    assert not should_stop(generation=9, max_generations=100, stagnant=9)


def test_ga_recovers_every_distance_from_vertex_1():
    matrix = network_matrix()
    found = paths_from_source(matrix, source=1, seed=1)
    assert set(found) == {2, 3, 4, 5, 6, 7}
    for target, result in found.items():
        assert is_simple_path(result.path, matrix, 1, target)
        assert result.cost == path_cost(result.path, matrix)
        assert result.cost == EXACT_FROM_1[target]
        assert result.generations_run <= 100


def test_single_run_to_vertex_7_finds_39():
    result = run_path_ga(network_matrix(), 1, 7, population_size=15, max_generations=100, seed=1)
    assert result.cost == 39
    assert result.path[0] == 1 and result.path[-1] == 7


def test_reject_bad_ga_arguments():
    matrix = network_matrix()
    with pytest.raises(ValueError):
        run_path_ga(matrix, 1, 1)
    with pytest.raises(ValueError):
        run_path_ga(matrix, 1, 7, population_size=1)
