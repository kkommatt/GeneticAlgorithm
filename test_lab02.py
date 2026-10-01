"""Tests for lab 2."""

import pytest

from labs.lab02_tsp import (
    DEFAULT_MUTATION_RATE,
    LAB_DISTANCE,
    adjacent_swap,
    brute_force_best,
    chromosome_space,
    cost_33_without_matrix,
    order_crossover,
    reverse_substring,
    run_tsp,
    swap_cities,
    tour_cost,
    two_best_indices,
    undirected_tour_count,
    validate_distance,
)

PARENT_A = [7, 3, 6, 1, 0, 2, 5, 4]
PARENT_B = [5, 4, 1, 7, 2, 3, 6, 0]
CHILD_A = [7, 3, 1, 2, 6, 0, 5, 4]
CHILD_B = [5, 4, 7, 3, 1, 2, 6, 0]


def test_order_crossover_example():
    first, second = order_crossover(PARENT_A, PARENT_B, 2, 5)
    assert first == CHILD_A
    assert second == CHILD_B
    assert sorted(first) == sorted(PARENT_A)
    assert sorted(second) == sorted(PARENT_B)


def test_raw_segment_swap_would_repeat_a_city():
    raw = PARENT_A[:2] + PARENT_B[2:6] + PARENT_A[6:]
    assert raw == [7, 3, 1, 7, 2, 3, 5, 4]
    assert len(raw) != len(set(raw))


def test_adjacent_swap_of_city_6():
    tour = [7, 3, 1, 2, 6, 0, 5, 4]
    assert adjacent_swap(tour, 4) == [7, 3, 1, 2, 0, 6, 5, 4]
    assert adjacent_swap([0, 1, 2], 2) == [2, 1, 0]


def test_named_mutation_alternatives():
    tour = [0, 1, 2, 3, 4]
    assert swap_cities(tour, 1, 3) == [0, 3, 2, 1, 4]
    assert reverse_substring(tour, 1, 3) == [0, 3, 2, 1, 4]
    with pytest.raises(ValueError):
        reverse_substring(tour, 3, 1)


def test_two_best_are_the_shortest_tours():
    assert two_best_indices([9, 4, 4, 7]) == (1, 2)


def test_lab_instance_optimum_and_the_ga():
    exact = brute_force_best(LAB_DISTANCE)
    found = run_tsp(LAB_DISTANCE, population_size=20, generations=80, seed=1)
    assert found.tour[0] == 0
    assert sorted(found.tour) == [0, 1, 2, 3, 4]
    assert found.cost == tour_cost(found.tour, LAB_DISTANCE)
    assert found.cost == exact.cost
    assert found.best_by_generation[-1] <= found.best_by_generation[0]


def test_search_space_counts():
    assert chromosome_space(4) == 6
    assert undirected_tour_count(4) == 3
    assert chromosome_space(5) == 24
    assert undirected_tour_count(5) == 12


def test_reject_n_below_3_and_a_bad_matrix():
    with pytest.raises(ValueError):
        validate_distance(((0, 1), (1, 0)))
    with pytest.raises(ValueError):
        chromosome_space(2)
    asymmetric = ((0, 1, 2), (9, 0, 3), (2, 3, 0))
    with pytest.raises(ValueError):
        validate_distance(asymmetric)
    negative = ((0, -1, 2), (-1, 0, 3), (2, 3, 0))
    with pytest.raises(ValueError):
        validate_distance(negative)
    ragged = ((0, 1, 1), (1, 0))
    with pytest.raises(ValueError):
        validate_distance(ragged)


def test_reject_a_tour_that_repeats_a_city():
    with pytest.raises(ValueError):
        tour_cost((0, 1, 1, 2), LAB_DISTANCE)


def test_cost_33_has_no_matrix():
    with pytest.raises(LookupError):
        cost_33_without_matrix()


def test_mutation_rate_default():
    assert DEFAULT_MUTATION_RATE == 0.1


def test_return_edge_is_counted():
    tour = (0, 1, 2, 3, 4)
    expected = 2 + 6 + 8 + 6 + 7
    assert tour_cost(tour, LAB_DISTANCE) == expected
