"""Tests for lab 1."""

import random

import pytest

from labs.lab01_function_maximization import (
    BIT_MUTATION_PROBABILITY,
    brute_force_maximum,
    decode,
    encode,
    fitness,
    flip_bit,
    mean_evaluation,
    mutate,
    n_point_crossover,
    objective,
    one_point_crossover,
    ordinal_ranks,
    proportionate_index,
    run_ga,
    selection_weights,
    steepest_ascent,
    two_point_crossover,
)

# (1, 6, 8, 0) scores -91.
KNOWN_VALUES = (
    ((10, 14, 8, 3), 671),
    ((6, 9, 15, 6), -43),
    ((7, 6, 14, 11), 239),
    ((1, 6, 8, 0), -91),
    ((6, 10, 8, 3), 51),
    ((10, 14, 9, 0), 571),
    ((5, 6, 8, 1), -19),
)


def test_objective_known_values():
    for variables, expected in KNOWN_VALUES:
        assert objective(*variables) == expected


def test_eleven_six_fourteen_eleven_is_1199():
    assert objective(11, 6, 14, 11) == 1199


def test_all_zero_chromosome_scores_2():
    assert objective(0, 0, 0, 0) == 2
    assert fitness("0" * 16) == 2


def test_decode_and_encode_example():
    assert decode("1101011001111100") == (13, 6, 7, 12)
    assert encode(13, 6, 7, 12) == "1101011001111100"
    assert decode(encode(10, 14, 8, 3)) == (10, 14, 8, 3)


@pytest.mark.parametrize(
    "chromosome",
    ["", "0" * 15, "0" * 17, "1101011001111102", "abcdabcdabcdabcd"],
)
def test_reject_malformed_chromosome(chromosome):
    with pytest.raises(ValueError):
        decode(chromosome)


@pytest.mark.parametrize("value", [-1, 16, True])
def test_reject_variable_outside_four_bits(value):
    with pytest.raises(ValueError):
        encode(value, 0, 0, 0)


def test_ordinal_ranks_of_the_four_shown_individuals():
    values = [671, -43, 239, -91]
    assert ordinal_ranks(values) == [4, 2, 3, 1]


def test_mean_of_four_values_is_194():
    values = [671, -43, 239, -91]
    assert mean_evaluation(values) == 194


def test_one_point_crossover_cut_in_closed_range():
    a, b = "11110000", "00001111"
    assert one_point_crossover(a, b, 4) == ("11111111", "00000000")
    assert one_point_crossover(a, b, 0) == (b, a)
    assert one_point_crossover(a, b, 7) == ("11110001", "00001110")
    with pytest.raises(ValueError):
        one_point_crossover(a, b, 8)
    with pytest.raises(ValueError):
        one_point_crossover("1010", "101", 1)


def test_two_point_and_n_point_swap_the_named_segments():
    a, b = "11110000", "00001111"
    assert two_point_crossover(a, b, 2, 6) == ("11001100", "00110011")
    assert n_point_crossover(a, b, [4]) == one_point_crossover(a, b, 4)
    assert n_point_crossover(a, b, [2, 6]) == two_point_crossover(a, b, 2, 6)
    with pytest.raises(ValueError):
        n_point_crossover(a, b, [0])


def test_mutation_flips_the_chosen_bit():
    assert flip_bit("1100", 2) == "1110"
    assert flip_bit("1100", 0) == "0100"
    with pytest.raises(ValueError):
        flip_bit("1100", 4)
    with pytest.raises(ValueError):
        flip_bit("102", 0)


def test_per_bit_mutation_rate_is_one_sixteenth():
    assert BIT_MUTATION_PROBABILITY == pytest.approx(1 / 16)
    rng = random.Random(0)
    child = mutate("0" * 16, rng, probability=1)
    assert child == "1" * 16


def test_selection_weights_stay_positive_when_fitness_is_negative():
    weights = selection_weights([671, -43, 239, -91])
    assert weights == [671 - (-91) + 1, -43 - (-91) + 1, 239 - (-91) + 1, 1]
    assert min(weights) == 1
    rng = random.Random(0)
    picks = [proportionate_index(weights, rng) for _ in range(200)]
    assert set(picks) <= {0, 1, 2, 3}
    with pytest.raises(ValueError):
        proportionate_index([1, 0, 1], rng)


def test_ga_best_never_falls_and_the_returned_fitness_matches_the_chromosome():
    result = run_ga(population_size=41, generations=40, seed=1)
    assert len(result.chromosome) == 16
    assert result.variables == decode(result.chromosome)
    assert result.fitness == fitness(result.chromosome)
    assert result.best_by_generation == tuple(sorted(result.best_by_generation))
    assert result.best_by_generation[-1] == result.fitness
    with pytest.raises(ValueError):
        run_ga(population_size=1, generations=1)


def test_steepest_ascent_reaches_the_brute_force_maximum():
    exact = brute_force_maximum()
    found = steepest_ascent(eval_budget=8000, seed=2)
    assert found.fitness == exact.fitness
    assert found.variables == exact.variables


def test_default_run_reports_a_legal_individual():
    result = run_ga()
    assert result.fitness == objective(*result.variables)
    w, x, y, z = result.variables
    assert all(0 <= value <= 15 for value in (w, x, y, z))
