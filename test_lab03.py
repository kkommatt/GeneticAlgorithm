"""Tests for lab 3."""

import pytest

from labs.lab03_gray_code import (
    binary_to_gray,
    epsilon,
    gray_list,
    gray_to_binary,
    hamming,
    hypercube_edges,
    integer_to_binary,
    triangle_holds,
)


def test_hamming_examples():
    assert hamming("karolin", "kathrin") == 3
    assert hamming("karolin", "kerstin") == 3
    assert hamming("kathrin", "kerstin") == 4
    assert hamming("0000", "1111") == 4
    assert hamming("2173896", "2233796") == 3


def test_metric_axioms_and_the_pointwise_proof():
    samples = ["karolin", "kathrin", "kerstin", "0000000"]
    for word in samples:
        assert hamming(word, word) == 0
    assert hamming("karolin", "kathrin") == hamming("kathrin", "karolin")
    for a in samples:
        for b in samples:
            for c in samples:
                assert triangle_holds(a, b, c)
                assert hamming(a, b) == sum(epsilon(a, b))


def test_unequal_length_is_rejected():
    with pytest.raises(ValueError):
        hamming("karolin", "karol")
    with pytest.raises(ValueError):
        epsilon("01", "0")


def test_reflected_lists_for_one_two_and_three_bits():
    assert gray_list(1) == ("0", "1")
    assert gray_list(2) == ("00", "01", "11", "10")
    assert gray_list(3) == ("000", "001", "011", "010", "110", "111", "101", "100")


def test_binary_gray_round_trip_and_agreement_with_the_list():
    for n in range(1, 9):
        codes = gray_list(n)
        assert len(codes) == len(set(codes)) == 2**n
        for value, code in enumerate(codes):
            binary = integer_to_binary(value, n)
            assert binary_to_gray(binary) == code
            assert gray_to_binary(code) == binary
            assert gray_to_binary(binary_to_gray(binary)) == binary
        if n < 8:
            first_half = gray_list(n + 1)[: 2**n]
            assert first_half == tuple("0" + code for code in codes)


def test_neighbors_and_the_cyclic_ends_differ_by_one_bit():
    for n in range(1, 8):
        codes = gray_list(n)
        pairs = list(zip(codes, codes[1:])) + [(codes[-1], codes[0])]
        assert all(hamming(a, b) == 1 for a, b in pairs)


def test_hypercube_has_an_edge_exactly_at_distance_one():
    edges = hypercube_edges(3)
    assert len(edges) == 3 * 2 ** (3 - 1)
    for left, right in edges:
        assert hamming(left, right) == 1
    assert ("000", "011") not in edges
    assert ("000", "001") in edges


def test_reject_empty_and_non_binary_strings():
    with pytest.raises(ValueError):
        binary_to_gray("")
    with pytest.raises(ValueError):
        gray_to_binary("102")
    with pytest.raises(ValueError):
        gray_list(0)
