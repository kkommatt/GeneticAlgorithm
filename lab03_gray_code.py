"""Lab 3. Hamming distance, and binary reflected Gray code in both directions.

G_n is built by reflection. binary_to_gray and gray_to_binary are the bit
operations that match that list: g = b xor (b >> 1), and the inverse prefix-xor.
"""

from __future__ import annotations


def hamming(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("Hamming distance is defined for equal-length strings")
    return sum(a != b for a, b in zip(left, right))


def epsilon(left: str, right: str) -> tuple[int, ...]:
    """1 at each position where the symbols differ, 0 elsewhere."""
    if len(left) != len(right):
        raise ValueError("epsilon is defined for equal-length strings")
    return tuple(int(a != b) for a, b in zip(left, right))


def triangle_holds(a: str, b: str, c: str) -> bool:
    """Pointwise step of the triangle proof, summed to the triangle inequality."""
    if not (len(a) == len(b) == len(c)):
        raise ValueError("the three strings must have equal length")
    for i in range(len(a)):
        e_ab = int(a[i] != b[i])
        e_ac = int(a[i] != c[i])
        e_bc = int(b[i] != c[i])
        if e_ac + e_bc < e_ab:
            return False
    return hamming(a, b) <= hamming(a, c) + hamming(c, b)


def _check_bits(bits: str) -> None:
    if not bits or any(symbol not in "01" for symbol in bits):
        raise ValueError("expected a non-empty string of 0 and 1")


def gray_list(n: int) -> tuple[str, ...]:
    """G_1 = 0, 1. G_{n+1} = 0 G_n followed by 1 and the reversal of G_n."""
    if n < 1:
        raise ValueError("n must be at least 1")
    codes = ["0", "1"]
    for _ in range(2, n + 1):
        reflected = list(reversed(codes))
        codes = ["0" + code for code in codes] + ["1" + code for code in reflected]
    return tuple(codes)


def binary_to_gray(bits: str) -> str:
    """g_0 = b_0, and g_i = b_i xor b_{i-1} for i > 0."""
    _check_bits(bits)
    out = [bits[0]]
    for previous, current in zip(bits, bits[1:]):
        out.append("1" if previous != current else "0")
    return "".join(out)


def gray_to_binary(bits: str) -> str:
    """b_0 = g_0, and b_i = b_{i-1} xor g_i for i > 0."""
    _check_bits(bits)
    out = [bits[0]]
    for gray_bit in bits[1:]:
        out.append("1" if out[-1] != gray_bit else "0")
    return "".join(out)


def integer_to_binary(value: int, n: int) -> str:
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0 <= value < 2**n:
        raise ValueError(f"value must lie in 0..{2**n - 1}")
    return f"{value:0{n}b}"


def hypercube_edges(n: int) -> tuple[tuple[str, str], ...]:
    """Undirected edges of the Boolean hypercube. Endpoints differ by one bit."""
    if n < 1:
        raise ValueError("n must be at least 1")
    edges: list[tuple[str, str]] = []
    for value in range(2**n):
        node = f"{value:0{n}b}"
        for index, bit in enumerate(node):
            flipped = "1" if bit == "0" else "0"
            other = node[:index] + flipped + node[index + 1 :]
            if node < other:
                edges.append((node, other))
    return tuple(edges)
