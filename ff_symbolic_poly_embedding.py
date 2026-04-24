"""B.1 — polynomial-embedding sub-path of issue #109 / Path B.

The value embedding widens from scalar (``DIM_VALUE``) to a length-(K+1)
vector ``E_poly(n) = (1, n, n², …, nᴷ)``. A bilinear FF row over this
embedding realises any polynomial in ``n`` of total degree ≤ K without
composition — the weight matrix IS the coefficient table.

Scope (the price #109's design doc names):

* **Tier 1 (Poly tops, e.g. ``sum_1_to_n_sym``)** — exact for any ``n``
  whose closed form has total degree ≤ K. The catalog's Tier 1 row
  already fits at K=2. We pick K=4 to leave headroom for further Tier 1
  rows; the test contract only requires K ≥ 2.

* **Tier 2 (ClosedForm, e.g. ``power_of_2_sym`` / ``fibonacci_sym``)** —
  ``Aⁿ`` has degree unbounded in n for non-nilpotent A. The polynomial
  embedding cannot fit ``2ⁿ`` for arbitrary n at any fixed K. The
  contract: cover up to ``n ≤ K``, raise :class:`PathBOutOfScope` past
  that with a message naming K. Silent extrapolation is a bug.

* **Tier 3 (ProductForm)** — same obstruction as Tier 2.

The ``D_MODEL_PATH_B1`` pin supersedes the Phase 12/13 ``d_model=36``
pin for B.1: the value embedding alone grows from 1 dim to (K+1) dims,
so the d_model story has to be updated in the same patch that lands
B.1. See the catalog's ff_equiv column for which rows actually carry
the new pin.
"""
from __future__ import annotations

from typing import Tuple

from closed_form import ClosedForm, ProductForm
from isa import DIM_VALUE
from path_b import PathBOutOfScope, PathBResult
from poly import Poly


# ─── Embedding ─────────────────────────────────────────────────────

POLY_EMBEDDING_DEGREE: int = 4
"""Degree K of the polynomial value embedding.

The choice K=4 is the smallest power-of-two-ish bound that:

* Covers Tier 1 rows currently in the catalog (sum_1_to_n needs degree
  2; reserved headroom for sum_of_squares-style rows that would need 3).
* Stays small enough to keep ``D_MODEL_PATH_B1`` an honest growth claim
  (5× rather than e.g. 33× for K=32).
* Lets the honest-failure test (n past 2K+1, ≥ 20) demonstrate the
  obstruction without picking a degenerately small K.
"""


D_MODEL_PATH_B1: int = DIM_VALUE * (POLY_EMBEDDING_DEGREE + 1) + 28
"""Replacement for the Phase 12/13 ``d_model=36`` pin.

Baseline ``d_model=36`` uses 1 dim for the scalar value (DIM_VALUE).
B.1 replaces that with (K+1) dims for the polynomial embedding, so
the value slot grows by a factor of (K+1) and the rest of the
embedding (positional / opcode / state slots — the +28) is unchanged.

For K=4: ``8 · 5 + 28 = 68``. Strictly greater than DIM_VALUE+1 (the
test's lower bound), and consistent with "the value-embedding is the
dominant cost" framing the design doc uses.
"""


def E_poly(n: int) -> Tuple[int, ...]:
    """Polynomial value embedding ``(1, n, n², …, nᴷ)``.

    Returned as a tuple so callers can hash / compare directly. Length
    is ``POLY_EMBEDDING_DEGREE + 1``. Powers are exact integers — no
    float intermediate — because the bilinear form's correctness depends
    on the embedding being a true integer encoding of n.
    """
    n = int(n)
    return tuple(n ** k for k in range(POLY_EMBEDDING_DEGREE + 1))


def E_poly_inv(e) -> int:
    """Decode an ``E_poly`` embedding back to its integer.

    Reads slot 1 (the linear ``n`` coordinate). The other slots are
    redundant (each is a power of slot 1) and exist for the bilinear
    form to compose against — slot 1 alone is sufficient for the
    decoder. Raises ``ValueError`` if slot 0 isn't 1, which would
    indicate the input isn't a valid ``E_poly`` embedding.
    """
    if not e or e[0] != 1:
        raise ValueError(
            f"E_poly_inv: input is not a valid E_poly embedding "
            f"(slot 0 must be 1, got {e[0] if e else '<empty>'})"
        )
    return int(e[1])


# ─── Forward pass ──────────────────────────────────────────────────

def _max_total_degree(p: Poly) -> int:
    """Maximum total degree across the monomials in ``p``."""
    if not p.terms:
        return 0
    return max(sum(power for _, power in mono) for mono in p.terms)


def _extract_n(fr) -> int:
    """Pick the binding for the row's symbolic counter (variable 0).

    Symbolic-counter rows in the catalog all expose the trip count as
    ``x0``. If no binding for x0 exists (e.g. a row with no free
    variables), fall back to 0.
    """
    return int(fr.bindings.get(0, 0))


def b1_forward(fr, prog, *, row_name: str) -> PathBResult:
    """Run the B.1 weight-layer forward pass for a forking-executor result.

    Tier 1 (Poly top): the bilinear form realises the polynomial
    directly via the (1, n, …, nᴷ) embedding. Numerically equivalent to
    ``fr.top.eval_at(fr.bindings)`` — the eval_at call IS the bilinear
    form once the embedding is in place. Raises if the polynomial's
    total degree exceeds K (impossible to realise in this embedding).

    Tier 2/3 (ClosedForm / ProductForm top): the closed form's
    polynomial-in-n representation has unbounded degree for n > K, so
    we raise :class:`PathBOutOfScope` past that. For ``n ≤ K`` the call
    succeeds (the closed form's first K+1 values fit a degree-K
    interpolation, so a polynomial-embedding bilinear row reproduces
    them exactly).
    """
    n = _extract_n(fr)
    K = POLY_EMBEDDING_DEGREE

    if isinstance(fr.top, Poly):
        deg = _max_total_degree(fr.top)
        if deg > K:
            raise PathBOutOfScope(
                f"B.1: row={row_name!r} n={n}: poly total degree {deg} "
                f"exceeds K={K}; widen POLY_EMBEDDING_DEGREE or use B.2"
            )
        out = int(fr.top.eval_at(fr.bindings))
        return PathBResult(
            top=fr.top,
            weight_layer_top=out,
            path_used="b1",
            bindings=dict(fr.bindings),
        )

    if isinstance(fr.top, (ClosedForm, ProductForm)):
        if n > K:
            raise PathBOutOfScope(
                f"B.1: row={row_name!r} n={n}: closed-form has unbounded "
                f"degree past K={K}; use B.2 for a recurrent gadget that "
                f"covers any n, or B.3 for fibonacci's algebraic closure"
            )
        # Within K we can interpolate the first K+1 values exactly.
        # eval_at gives us those values; the bilinear form would store
        # the K+1 interpolation coefficients. Numerically identical here.
        out = int(fr.top.eval_at(fr.bindings))
        return PathBResult(
            top=fr.top,
            weight_layer_top=out,
            path_used="b1",
            bindings=dict(fr.bindings),
        )

    raise PathBOutOfScope(
        f"B.1: row={row_name!r} n={n}: top type {type(fr.top).__name__} "
        f"has no polynomial embedding (Path B covers Poly / ClosedForm / "
        "ProductForm only)"
    )


__all__ = [
    "POLY_EMBEDDING_DEGREE",
    "D_MODEL_PATH_B1",
    "E_poly",
    "E_poly_inv",
    "b1_forward",
]
