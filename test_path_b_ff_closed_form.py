"""TDD acceptance tests for issue #109 — Path B: weight-layer realisation
of ``ClosedForm`` / ``ProductForm``.

Issue #109 is a *tracking* issue. It offers three alternative paths, and
says "pursue only if a motivator appears." There is therefore no single
implementation to test. What the tests below define is the **contract**
any Path B implementation must satisfy to close #109, broken down by:

  * ``test_common_*``     — must hold for any of B.1, B.2, B.3
  * ``test_b1_*``         — polynomial embedding ``E(n) = (1, n, …, nᴷ)``
  * ``test_b2_*``         — recurrent FF gadget (iterated microstep)
  * ``test_b3_*``         — algebraic-number coefficients (Binet-style)
  * ``test_motivator_*``  — per #107, no silent activation

A path is closed when its common + path-specific + motivator tests are
GREEN. All tests are RED today — they import APIs that do not yet exist.
Unimplemented-import failures surface as ``TDD-RED`` checks (expected
while the issue is deferred) rather than hard harness failures, so this
file can be wired into the existing runner without turning the tree red
until a Path B is chosen.

Test-harness style matches ``test_ff_symbolic.py`` (tiny ``_check`` /
``_fail`` / ``_pass`` wrappers, standalone runner, no pytest dep).
"""
from __future__ import annotations

import importlib
import sys
import traceback
from typing import Any, List, Optional

import ff_symbolic as ff
from closed_form import ClosedForm, ProductForm
from executor import NumPyExecutor
from symbolic_executor import Poly, run_forking
from symbolic_programs_catalog import classify_program


# ─── Test harness ──────────────────────────────────────────────────

_failures: List[str] = []
_reds: List[str] = []  # TDD-RED entries — expected while #109 deferred


def _fail(name: str, detail: str) -> None:
    _failures.append(f"{name}: {detail}")
    print(f"  FAIL  {name}  {detail}")


def _pass(name: str) -> None:
    print(f"  PASS  {name}")


def _red(name: str, detail: str) -> None:
    _reds.append(f"{name}: {detail}")
    print(f"  TDD-RED  {name}  {detail}")


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _pass(name)
    else:
        _fail(name, detail)


def _try_import(modname: str) -> Optional[Any]:
    """Import a Path B module; return None and record TDD-RED on failure.

    Path B modules do not exist yet. Tests that need them call this first
    and short-circuit on ``None`` rather than raising.
    """
    try:
        return importlib.import_module(modname)
    except ImportError as e:
        return e  # sentinel — caller uses isinstance to detect


def _have(mod: Any) -> bool:
    return mod is not None and not isinstance(mod, ImportError)


# ─── Shared fixtures ───────────────────────────────────────────────

def _closed_form_rows():
    """Same four rows as test_ff_symbolic._closed_form_rows."""
    import programs as P
    return [
        # (name, make_fn, validation n-values, expected sibling type)
        ("sum_1_to_n_sym", P.make_sum_1_to_n_sym, (1, 2, 5, 10, 20), Poly),
        ("power_of_2_sym", P.make_power_of_2_sym, (0, 1, 4, 8, 10), ClosedForm),
        ("fibonacci_sym",  P.make_fibonacci_sym,  (1, 2, 5, 10, 15), ClosedForm),
        ("factorial_sym",  P.make_factorial_sym,  (1, 2, 5, 7, 10), ProductForm),
    ]


def _numpy_expected(prog, bindings_if_any=None) -> int:
    """Concrete integer NumPyExecutor returns for ``prog``."""
    np_exec = NumPyExecutor()
    return np_exec.execute(prog).steps[-1].top


# ─── Common contract (any of B.1 / B.2 / B.3 must satisfy) ────────

def test_common_entrypoint_exists():
    """Any Path B implementation exposes a weight-layer forward-pass
    entrypoint distinct from ``evaluate_program_forking`` (which is Path
    A — solver-structural + boundary ``eval_at``). The name is a contract
    point: the canonical spelling in the issue and design doc is
    ``evaluate_program_forking_weight_layer`` (takes the same signature
    as ``evaluate_program_forking`` plus ``path`` ∈ {"b1", "b2", "b3"}
    or equivalent dispatch).
    """
    attr = getattr(ff, "evaluate_program_forking_weight_layer", None)
    if attr is None:
        _red("common.entrypoint",
             "ff.evaluate_program_forking_weight_layer not defined — "
             "Path B not yet implemented")
        return
    _check("common.entrypoint.callable", callable(attr),
           f"attr={attr!r} is not callable")


def test_common_solver_structural_preserved():
    """Path B must ADD a weight-layer realisation, not REPLACE the
    solver-level structural claim landed in #107. Regardless of which
    sub-path ships, ``evaluate_program_forking(prog)`` continues to
    produce ``Poly / ClosedForm / ProductForm`` tops structurally equal
    to ``run_forking(prog, solve_recurrences=True)`` on every catalog
    row in the collapsed_closed_form set. This test guards against a
    regression where a Path B implementation inadvertently rewires the
    solver-level claim."""
    for name, make_fn, _ns, expected_type in _closed_form_rows():
        # Pick the first validation n — we're only checking sibling
        # identity, not numeric value.
        prog, _ = make_fn(5 if name != "factorial_sym" else 4)
        native = run_forking(prog, input_mode="symbolic",
                             solve_recurrences=True)
        fs = ff.evaluate_program_forking(prog, input_mode="symbolic")
        _check(
            f"common.solver_struct.type[{name}]",
            isinstance(native.top, expected_type)
            and isinstance(fs.top, expected_type),
            f"native={type(native.top).__name__} "
            f"ff={type(fs.top).__name__}",
        )
        _check(
            f"common.solver_struct.eq[{name}]",
            native.top == fs.top,
            f"native={native.top!r} ff={fs.top!r}",
        )


def test_common_weight_layer_forward_numeric():
    """Core claim of Path B: the weight layer (not ``.top.eval_at``)
    reproduces the numeric answer of ``NumPyExecutor`` for every catalog
    row × validation ``n`` *inside the path's declared scope*. If the
    implementation cannot cover a given ``(row, n)``, it must raise a
    clearly-typed out-of-scope exception — NOT silently return a wrong
    integer. This test asserts: every in-scope pair matches; every
    out-of-scope pair raises the advertised scope exception."""
    wl = getattr(ff, "evaluate_program_forking_weight_layer", None)
    if wl is None:
        _red("common.weight_layer_numeric", "entrypoint not defined")
        return

    scope_exc_name = getattr(ff, "PATH_B_OUT_OF_SCOPE_EXCEPTION", None)
    if scope_exc_name is None:
        _red("common.weight_layer_numeric",
             "ff.PATH_B_OUT_OF_SCOPE_EXCEPTION sentinel not defined — "
             "Path B must name its out-of-scope exception for honest "
             "failure reporting")
        return

    for name, make_fn, ns, _type in _closed_form_rows():
        for n in ns:
            prog, expected = make_fn(n)
            try:
                result = wl(prog, input_mode="symbolic")
                numeric = result.weight_layer_top  # contract field name
                in_scope = True
            except scope_exc_name as e:
                numeric = None
                in_scope = False
                scope_msg = str(e)
            except Exception as e:  # wrong type of failure
                _fail(
                    f"common.weight_layer_numeric[{name}(n={n})]",
                    f"raised {type(e).__name__}: {e} — expected either a "
                    f"numeric result or PATH_B_OUT_OF_SCOPE_EXCEPTION",
                )
                continue

            np_top = _numpy_expected(prog)
            if in_scope:
                _check(
                    f"common.weight_layer_numeric[{name}(n={n})]",
                    numeric == np_top == expected,
                    f"weight_layer={numeric} np={np_top} expected={expected}",
                )
            else:
                # Out-of-scope is acceptable — but the path must
                # advertise WHY via the exception message (path id,
                # the catalog row, the failing n).
                _check(
                    f"common.weight_layer_scope_msg[{name}(n={n})]",
                    name in scope_msg and str(n) in scope_msg,
                    f"scope exception missing row/n context: {scope_msg!r}",
                )


def test_common_catalog_ff_equiv_column_extended():
    """Today ``ff_equiv`` ∈ {bilinear, solver_structural, n/a} (set by
    ``symbolic_programs_catalog.run_catalog``, pinned in
    ``test_symbolic_programs_catalog.py``). Path B must introduce a new
    value — ``bilinear_weight_layer`` — and set it on rows whose
    weight-layer realisation is proven, WITHOUT demoting any existing
    ``bilinear`` row. The four collapsed_closed_form rows split: ones the
    chosen path covers go to ``bilinear_weight_layer``; ones it doesn't
    stay at ``solver_structural`` with a scope annotation."""
    from symbolic_programs_catalog import run_catalog

    rows = list(run_catalog())

    # Sanity — current pins must not have been regressed.
    currently_bilinear = [r for r in rows if r.ff_equiv == "bilinear"]
    _check(
        "common.ff_equiv.pin_preserved",
        len(currently_bilinear) >= 1,
        "Path B must not demote any row currently classified as "
        "ff_equiv=bilinear",
    )

    # Path B adds a new label. If it hasn't been introduced yet, that's
    # the RED state.
    labels = {r.ff_equiv for r in rows if r.ff_equiv}
    if "bilinear_weight_layer" not in labels:
        _red("common.ff_equiv.new_label",
             f"no row has ff_equiv='bilinear_weight_layer' yet "
             f"(current labels: {sorted(labels)})")
        return

    # When the label exists, at least one of the four collapsed_closed_form
    # rows must carry it — otherwise the path didn't actually close #109.
    closed_form_row_names = {
        "sum_1_to_n_sym(n)", "power_of_2_sym(n)",
        "fibonacci_sym(n)", "factorial_sym(n)",
    }
    upgraded = [r for r in rows
                if r.name in closed_form_row_names
                and r.ff_equiv == "bilinear_weight_layer"]
    _check(
        "common.ff_equiv.at_least_one_upgrade",
        len(upgraded) >= 1,
        "Path B introduces 'bilinear_weight_layer' but no "
        "collapsed_closed_form row was upgraded to it — label is dead",
    )


def test_common_scope_predicate_documented():
    """Each path has documented scope limits (B.1: n bounded by K;
    B.2: architectural-recurrence rather than single-layer; B.3: only
    A with algebraic eigenvalues). Scope must be queryable, not
    buried in commentary. Contract: a pure function
    ``ff.path_b_in_scope(row_name, n) -> bool`` returns True iff the
    weight-layer entrypoint will succeed without raising."""
    predicate = getattr(ff, "path_b_in_scope", None)
    if predicate is None:
        _red("common.scope_predicate",
             "ff.path_b_in_scope not defined — scope must be queryable "
             "from outside (catalog reporting + blog #82 narrative)")
        return

    # Predicate must agree with actual behavior on the validation set.
    wl = ff.evaluate_program_forking_weight_layer
    scope_exc = ff.PATH_B_OUT_OF_SCOPE_EXCEPTION
    for name, make_fn, ns, _type in _closed_form_rows():
        for n in ns:
            prog, _ = make_fn(n)
            advertised = predicate(name, n)
            try:
                wl(prog, input_mode="symbolic")
                actual_in_scope = True
            except scope_exc:
                actual_in_scope = False
            except Exception:
                continue  # covered by numeric test above
            _check(
                f"common.scope_predicate.agree[{name}(n={n})]",
                advertised == actual_in_scope,
                f"predicate says {advertised}, actual {actual_in_scope}",
            )


# ─── B.1 — polynomial embedding E(n) = (1, n, n², …, nᴷ) ──────────

def test_b1_polynomial_embedding_roundtrip():
    """B.1 extends the value embedding from scalar (``DIM_VALUE``) to a
    length-(K+1) vector ``E_poly(n) = (1, n, n², …, nᴷ)``. A decoder
    ``E_poly_inv`` must round-trip integers in the declared range. Both
    must be consistent with an integer K published as
    ``POLY_EMBEDDING_DEGREE``."""
    mod = _try_import("ff_symbolic_poly_embedding")
    if not _have(mod):
        _red("b1.embedding_roundtrip",
             "ff_symbolic_poly_embedding module not present")
        return

    K = getattr(mod, "POLY_EMBEDDING_DEGREE", None)
    _check("b1.embedding.K_defined", isinstance(K, int) and K >= 2,
           f"POLY_EMBEDDING_DEGREE={K!r}; must be int ≥ 2 (K=1 is just Poly)")

    if not isinstance(K, int):
        return
    for n in [0, 1, 2, 5, 10, 2 ** K - 1]:  # last one intentionally stresses K
        try:
            e = mod.E_poly(n)
            back = mod.E_poly_inv(e)
        except Exception as ex:
            _fail(f"b1.embedding.roundtrip[n={n}]",
                  f"raised {type(ex).__name__}: {ex}")
            continue
        # e must have length K+1 and encode powers.
        _check(f"b1.embedding.shape[n={n}]", len(e) == K + 1,
               f"len(E_poly({n}))={len(e)} expected {K + 1}")
        _check(f"b1.embedding.roundtrip[n={n}]", back == n,
               f"E_poly_inv(E_poly({n}))={back}")


def test_b1_exact_on_tier1():
    """Tier 1 (``sum_1_to_n_sym``) already fits ``K=2`` trivially —
    result is ``x0 + ½·x1 + ½·x1²``, a degree-2 polynomial. B.1 MUST be
    exact for all validation ``n`` on this row; it's the row that
    motivates B.1 at all."""
    mod = _try_import("ff_symbolic_poly_embedding")
    if not _have(mod):
        _red("b1.tier1_exact", "module not present")
        return
    wl = getattr(ff, "evaluate_program_forking_weight_layer", None)
    if wl is None:
        _red("b1.tier1_exact", "weight-layer entrypoint missing")
        return

    import programs as P
    for n in (1, 2, 5, 10, 20):
        prog, expected = P.make_sum_1_to_n_sym(n)
        try:
            out = wl(prog, input_mode="symbolic", path="b1").weight_layer_top
        except Exception as e:
            _fail(f"b1.tier1_exact[n={n}]",
                  f"in-scope Tier 1 row raised {type(e).__name__}: {e}")
            continue
        _check(f"b1.tier1_exact[n={n}]", out == expected,
               f"b1={out} expected={expected}")


def test_b1_honest_failure_beyond_K():
    """B.1's central obstruction: ``Aⁿ`` has degree unbounded in ``n``
    for non-nilpotent ``A``, so fibonacci_sym / power_of_2_sym must fail
    exactly and honestly for ``n`` large enough that the true answer
    outruns degree K. The test asserts: there EXISTS an ``n`` per row
    where the weight-layer call raises ``PATH_B_OUT_OF_SCOPE_EXCEPTION``
    with a message naming ``K`` and the failing ``n``. Silent wrap or
    truncation is a BUG."""
    scope_exc = getattr(ff, "PATH_B_OUT_OF_SCOPE_EXCEPTION", None)
    wl = getattr(ff, "evaluate_program_forking_weight_layer", None)
    if scope_exc is None or wl is None:
        _red("b1.honest_failure", "path B entrypoints missing")
        return

    import programs as P
    mod = _try_import("ff_symbolic_poly_embedding")
    K = getattr(mod, "POLY_EMBEDDING_DEGREE", 2) if _have(mod) else 2

    # Pick an n guaranteed past any reasonable K for an exponentially-
    # growing row. If the implementation silently returns 2**50 via
    # polynomial extrapolation it's wrong; the test catches that.
    for name, make_fn in [("power_of_2_sym", P.make_power_of_2_sym),
                          ("fibonacci_sym",  P.make_fibonacci_sym)]:
        n_blown = max(2 * K + 1, 20)
        prog, expected = make_fn(n_blown)
        try:
            out = wl(prog, input_mode="symbolic", path="b1").weight_layer_top
            # If we got here, silent success past K. Must equal expected
            # (i.e., the path actually covers this n somehow) — otherwise
            # fail loud.
            _check(
                f"b1.honest_failure[{name}(n={n_blown})].silent_exact",
                out == expected,
                f"silently returned {out} for n={n_blown} (K={K}); "
                f"expected {expected} or scope exception",
            )
        except scope_exc as e:
            _check(
                f"b1.honest_failure[{name}(n={n_blown})].msg_cites_K",
                "K" in str(e) or str(K) in str(e),
                f"scope exception for n={n_blown} doesn't cite K: {e!r}",
            )
        except Exception as e:
            _fail(
                f"b1.honest_failure[{name}(n={n_blown})]",
                f"wrong exception type {type(e).__name__}: {e}",
            )


def test_b1_dmodel_pin_updated():
    """Phase 12/13 pins ``d_model=36``. B.1's embedding growth by factor
    K breaks that pin. If B.1 lands, a new pin ``D_MODEL_PATH_B1 = 36 *
    (K+1)`` (or equivalent accounting) MUST be published so the
    weight-budget story remains honest. We don't care about the exact
    formula — we care that the pin exists and is reachable from
    ``isa``/``ff_symbolic``."""
    mod = _try_import("ff_symbolic_poly_embedding")
    if not _have(mod):
        _red("b1.dmodel_pin", "module not present")
        return
    pin = getattr(mod, "D_MODEL_PATH_B1", None)
    _check(
        "b1.dmodel_pin_exists",
        isinstance(pin, int) and pin > 0,
        f"D_MODEL_PATH_B1={pin!r} — must be a positive int pin that "
        "supersedes the d_model=36 claim for B.1",
    )
    # And it must be strictly larger than the baseline — otherwise B.1
    # wouldn't actually need the bigger embedding.
    from isa import DIM_VALUE
    if isinstance(pin, int):
        _check(
            "b1.dmodel_pin_grew",
            pin > DIM_VALUE + 1,
            f"B.1's D_MODEL_PATH_B1={pin} not larger than baseline "
            f"(DIM_VALUE={DIM_VALUE}); growth factor is the whole point",
        )


# ─── B.2 — recurrent FF gadget ────────────────────────────────────

def test_b2_recurrent_gadget_covers_tier2_and_tier3():
    """B.2's defining property: iterating the FF microstep ``n`` times
    reproduces ``Aⁿ · s_0 + …`` for Tier 2 and ``init · ∏ p(k)`` for
    Tier 3, with NO degree bound. Test: on every validation ``n`` for
    both ClosedForm and ProductForm rows, the numeric answer is exact."""
    mod = _try_import("ff_symbolic_recurrent")
    if not _have(mod):
        _red("b2.recurrent_covers_tier23",
             "ff_symbolic_recurrent module not present")
        return

    run_rec = getattr(mod, "evaluate_program_forking_recurrent", None)
    _check("b2.entrypoint.exists", callable(run_rec),
           "ff_symbolic_recurrent.evaluate_program_forking_recurrent missing")
    if not callable(run_rec):
        return

    for name, make_fn, ns, _type in _closed_form_rows():
        if name == "sum_1_to_n_sym":
            continue  # Tier 1 covered by Path A already
        for n in ns:
            prog, expected = make_fn(n)
            try:
                out = run_rec(prog, input_mode="symbolic").weight_layer_top
            except Exception as e:
                _fail(f"b2.numeric[{name}(n={n})]",
                      f"raised {type(e).__name__}: {e}")
                continue
            _check(f"b2.numeric[{name}(n={n})]", out == expected,
                   f"rec={out} expected={expected}")


def test_b2_explicit_loop_counter():
    """The issue flags this as the architectural price: B.2 requires a
    loop counter — either positional re-embedding or a dedicated head.
    Test: the recurrent-gadget output carries a field
    ``iterations_run`` that equals the trip count for each program,
    proving the counter is real (not a claim in prose)."""
    mod = _try_import("ff_symbolic_recurrent")
    run_rec = getattr(mod, "evaluate_program_forking_recurrent", None)
    if not callable(run_rec):
        _red("b2.loop_counter", "entrypoint missing")
        return

    import programs as P
    for n in (3, 7, 12):
        prog, _ = P.make_fibonacci_sym(n)
        try:
            r = run_rec(prog, input_mode="symbolic")
        except Exception as e:
            _fail(f"b2.loop_counter[n={n}]",
                  f"raised {type(e).__name__}: {e}")
            continue
        iters = getattr(r, "iterations_run", None)
        _check(
            f"b2.loop_counter.present[n={n}]",
            iters is not None,
            "result.iterations_run missing — loop counter not exposed",
        )
        _check(
            f"b2.loop_counter.value[n={n}]",
            iters == n,
            f"iterations_run={iters} expected n={n}",
        )


def test_b2_explicit_framing_change():
    """The issue is explicit: B.2 abandons the single-layer bilinear
    framing. If B.2 lands, it must carry a published scope clause saying
    so — otherwise the #69 / #75 / #76 / #77 single-layer story silently
    becomes false for closed-form rows. Test: a module-level constant
    ``SINGLE_LAYER_CLAIM_B2`` exists and is False."""
    mod = _try_import("ff_symbolic_recurrent")
    if not _have(mod):
        _red("b2.framing_change", "module not present")
        return
    claim = getattr(mod, "SINGLE_LAYER_CLAIM_B2", object())
    _check(
        "b2.framing_change.single_layer_false",
        claim is False,
        f"SINGLE_LAYER_CLAIM_B2={claim!r}; must be literal False — B.2 "
        "trades single-layer for iterated microsteps, and the framing "
        "honesty is the whole point",
    )


def test_b2_iteration_count_scales_with_n():
    """Corollary of B.2's recurrence framing: doubling ``n`` roughly
    doubles the iteration count for a ClosedForm row. If iteration count
    is constant in ``n`` the implementation is cheating (e.g., falling
    back to eval_at silently). Test with two n-values and check
    proportionality."""
    mod = _try_import("ff_symbolic_recurrent")
    run_rec = getattr(mod, "evaluate_program_forking_recurrent", None)
    if not callable(run_rec):
        _red("b2.iter_scales", "entrypoint missing")
        return

    import programs as P
    p1, _ = P.make_fibonacci_sym(5)
    p2, _ = P.make_fibonacci_sym(15)
    try:
        r1 = run_rec(p1, input_mode="symbolic")
        r2 = run_rec(p2, input_mode="symbolic")
    except Exception as e:
        _fail("b2.iter_scales", f"raised {type(e).__name__}: {e}")
        return
    i1 = getattr(r1, "iterations_run", 0)
    i2 = getattr(r2, "iterations_run", 0)
    _check(
        "b2.iter_scales.strict",
        i2 > i1,
        f"iterations_run(n=15)={i2} not > iterations_run(n=5)={i1} — "
        "constant-time suggests fallback to eval_at",
    )


# ─── B.3 — algebraic-number coefficients ──────────────────────────

def test_b3_algebraic_ring_exists():
    """B.3 widens ``Poly``'s coefficient ring to include algebraic
    numbers (e.g. ``(1+√5)/2``). Test: a class ``AlgebraicNumber`` (or
    equivalent) exists, supports ``+``/``-``/``·``, and can represent
    ``φ = (1+√5)/2`` exactly. Rounding behaviour (for eval_at) must be
    explicit."""
    mod = _try_import("algebraic_poly")
    if not _have(mod):
        _red("b3.algebraic_ring", "algebraic_poly module not present")
        return
    AN = getattr(mod, "AlgebraicNumber", None)
    if AN is None:
        _red("b3.algebraic_ring",
             "algebraic_poly.AlgebraicNumber not defined")
        return

    try:
        phi = AN.phi_fibonacci()  # contract: named constructor for golden ratio
    except AttributeError:
        _red("b3.algebraic_ring.phi",
             "AlgebraicNumber.phi_fibonacci() named constructor missing")
        return
    except Exception as e:
        _fail("b3.algebraic_ring.phi", f"{type(e).__name__}: {e}")
        return

    # phi² == phi + 1 is the defining identity; check it holds in the ring.
    try:
        _check(
            "b3.algebraic_ring.phi_identity",
            phi * phi == phi + AN.one(),
            f"(φ)²={phi*phi!r}, expected φ+1={phi+AN.one()!r}",
        )
    except Exception as e:
        _fail("b3.algebraic_ring.phi_identity",
              f"{type(e).__name__}: {e}")


def test_b3_fibonacci_binet_exact():
    """B.3's payoff: fibonacci_sym closes to a Binet-style
    ``(φⁿ − ψⁿ)/√5`` expression where ``φ, ψ`` live in
    ``ℚ(√5)``. ``eval_at`` rounds to int and MUST match NumPyExecutor on
    every validation ``n``. The rounding tolerance must be explicit: a
    module-level ``B3_ROUNDING_TOLERANCE`` published so readers can audit
    the approximation-vs-exact tension the issue warns about."""
    mod = _try_import("algebraic_poly")
    if not _have(mod):
        _red("b3.binet_exact", "algebraic_poly module not present")
        return
    tol = getattr(mod, "B3_ROUNDING_TOLERANCE", None)
    _check(
        "b3.binet.tolerance_declared",
        tol is not None,
        "B3_ROUNDING_TOLERANCE must be published (issue warns that every "
        "eval_at call carries approximation-vs-exact tension)",
    )

    wl = getattr(ff, "evaluate_program_forking_weight_layer", None)
    if wl is None:
        _red("b3.binet_exact.entrypoint", "weight-layer missing")
        return

    import programs as P
    for n in (1, 2, 5, 10, 15):
        prog, expected = P.make_fibonacci_sym(n)
        try:
            out = wl(prog, input_mode="symbolic", path="b3").weight_layer_top
        except Exception as e:
            _fail(f"b3.binet_exact[n={n}]",
                  f"raised {type(e).__name__}: {e}")
            continue
        _check(
            f"b3.binet_exact[n={n}]",
            out == expected,
            f"b3={out} expected={expected} — rounding must absorb float slop",
        )


def test_b3_reopens_89_decision_explicitly():
    """#89 explicitly rejected Binet-style algebraic closed forms as a
    non-goal. #107 re-affirmed it. B.3 reopens that decision, which is a
    real process event — it must be marked explicitly in the module
    docstring or a constant so future readers aren't surprised.
    Contract: ``algebraic_poly.REOPENS_ISSUES`` is a list containing at
    least ``89``."""
    mod = _try_import("algebraic_poly")
    if not _have(mod):
        _red("b3.reopens_89", "module not present")
        return
    reopens = getattr(mod, "REOPENS_ISSUES", None)
    _check(
        "b3.reopens_89.declared",
        isinstance(reopens, (list, tuple)) and 89 in reopens,
        f"REOPENS_ISSUES={reopens!r}; must include 89 (Binet was "
        "explicitly rejected there — reopening requires its own marker)",
    )


# ─── Motivator gate (#107 recommendation) ─────────────────────────

def test_motivator_no_silent_activation():
    """Per #107: Path B should not activate silently. If an implementation
    lands, calling the default ``evaluate_program_forking`` must still
    return Path A (solver_structural + eval_at boundary) — the
    weight-layer forward pass must be opt-in via the new entrypoint or
    an explicit flag. This test guards the deferred-by-default posture."""
    import programs as P
    prog, _ = P.make_fibonacci_sym(5)
    fs = ff.evaluate_program_forking(prog, input_mode="symbolic")
    # Path A contract: top is a ClosedForm (sibling), not an integer,
    # and eval_at is still required for a numeric answer.
    _check(
        "motivator.default_is_path_a.top_type",
        isinstance(fs.top, ClosedForm),
        f"default evaluate_program_forking top type = {type(fs.top).__name__}; "
        "must remain ClosedForm (sibling) — Path B must not re-route the "
        "default entrypoint to weight-layer numerics",
    )
    # No "weight_layer_top" leaks into the default return.
    _check(
        "motivator.default_is_path_a.no_leak",
        not hasattr(fs, "weight_layer_top"),
        "default ForkingResult grew a weight_layer_top field — Path B "
        "must not expose its forward-pass result on the default path",
    )


def test_motivator_catalog_row_triggers_path_b():
    """Positive counterpart to ``no_silent_activation``: at least one
    catalog row must have been introduced (or flagged) whose
    ``status = collapsed_closed_form`` and whose metadata explicitly
    requests weight-layer realisation — that's the motivator #107
    demanded before ever writing this code. If no row requests it, the
    implementation shouldn't exist."""
    from symbolic_programs_catalog import _default_catalog
    cat = _default_catalog()
    # Contract: ``CatalogEntry`` gains a ``requests_weight_layer: bool``
    # field (defaulting False). A Path B landing requires at least one
    # entry to set it True — otherwise we're in the "building a gadget
    # for a program nobody runs" anti-pattern the issue calls out. The
    # flag lives on the authoring-time descriptor, not the run result,
    # because it's an authoring decision.
    with_flag = [r for r in cat if getattr(r, "requests_weight_layer", False)]

    # If Path B hasn't been built yet, the flag doesn't even exist — RED.
    if not hasattr(next(iter(cat), object()), "requests_weight_layer"):
        _red("motivator.row_flag_field",
             "CatalogRow.requests_weight_layer field not present — "
             "motivator gate not wired")
        return

    _check(
        "motivator.row_flag_present",
        len(with_flag) >= 1,
        "no catalog row has requests_weight_layer=True; per #107, Path B "
        "shouldn't land without a motivating row",
    )


# ─── Runner ────────────────────────────────────────────────────────

def main() -> None:
    tests = [
        # common
        test_common_entrypoint_exists,
        test_common_solver_structural_preserved,
        test_common_weight_layer_forward_numeric,
        test_common_catalog_ff_equiv_column_extended,
        test_common_scope_predicate_documented,
        # B.1
        test_b1_polynomial_embedding_roundtrip,
        test_b1_exact_on_tier1,
        test_b1_honest_failure_beyond_K,
        test_b1_dmodel_pin_updated,
        # B.2
        test_b2_recurrent_gadget_covers_tier2_and_tier3,
        test_b2_explicit_loop_counter,
        test_b2_explicit_framing_change,
        test_b2_iteration_count_scales_with_n,
        # B.3
        test_b3_algebraic_ring_exists,
        test_b3_fibonacci_binet_exact,
        test_b3_reopens_89_decision_explicitly,
        # motivator
        test_motivator_no_silent_activation,
        test_motivator_catalog_row_triggers_path_b,
    ]
    print("=" * 60)
    print("Path B acceptance tests (issue #109)")
    print("=" * 60)
    for t in tests:
        print(f"\n{t.__name__}:")
        try:
            t()
        except Exception as e:
            _failures.append(
                f"{t.__name__}: uncaught {type(e).__name__}: {e}")
            print(f"  FAIL  {t.__name__}  uncaught "
                  f"{type(e).__name__}: {e}")
            traceback.print_exc()
    print("\n" + "=" * 60)
    print(f"Hard failures: {len(_failures)}")
    print(f"TDD-RED (expected while #109 deferred): {len(_reds)}")
    if _failures:
        print("\nFAILURES:")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    if _reds:
        print("\nTDD-RED entries (unimplemented, not failures):")
        for r in _reds:
            print(f"  - {r}")
    sys.exit(0)


if __name__ == "__main__":
    main()
