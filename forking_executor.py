"""Forking symbolic executor (issue #70) + loop closed-form solver (issue #89).

Extends the branchless :func:`symbolic_executor.run_symbolic` with:

  - **Guarded traces** — JZ/JNZ on a symbolic condition forks into two
    complementary paths (:class:`guarded.Guard`), reconvening at HALT as
    a :class:`guarded.GuardedPoly` case table.
  - **Bounded-loop unrolling** — ``input_mode="concrete"`` pushes literal
    PUSH args so every branch collapses deterministically.
  - **Loop-symbolic detection** — a back-edge visited twice with the
    same (pc, sp) and a still-symbolic condition halts the path with
    :class:`symbolic_executor.SymbolicLoopSymbolic`.
  - **Closed-form recurrences** (issue #89) — the classifier in this
    module walks a loop body, extracts linear / polynomial recurrences,
    and emits :class:`closed_form.ClosedForm` /
    :class:`closed_form.ProductForm` tops for :meth:`eval_at`.

The public entry point is :func:`run_forking`. :func:`collapse_report`
and :func:`guarded_to_mermaid` are reporting helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Mapping, Optional, Tuple, Union

import isa
from isa import (
    _trunc_div,
    _trunc_rem,
    _to_i32,
    MASK32,
)

from arithmetic_ops import ArithmeticOps, DEFAULT_ARITHMETIC_OPS
from bitvec import BitVec
from closed_form import ClosedForm, ProductForm
from guarded import (
    Guard, GuardedPoly,
    RationalStackValue, SymbolicIntAst,
    _canonical_guards, _guards_complementary,
)
from poly import Poly
from symbolic_types import (
    REL_EQ, REL_NE, REL_LT, REL_LE, REL_GT, REL_GE,
    _NEGATE_REL, _REL_SYMBOL,
    IndicatorPoly, RationalPoly, SymbolicRemainder,
)
# Opcode-classification constants + exceptions + SymbolicResult live in
# the (slim) symbolic_executor module.
from symbolic_executor import (
    SymbolicResult,
    SymbolicStackUnderflow,
    SymbolicOpNotSupported,
    SymbolicLoopSymbolic,
    SymbolicPathExplosion,
    _POLY_OPS, _BRANCH_OPS, _FORKING_OPS,
    _CMP_BIN_OPS, _CMP_UNARY_OPS, _CMP_OPS,
    _BIN_OP_RELATION,
    _BIT_BIN_OPS, _BIT_UN_OPS, _BITVEC_OPCODES,
    run_symbolic,  # collapse_report calls this
)

# ─── Forking executor (issue #70) ─────────────────────────────────

# Default caps. Catalog programs stay under these comfortably; the
# symbolic-loop exit trips long before we get close.
DEFAULT_MAX_PATHS = 64
DEFAULT_MAX_STEPS = 50_000


def _as_concrete_int(p: "RationalStackValue") -> Optional[int]:
    """Return integer value if ``p`` has no variables; else None.

    Rational stack values (``RationalPoly`` / ``SymbolicRemainder``) never
    collapse to a concrete int for branching purposes — DIV_S/REM_S past a
    subsequent JZ/JNZ is out of scope for issue #75, so return ``None``
    and let the caller fall into the symbolic-cond path (which will then
    raise when it tries to wrap the value in a Guard).

    :class:`IndicatorPoly` *can* collapse: when its inner ``poly`` is
    concrete, the indicator evaluates to 0 or 1 deterministically and
    JZ/JNZ should follow the branch without forking. Issue #76 needs this
    so ``compare_lt_s(3, 5)`` (concrete inputs) collapses straight to
    ``1`` even when consumed by a later branch.
    """
    if isinstance(p, IndicatorPoly):
        inner = _as_concrete_int(p.poly)
        if inner is None:
            return None
        return 1 if _relation_holds(p.relation, inner) else 0
    if isinstance(p, BitVec):
        # Bit-vector fragment (issue #77). A BitVec with no free
        # variables reduces to an i32 literal via :meth:`eval_at`, so
        # JZ/JNZ on it takes the concretely-decided branch rather than
        # forking. Matches the IndicatorPoly(Poly) path above and
        # unblocks ``popcount_loop(n)`` at concrete ``n``.
        if p.variables():
            return None
        return int(p.eval_at({}))
    if not isinstance(p, Poly):
        return None
    if not p.terms:
        return 0
    if len(p.terms) == 1 and () in p.terms:
        return int(p.terms[()])
    for mono in p.terms:
        if mono:
            return None
    # Only the constant monomial appears (possibly not present).
    return int(p.terms.get((), 0))


@dataclass
class _Path:
    """One symbolic execution thread.

    ``visited_branches`` records ``(pc, sp)`` pairs where this path took
    a symbolic branch; revisiting such a pair with a still-symbolic cond
    at the same site is the loop-symbolic signal.

    Variables are indexed by the PC of the PUSH instruction that
    allocated them, so forked paths sharing a prefix also share the
    variable ids of pre-fork PUSHes — and post-fork PUSHes at distinct
    static sites get distinct ids even across paths.
    """
    pc: int
    stack: Tuple["RationalStackValue", ...]
    guards: Tuple[Guard, ...]
    bindings: Dict[int, int]
    n_heads: int = 0
    visited_branches: frozenset = field(default_factory=frozenset)
    loop_unrolled: bool = False  # True if this path ever took a back-edge
    halted_top: Optional["RationalStackValue"] = None
    locals_: Dict[int, "RationalStackValue"] = field(default_factory=dict)

    def with_(self, **kwargs) -> "_Path":
        """Return a copy with selected fields replaced."""
        base = dict(
            pc=self.pc, stack=self.stack, guards=self.guards,
            bindings=dict(self.bindings),
            n_heads=self.n_heads, visited_branches=self.visited_branches,
            loop_unrolled=self.loop_unrolled, halted_top=self.halted_top,
            locals_=dict(self.locals_),
        )
        base.update(kwargs)
        return _Path(**base)


@dataclass
class ForkingResult:
    """Outcome of running a program via the forking executor.

    ``top`` collapses:
      - to a single :class:`Poly` if all halted paths agree;
      - to a :class:`GuardedPoly` when paths disagree;
      - to ``None`` if no path halted (loop_symbolic-only outcome).

    ``status`` is one of ``"straight" | "guarded" | "unrolled" |
    "loop_symbolic" | "path_explosion" | "blocked_underflow"``.
    """

    top: Optional[Union[Poly, GuardedPoly]]
    status: str
    n_heads: int                  # max k_heads across halted paths
    bindings: Dict[int, int]      # union of all paths' bindings
    n_halted: int = 0
    n_loop_symbolic: int = 0
    paths_explored: int = 0


def _eq_guard(p: Poly, eq_zero: bool) -> Guard:
    """Backwards-compat shim: build an EQ/NE guard from a bool flag.

    Pre-#76 callers built guards with ``eq_zero=True/False``. New code
    should construct :class:`Guard` with the relation directly.
    """
    return Guard(poly=p, relation=REL_EQ if eq_zero else REL_NE)


def _branch_guards(cond: "RationalStackValue", op: int) -> Tuple[Guard, Guard]:
    """Build (take_guard, skip_guard) for a JZ/JNZ on a symbolic ``cond``.

    For a plain :class:`Poly` cond, JZ takes when ``cond == 0`` (skip
    when ``cond != 0``); JNZ flips. For an :class:`IndicatorPoly` cond
    with ``(poly, R)``, "cond == 0" ⇔ "``poly`` does NOT satisfy R" ⇔
    "``poly (negate R) 0``", so we hoist the comparison's polynomial
    and relation into the guard rather than wrapping the indicator
    inside one — that way the resulting :class:`GuardedPoly` carries
    LT/LE/GT/GE guards directly and matches the semantics the catalog
    needs to render with the right ``<``/``<=``/``>``/``>=`` symbols.

    Raises :class:`SymbolicOpNotSupported` for cond types we can't gate
    on (RationalPoly / SymbolicRemainder — DIV_S/REM_S past JZ/JNZ
    isn't in scope yet).
    """
    if isinstance(cond, IndicatorPoly):
        # cond == 0  ⇔  not (poly R 0)  ⇔  poly (negate R) 0
        zero_relation = _NEGATE_REL[cond.relation]
        nonzero_relation = cond.relation
        zero_guard = Guard(poly=cond.poly, relation=zero_relation)
        nonzero_guard = Guard(poly=cond.poly, relation=nonzero_relation)
    elif isinstance(cond, Poly):
        zero_guard = Guard(poly=cond, relation=REL_EQ)
        nonzero_guard = Guard(poly=cond, relation=REL_NE)
    else:
        # BitVec / RationalPoly / SymbolicRemainder with free variables:
        # out of scope. Concrete-mode BitVec conds are handled earlier
        # by :func:`_as_concrete_int` (issue #77); this branch only
        # fires for truly symbolic non-Poly conds.
        raise SymbolicOpNotSupported(
            f"JZ/JNZ on a {type(cond).__name__} cond is out of scope; "
            "branching past DIV_S/REM_S is a follow-up"
        )
    # JZ: take when cond == 0. JNZ: take when cond != 0.
    if op == isa.OP_JZ:
        return zero_guard, nonzero_guard
    return nonzero_guard, zero_guard


def run_forking(prog: List[isa.Instruction], *,
                input_mode: str = "symbolic",
                max_paths: int = DEFAULT_MAX_PATHS,
                max_steps: int = DEFAULT_MAX_STEPS,
                arithmetic_ops: Optional[ArithmeticOps] = None,
                solve_recurrences: bool = True) -> ForkingResult:
    """Forking symbolic executor with finite-conditional + bounded-loop support.

    ``input_mode``:
      - ``"symbolic"``: each PUSH allocates a fresh variable. Branches on
        symbolic conditions fork the path. Suitable for ``collapsed_guarded``.
      - ``"concrete"``: each PUSH pushes its literal arg (no variables).
        All branches collapse deterministically; loops unroll naturally.
        Suitable for ``collapsed_unrolled``.

    ``arithmetic_ops``: override the ADD/SUB/MUL primitives applied to
    Poly stack entries. Defaults to :data:`DEFAULT_ARITHMETIC_OPS` (plain
    Poly ``+ - *``). :mod:`ff_symbolic` passes its bilinear-FF
    interpretation here (issue #68 S3) to demonstrate equivalence across
    control flow.

    ``solve_recurrences`` (issue #89): when ``True`` (default), paths
    that would otherwise halt with ``loop_symbolic`` are routed through
    :func:`_try_solve_recurrence`. On success the loop is replaced with
    a single closed-form halted path carrying a :class:`Poly` /
    :class:`ClosedForm` / :class:`ProductForm` top; the status becomes
    ``"closed_form"``. On failure the path falls through to the
    unchanged ``loop_symbolic`` behaviour. Set to ``False`` to reproduce
    the pre-#89 ``blocked_loop_symbolic`` path exactly.

    The executor uses a worklist. Each fork splits the path into two new
    paths carrying complementary guards. When a path's top polynomial is
    concrete at a branch, the branch is followed deterministically. A
    symbolic back-edge that revisits ``(pc, sp)`` halts the path with
    ``loop_symbolic`` (unless ``solve_recurrences`` catches it first).
    """
    if input_mode not in ("symbolic", "concrete"):
        raise ValueError(f"unknown input_mode {input_mode!r}")
    ops = arithmetic_ops if arithmetic_ops is not None else DEFAULT_ARITHMETIC_OPS
    # Closed-form paths accumulate here when `_try_solve_recurrence`
    # succeeds at a back-edge revisit; they're merged alongside
    # ``halted`` at the end and flipped to status="closed_form".
    closed_form_paths: List[_Path] = []

    # Pre-flight: reject programs with non-polynomial, non-branch opcodes.
    for instr in prog:
        if instr.op not in _FORKING_OPS:
            name = isa.OP_NAMES.get(instr.op, f"?{instr.op}")
            raise SymbolicOpNotSupported(
                f"op {name!r} is out of scope for the forking executor "
                f"(polynomial + JZ/JNZ only)"
            )

    init = _Path(
        pc=0, stack=(), guards=(),
        bindings={}, n_heads=0,
        visited_branches=frozenset(), loop_unrolled=False,
    )
    worklist: List[_Path] = [init]
    halted: List[_Path] = []
    loop_symbolic_paths: List[_Path] = []
    paths_explored = 0
    total_steps = 0
    underflow_seen = False

    def _spawn(new: _Path):
        if len(worklist) + len(halted) + len(loop_symbolic_paths) + 1 > max_paths:
            raise SymbolicPathExplosion(
                f"path count exceeds max_paths={max_paths}"
            )
        worklist.append(new)

    try:
        while worklist:
            path = worklist.pop()
            paths_explored += 1
            # Step this path until it halts, forks, or loops symbolically.
            while True:
                total_steps += 1
                if total_steps > max_steps:
                    raise SymbolicPathExplosion(
                        f"total step count exceeds max_steps={max_steps}"
                    )
                if path.pc < 0 or path.pc >= len(prog):
                    # Implicit fall-off-end acts as HALT with current top.
                    path = path.with_(
                        halted_top=path.stack[-1] if path.stack
                        else Poly.constant(0),
                    )
                    halted.append(path)
                    break
                instr = prog[path.pc]
                op = instr.op
                if op == isa.OP_HALT:
                    path = path.with_(
                        halted_top=path.stack[-1] if path.stack
                        else Poly.constant(0),
                    )
                    halted.append(path)
                    break

                # Non-branch, non-halt → advance n_heads and apply op.
                if op != isa.OP_JZ and op != isa.OP_JNZ:
                    try:
                        stack, new_locals = _apply_poly_op(path, instr, input_mode, ops)
                    except SymbolicStackUnderflow:
                        underflow_seen = True
                        # drop this path; don't propagate partial result
                        break
                    new_bindings = path.bindings
                    if op == isa.OP_PUSH and input_mode == "symbolic":
                        # Variable id = PUSH's pc (stable across forked paths).
                        new_bindings = dict(path.bindings)
                        new_bindings[path.pc] = int(instr.arg)
                    with_kwargs = dict(
                        pc=path.pc + 1,
                        stack=stack,
                        n_heads=path.n_heads + (0 if op == isa.OP_NOP else 1),
                        bindings=new_bindings,
                    )
                    if new_locals is not None:
                        with_kwargs["locals_"] = new_locals
                    path = path.with_(**with_kwargs)
                    continue

                # JZ / JNZ: pop cond, decide branch.
                if not path.stack:
                    underflow_seen = True
                    break
                cond = path.stack[-1]
                popped_stack = path.stack[:-1]
                path = path.with_(
                    n_heads=path.n_heads + 1,
                    stack=popped_stack,
                )
                sp = len(popped_stack)
                target = int(instr.arg)
                fall_through = path.pc + 1
                is_back_edge = target <= path.pc

                concrete = _as_concrete_int(cond)
                if concrete is not None:
                    taken = (concrete == 0) if op == isa.OP_JZ else (concrete != 0)
                    new_pc = target if taken else fall_through
                    path = path.with_(
                        pc=new_pc,
                        loop_unrolled=path.loop_unrolled or (is_back_edge and taken),
                    )
                    continue

                # Symbolic condition → fork. Check for symbolic back-edge revisit.
                site = (path.pc, sp, op)
                if is_back_edge and site in path.visited_branches:
                    # This path already forked at this back-edge once with
                    # a symbolic cond — seeing it again means no progress.
                    # Issue #89: before halting with loop_symbolic, try
                    # to derive a closed form for the loop.
                    closed_top = None
                    if solve_recurrences:
                        # Re-attach cond to the stack for the solver so
                        # its "back-edge state" view matches the
                        # executor's (``path`` here already popped the
                        # cond above).
                        solver_path = path.with_(
                            stack=path.stack + (cond,),
                        )
                        closed_top = _try_solve_recurrence(
                            prog, solver_path, input_mode, ops,
                        )
                    if closed_top is not None:
                        # Replace the loop with a single closed-form
                        # halted path. Guards stay empty — the closed
                        # form covers all iteration counts, so prior
                        # iteration-specific halts are subsumed.
                        closed_path = path.with_(
                            halted_top=closed_top,
                            n_heads=path.n_heads,
                            guards=(),
                        )
                        closed_form_paths.append(closed_path)
                        break
                    loop_symbolic_paths.append(path)
                    break
                new_visited = path.visited_branches | {site}

                # _branch_guards understands both bare-Poly and
                # IndicatorPoly conds — for the latter it hoists the
                # comparison's relation directly into the guards.
                take_guard, skip_guard = _branch_guards(cond, op)

                take_path = path.with_(
                    pc=target,
                    guards=path.guards + (take_guard,),
                    visited_branches=new_visited,
                )
                skip_path = path.with_(
                    pc=fall_through,
                    guards=path.guards + (skip_guard,),
                    visited_branches=new_visited,
                )
                _spawn(skip_path)
                _spawn(take_path)
                break  # current thread is replaced by the two new ones
    except SymbolicPathExplosion:
        # Collect what we have and report.
        return ForkingResult(
            top=None, status="path_explosion",
            n_heads=max(
                (p.n_heads for p in halted + loop_symbolic_paths
                 + closed_form_paths),
                default=0,
            ),
            bindings={}, n_halted=len(halted),
            n_loop_symbolic=len(loop_symbolic_paths),
            paths_explored=paths_explored,
        )

    # Issue #89: closed-form halted paths. When the solver fires, the
    # closed-form top covers all iteration counts of the loop, which
    # subsumes the iteration-specific halts produced before the
    # back-edge revisit. Keeping the per-iteration halted paths
    # alongside the closed form would duplicate signal and force the
    # output into a GuardedPoly — so we treat ``closed_form_paths`` as
    # the authoritative output whenever it's non-empty.
    if closed_form_paths:
        # Merge bindings from every path (pre-loop prefix state is
        # already baked into the closed form, but other paths may carry
        # bindings the caller wants for the three-way eval check).
        merged_bindings: Dict[int, int] = {}
        for p in closed_form_paths + halted + loop_symbolic_paths:
            merged_bindings.update(p.bindings)
        cf_tops = {p.halted_top for p in closed_form_paths}
        if len(cf_tops) == 1:
            top_val: Union[Poly, GuardedPoly, ClosedForm, ProductForm] = next(iter(cf_tops))
        else:
            # Multiple distinct closed forms — shouldn't happen for the
            # single-loop catalog programs, but fall back to the first.
            top_val = closed_form_paths[0].halted_top
        n_heads = max((p.n_heads for p in closed_form_paths), default=0)
        return ForkingResult(
            top=top_val, status="closed_form", n_heads=n_heads,
            bindings=merged_bindings, n_halted=len(closed_form_paths),
            n_loop_symbolic=len(loop_symbolic_paths),
            paths_explored=paths_explored,
        )

    # Combine halted tops.
    if not halted:
        if loop_symbolic_paths:
            return ForkingResult(
                top=None, status="loop_symbolic", n_heads=0,
                bindings={}, n_halted=0,
                n_loop_symbolic=len(loop_symbolic_paths),
                paths_explored=paths_explored,
            )
        if underflow_seen:
            return ForkingResult(
                top=None, status="blocked_underflow", n_heads=0,
                bindings={}, n_halted=0, n_loop_symbolic=0,
                paths_explored=paths_explored,
            )
        return ForkingResult(
            top=None, status="blocked_underflow", n_heads=0,
            bindings={}, n_halted=0, n_loop_symbolic=0,
            paths_explored=paths_explored,
        )

    merged_bindings: Dict[int, int] = {}
    for p in halted:
        merged_bindings.update(p.bindings)

    tops = [(p.guards, p.halted_top) for p in halted]
    # Determine status.
    any_forked = any(p.guards for p in halted)
    any_looped = any(p.loop_unrolled for p in halted) or bool(loop_symbolic_paths)

    # Build the top: a single Poly if all agree and no guards, else GuardedPoly.
    unique_values = {v for _, v in tops}
    if not any_forked and len(unique_values) == 1:
        top_val = next(iter(unique_values))
    else:
        top_val = _build_guarded_poly(tops)

    if loop_symbolic_paths and not halted:
        status = "loop_symbolic"
    elif loop_symbolic_paths:
        # Partial collapse: some paths halted, others hit symbolic loops.
        status = "loop_symbolic"
    elif any_looped:
        status = "unrolled"
    elif any_forked:
        status = "guarded"
    else:
        status = "straight"

    n_heads = max((p.n_heads for p in halted), default=0)
    return ForkingResult(
        top=top_val, status=status, n_heads=n_heads,
        bindings=merged_bindings, n_halted=len(halted),
        n_loop_symbolic=len(loop_symbolic_paths),
        paths_explored=paths_explored,
    )


def _apply_poly_op(path: _Path, instr: isa.Instruction,
                   input_mode: str,
                   arithmetic_ops: ArithmeticOps = DEFAULT_ARITHMETIC_OPS
                   ) -> Tuple[Tuple[RationalStackValue, ...],
                              Optional[Dict[int, RationalStackValue]]]:
    """Apply a non-branch opcode to ``path.stack`` and return the new stack.

    Returns ``(new_stack, new_locals_or_None)``. ``new_locals`` is
    non-``None`` only when the op mutated the locals store (LOCAL_SET /
    LOCAL_TEE, issue #102 ops); callers should fold it into the path
    via :meth:`_Path.with_`. A ``None`` lets non-LOCAL ops stay
    allocation-free.

    ``arithmetic_ops`` picks the ADD/SUB/MUL/DIV_S/REM_S implementations.
    Defaults to Poly's native operators (plus RationalPoly/SymbolicRemainder
    wrappers for DIV_S/REM_S); :mod:`ff_symbolic` passes its bilinear-FF
    primitives.
    """
    op = instr.op
    stack = list(path.stack)

    def _pop() -> RationalStackValue:
        if not stack:
            raise SymbolicStackUnderflow(f"pop from empty stack at pc={path.pc}")
        return stack.pop()

    if op == isa.OP_PUSH:
        if input_mode == "symbolic":
            stack.append(Poly.variable(path.pc))
        else:
            stack.append(Poly.constant(int(instr.arg)))
    elif op == isa.OP_POP:
        _pop()
    elif op == isa.OP_DUP:
        if not stack:
            raise SymbolicStackUnderflow(f"dup on empty stack at pc={path.pc}")
        stack.append(stack[-1])
    elif op == isa.OP_ADD:
        b = _pop(); a = _pop()
        if isinstance(a, BitVec) or isinstance(b, BitVec):
            if not isinstance(a, (Poly, BitVec)) or not isinstance(b, (Poly, BitVec)):
                raise SymbolicOpNotSupported(
                    "ADD mixing BitVec with rational/indicator entries is out of scope"
                )
            if arithmetic_ops.bit_add is None:
                raise SymbolicOpNotSupported(
                    "arithmetic_ops.bit_add is not wired; pass a bit_add primitive"
                )
            stack.append(arithmetic_ops.bit_add(a, b))
        elif not isinstance(a, Poly) or not isinstance(b, Poly):
            raise SymbolicOpNotSupported(
                "ADD on rational stack entries is out of scope "
                "(composition past one DIV_S/REM_S is a follow-up)"
            )
        else:
            stack.append(arithmetic_ops.add(a, b))
    elif op == isa.OP_SUB:
        b = _pop(); a = _pop()
        if isinstance(a, BitVec) or isinstance(b, BitVec):
            if not isinstance(a, (Poly, BitVec)) or not isinstance(b, (Poly, BitVec)):
                raise SymbolicOpNotSupported(
                    "SUB mixing BitVec with rational/indicator entries is out of scope"
                )
            if arithmetic_ops.bit_sub is None:
                raise SymbolicOpNotSupported(
                    "arithmetic_ops.bit_sub is not wired; pass a bit_sub primitive"
                )
            stack.append(arithmetic_ops.bit_sub(a, b))
        elif not isinstance(a, Poly) or not isinstance(b, Poly):
            raise SymbolicOpNotSupported(
                "SUB on rational stack entries is out of scope "
                "(composition past one DIV_S/REM_S is a follow-up)"
            )
        else:
            stack.append(arithmetic_ops.sub(a, b))
    elif op == isa.OP_MUL:
        b = _pop(); a = _pop()
        if isinstance(a, BitVec) or isinstance(b, BitVec):
            if not isinstance(a, (Poly, BitVec)) or not isinstance(b, (Poly, BitVec)):
                raise SymbolicOpNotSupported(
                    "MUL mixing BitVec with rational/indicator entries is out of scope"
                )
            if arithmetic_ops.bit_mul is None:
                raise SymbolicOpNotSupported(
                    "arithmetic_ops.bit_mul is not wired; pass a bit_mul primitive"
                )
            stack.append(arithmetic_ops.bit_mul(a, b))
        elif not isinstance(a, Poly) or not isinstance(b, Poly):
            raise SymbolicOpNotSupported(
                "MUL on rational stack entries is out of scope "
                "(composition past one DIV_S/REM_S is a follow-up)"
            )
        else:
            stack.append(arithmetic_ops.mul(a, b))
    elif op == isa.OP_DIV_S:
        b = _pop(); a = _pop()
        if not isinstance(a, Poly) or not isinstance(b, Poly):
            raise SymbolicOpNotSupported(
                "DIV_S on rational stack entries is out of scope "
                "(composition past one DIV_S/REM_S is a follow-up)"
            )
        if arithmetic_ops.div_s is None:
            raise SymbolicOpNotSupported(
                "arithmetic_ops.div_s is not wired; pass a div_s primitive"
            )
        stack.append(arithmetic_ops.div_s(a, b))
    elif op == isa.OP_REM_S:
        b = _pop(); a = _pop()
        if not isinstance(a, Poly) or not isinstance(b, Poly):
            raise SymbolicOpNotSupported(
                "REM_S on rational stack entries is out of scope "
                "(composition past one DIV_S/REM_S is a follow-up)"
            )
        if arithmetic_ops.rem_s is None:
            raise SymbolicOpNotSupported(
                "arithmetic_ops.rem_s is not wired; pass a rem_s primitive"
            )
        stack.append(arithmetic_ops.rem_s(a, b))
    elif op in _CMP_BIN_OPS:
        b = _pop(); a = _pop()
        if isinstance(a, Poly) and isinstance(b, Poly):
            cmp_fn = arithmetic_ops.cmp(op)
            if cmp_fn is None:
                raise SymbolicOpNotSupported(
                    f"arithmetic_ops.cmp({isa.OP_NAMES[op]}) is not wired"
                )
            stack.append(cmp_fn(a, b))
        elif isinstance(a, (Poly, BitVec)) and isinstance(b, (Poly, BitVec)):
            # is_power_of_2 composes POPCNT (BitVec) with PUSH 1 (Poly).
            # Matches the Poly path's ``a - b`` = ``SP-1 - top`` difference.
            stack.append(IndicatorPoly(
                poly=BitVec(op="SUB", operands=(a, b)),
                relation=_BIN_OP_RELATION[op],
            ))
        else:
            raise SymbolicOpNotSupported(
                f"{isa.OP_NAMES[op]} on non-Poly stack entries is out "
                "of scope (composition past one DIV_S/REM_S/comparison "
                "is a follow-up)"
            )
    elif op == isa.OP_EQZ:
        a = _pop()
        if isinstance(a, Poly):
            if arithmetic_ops.eqz is None:
                raise SymbolicOpNotSupported(
                    "arithmetic_ops.eqz is not wired; pass an eqz primitive"
                )
            stack.append(arithmetic_ops.eqz(a))
        elif isinstance(a, BitVec):
            stack.append(IndicatorPoly(poly=a, relation=REL_EQ))
        else:
            raise SymbolicOpNotSupported(
                "EQZ on non-Poly stack entries is out of scope "
                "(composition past one DIV_S/REM_S/comparison is a follow-up)"
            )
    elif op == isa.OP_SWAP:
        if len(stack) < 2:
            raise SymbolicStackUnderflow(f"swap needs 2 entries at pc={path.pc}")
        stack[-1], stack[-2] = stack[-2], stack[-1]
    elif op == isa.OP_OVER:
        if len(stack) < 2:
            raise SymbolicStackUnderflow(f"over needs 2 entries at pc={path.pc}")
        stack.append(stack[-2])
    elif op == isa.OP_ROT:
        if len(stack) < 3:
            raise SymbolicStackUnderflow(f"rot needs 3 entries at pc={path.pc}")
        a, b, c = stack[-3], stack[-2], stack[-1]
        stack[-3], stack[-2], stack[-1] = b, c, a
    elif op in _BIT_BIN_OPS:
        b = _pop(); a = _pop()
        if not isinstance(a, (Poly, BitVec)) or not isinstance(b, (Poly, BitVec)):
            raise SymbolicOpNotSupported(
                f"{isa.OP_NAMES[op]} on rational/indicator entries is out of scope"
            )
        bit_fn = arithmetic_ops.bit_binary(op)
        if bit_fn is None:
            raise SymbolicOpNotSupported(
                f"arithmetic_ops.bit_binary({isa.OP_NAMES[op]}) is not wired"
            )
        stack.append(bit_fn(a, b))
    elif op in _BIT_UN_OPS:
        a = _pop()
        if not isinstance(a, (Poly, BitVec)):
            raise SymbolicOpNotSupported(
                f"{isa.OP_NAMES[op]} on rational/indicator entries is out of scope"
            )
        bit_fn = arithmetic_ops.bit_unary(op)
        if bit_fn is None:
            raise SymbolicOpNotSupported(
                f"arithmetic_ops.bit_unary({isa.OP_NAMES[op]}) is not wired"
            )
        stack.append(bit_fn(a))
    elif op == isa.OP_NOP:
        pass
    elif op == isa.OP_LOCAL_GET:
        # Slot read — polynomial-closed by construction (issues #100, #102).
        # Uninitialized slot is a hard error (matches run_symbolic).
        if instr.arg not in path.locals_:
            raise SymbolicStackUnderflow(
                f"LOCAL_GET of uninitialized slot {instr.arg}"
            )
        stack.append(path.locals_[instr.arg])
    elif op == isa.OP_LOCAL_SET:
        # Pops top → writes to slot. Caller must fold new_locals into path.
        if not stack:
            raise SymbolicStackUnderflow("LOCAL_SET on empty stack")
        new_locals = dict(path.locals_)
        new_locals[instr.arg] = _pop()
        return tuple(stack), new_locals
    elif op == isa.OP_LOCAL_TEE:
        # Peeks top (leaves it on stack) → writes to slot.
        if not stack:
            raise SymbolicStackUnderflow("LOCAL_TEE on empty stack")
        new_locals = dict(path.locals_)
        new_locals[instr.arg] = stack[-1]
        return tuple(stack), new_locals
    else:  # pragma: no cover
        raise SymbolicOpNotSupported(f"op {op} unexpected in _apply_poly_op")

    return tuple(stack), None


def _build_guarded_poly(
    tops: List[Tuple[Tuple[Guard, ...], Poly]],
) -> Union[Poly, GuardedPoly]:
    """Merge per-path ``(guards, value)`` pairs into a single GuardedPoly.

    Paths with the same value polynomial are combined by merging their
    guard chains. If all paths produce the same value *and* their guards
    together span the full domain (the trivial case of a single path
    with empty guards), we return the bare Poly.
    """
    # Group by value poly.
    by_value: Dict[Poly, List[Tuple[Guard, ...]]] = {}
    for gs, v in tops:
        by_value.setdefault(v, []).append(_canonical_guards(gs))

    # If only one distinct value and at least one path has no guards, it's unconditional.
    if len(by_value) == 1:
        sole_value = next(iter(by_value))
        guard_chains = by_value[sole_value]
        if any(len(gs) == 0 for gs in guard_chains):
            return sole_value

    cases: List[Tuple[Tuple[Guard, ...], Poly]] = []
    for value, guard_chains in by_value.items():
        for gs in guard_chains:
            cases.append((gs, value))
    return GuardedPoly(cases=tuple(cases))


# ─── Loop-invariant inference (issue #89) ────────────────────────
#
# The pipeline outlined in ``dev/loop_invariant_inference.md``:
#
#   1. Detect the back-edge that triggered loop_symbolic. The target of
#      that JZ/JNZ is the loop header.
#   2. Re-execute the program prefix (pc=0..loop_header_pc-1) straight-
#      line to recover the initial loop state.
#   3. Execute one body iteration starting at loop_header_pc with fresh
#      "slot" variables for each loop-carried stack entry, following
#      "stay in loop" decisions at any JZ/JNZ inside the body. The
#      transition polynomials fall out as the stack at back_edge_pc.
#   4. Classify:
#        - Tier 1 — affine with a polynomial driver in the counter.
#          Closed form stays in :class:`Poly` (Faulhaber).
#        - Tier 2 — linear map with constant integer coefficients on
#          the dependent slots. Emits :class:`ClosedForm`.
#        - Tier 3 — ``acc <- acc * p(counter)`` with linear counter
#          decrement. Emits :class:`ProductForm`.
#
# Keeping the solver self-contained below ``_build_guarded_poly`` so
# ``run_forking`` only carries a single hook site — no entanglement
# with the forking driver's worklist / guard machinery.


# Slot variables for the fresh loop-carried inputs. High indices so
# they never collide with real PC-indexed variables.
_SLOT_VAR_BASE = 1_000_000


def _substitute_poly(poly: Poly, subs: Mapping[int, int]) -> Poly:
    """Partial substitution: replace ``x_v`` with ``subs[v]`` for each
    ``v in subs``; leave other variables symbolic.

    Returns a new :class:`Poly`. Used to fold body-local PUSH
    constants (the ``PUSH 1; SUB`` / ``PUSH 1; JNZ`` sites inside a
    loop body) into the transition polynomials so the classifier sees
    clean expressions in the slot variables alone.
    """
    out: Dict[Monomial, Union[int, Fraction]] = {}
    for mono, coeff in poly.terms.items():
        new_mono: List[Tuple[int, int]] = []
        factor: Union[int, Fraction] = 1
        for v, p in mono:
            if v in subs:
                factor *= subs[v] ** p
            else:
                new_mono.append((v, p))
        key = tuple(new_mono)
        out[key] = out.get(key, 0) + coeff * factor
    return Poly(out)


def _run_prefix(prog: List[isa.Instruction], end_pc: int,
                input_mode: str,
                ops: ArithmeticOps
                ) -> Optional[Tuple[Tuple["RationalStackValue", ...], Dict[int, int]]]:
    """Straight-line execute ``prog[:end_pc]`` and return ``(stack, bindings)``.

    Returns ``None`` if the prefix contains a symbolic JZ/JNZ (can't
    unambiguously enter the loop header) or raises underflow. Concrete
    JZ/JNZ (e.g. PUSH k; JNZ target) are followed deterministically.
    """
    path = _Path(
        pc=0, stack=(), guards=(),
        bindings={}, n_heads=0,
        visited_branches=frozenset(), loop_unrolled=False,
    )
    steps = 0
    while path.pc < end_pc and path.pc < len(prog):
        steps += 1
        if steps > 10_000:
            return None
        instr = prog[path.pc]
        op = instr.op
        if op == isa.OP_HALT:
            return None
        if op not in (isa.OP_JZ, isa.OP_JNZ):
            try:
                stack, new_locals = _apply_poly_op(path, instr, input_mode, ops)
            except (SymbolicStackUnderflow, SymbolicOpNotSupported):
                return None
            new_bindings = path.bindings
            if op == isa.OP_PUSH and input_mode == "symbolic":
                new_bindings = dict(path.bindings)
                new_bindings[path.pc] = int(instr.arg)
            with_kwargs = dict(
                pc=path.pc + 1,
                stack=stack,
                bindings=new_bindings,
                n_heads=path.n_heads + (0 if op == isa.OP_NOP else 1),
            )
            if new_locals is not None:
                with_kwargs["locals_"] = new_locals
            path = path.with_(**with_kwargs)
            continue
        # JZ / JNZ: only concrete conds are OK in a prefix.
        if not path.stack:
            return None
        cond = path.stack[-1]
        popped = path.stack[:-1]
        path = path.with_(stack=popped, n_heads=path.n_heads + 1)
        concrete = _as_concrete_int(cond)
        if concrete is None:
            # Try one more resolution: substitute current bindings, then
            # check for constancy. Handles ``PUSH 1; JNZ`` in symbolic
            # mode where the PUSH allocated a var bound to 1.
            if isinstance(cond, Poly):
                resolved = _substitute_poly(cond, path.bindings)
                if not resolved.variables():
                    concrete = int(resolved.eval_at({}))
        if concrete is None:
            return None  # symbolic branch in prefix — out of scope
        taken = (concrete == 0) if op == isa.OP_JZ else (concrete != 0)
        path = path.with_(pc=int(instr.arg) if taken else path.pc + 1)
    if path.pc != end_pc:
        return None
    return tuple(path.stack), dict(path.bindings)


def _run_body_iteration(
    prog: List[isa.Instruction],
    loop_header_pc: int,
    back_edge_pc: int,
    n_slots: int,
    input_mode: str,
    ops: ArithmeticOps,
) -> Optional[Tuple[Tuple["RationalStackValue", ...], Dict[int, int]]]:
    """Execute one loop body from ``loop_header_pc`` with ``n_slots``
    fresh slot variables as the initial stack.

    Returns ``(transition_stack, body_bindings)`` where
    ``transition_stack`` is the stack just after the back-edge cond
    has been popped (so its length is ``n_slots`` again, describing
    the updated loop-carried slice), or ``None`` if the body can't be
    closed (symbolic branch inside the loop that isn't the back-edge,
    non-trivial exit shape, etc.).

    ``body_bindings`` maps every PC-indexed PUSH inside the body to
    its concrete arg value so the caller can substitute them out.
    """
    if loop_header_pc >= back_edge_pc or back_edge_pc >= len(prog):
        return None
    back_edge_op = prog[back_edge_pc].op
    if back_edge_op not in (isa.OP_JZ, isa.OP_JNZ):
        return None

    # Fresh slot variables — high PC-free indices.
    init_stack: Tuple["RationalStackValue", ...] = tuple(
        Poly.variable(_SLOT_VAR_BASE + i) for i in range(n_slots)
    )
    path = _Path(
        pc=loop_header_pc, stack=init_stack, guards=(),
        bindings={}, n_heads=0,
        visited_branches=frozenset(), loop_unrolled=False,
    )

    def _resolve_concrete(cond: "RationalStackValue",
                          bindings: Dict[int, int]) -> Optional[int]:
        c = _as_concrete_int(cond)
        if c is not None:
            return c
        if isinstance(cond, Poly):
            resolved = _substitute_poly(cond, bindings)
            if not resolved.variables():
                return int(resolved.eval_at({}))
        return None

    steps = 0
    while True:
        steps += 1
        if steps > 10_000:
            return None
        if path.pc == back_edge_pc:
            # Body complete: the back-edge's cond is on top, pop it
            # and return the slot transition.
            if not path.stack:
                return None
            cond = path.stack[-1]
            resolved = _resolve_concrete(cond, path.bindings)
            # We must be guaranteed to take the back-edge (else this
            # isn't a fixed-shape loop). PUSH k; JZ back when k == 0,
            # PUSH k; JNZ back when k != 0. If the cond doesn't
            # resolve to the "take" value, inference fails.
            if resolved is None:
                return None
            taken = (resolved == 0) if back_edge_op == isa.OP_JZ else (resolved != 0)
            if not taken:
                return None
            trans_stack = path.stack[:-1]
            if len(trans_stack) != n_slots:
                # Stack depth must be preserved across one iteration.
                return None
            return trans_stack, dict(path.bindings)
        if path.pc < 0 or path.pc >= len(prog):
            return None
        instr = prog[path.pc]
        op = instr.op
        if op == isa.OP_HALT:
            return None
        if op not in (isa.OP_JZ, isa.OP_JNZ):
            try:
                stack, new_locals = _apply_poly_op(path, instr, input_mode, ops)
            except (SymbolicStackUnderflow, SymbolicOpNotSupported):
                return None
            new_bindings = path.bindings
            if op == isa.OP_PUSH and input_mode == "symbolic":
                new_bindings = dict(path.bindings)
                new_bindings[path.pc] = int(instr.arg)
            with_kwargs = dict(
                pc=path.pc + 1,
                stack=stack,
                bindings=new_bindings,
                n_heads=path.n_heads + (0 if op == isa.OP_NOP else 1),
            )
            if new_locals is not None:
                with_kwargs["locals_"] = new_locals
            path = path.with_(**with_kwargs)
            continue
        # JZ/JNZ inside the body (pc < back_edge_pc): must be the
        # exit test. Always take the "stay in loop" branch.
        if not path.stack:
            return None
        cond = path.stack[-1]
        popped = path.stack[:-1]
        path = path.with_(stack=popped, n_heads=path.n_heads + 1)
        target = int(instr.arg)
        fall_through = path.pc + 1
        concrete = _resolve_concrete(cond, path.bindings)
        if concrete is not None:
            taken = (concrete == 0) if op == isa.OP_JZ else (concrete != 0)
            path = path.with_(pc=target if taken else fall_through)
            continue
        # Symbolic exit: pick the branch that stays inside the loop body.
        in_range = lambda p: loop_header_pc <= p < back_edge_pc + 1
        if op == isa.OP_JZ:
            # JZ takes when cond == 0 (exit); skip stays in loop.
            stay_pc = fall_through if in_range(fall_through) else target
        else:
            # JNZ takes when cond != 0. For the exit test inside a
            # loop, the "take" direction is usually the continue-loop
            # direction. Pick whichever target is inside the body.
            stay_pc = target if in_range(target) else fall_through
        if not in_range(stay_pc):
            return None
        path = path.with_(pc=stay_pc)


def _extract_linear(poly: Poly, slot_vars: List[int]
                    ) -> Optional[Tuple[Dict[int, int], int]]:
    """If ``poly`` is affine in ``slot_vars`` with integer coefficients,
    return ``(coeffs, const)`` where ``coeffs[v]`` is the coefficient
    on slot ``v`` and ``const`` is the constant term. Otherwise
    return ``None``.

    Any monomial with non-slot variables, non-integer coefficients,
    or degree > 1 in slot vars disqualifies — the transition isn't
    a pure linear map over the slot slice.
    """
    coeffs: Dict[int, int] = {v: 0 for v in slot_vars}
    const: int = 0
    slot_set = set(slot_vars)
    for mono, c in poly.terms.items():
        if isinstance(c, Fraction):
            if c.denominator != 1:
                return None
            c = int(c.numerator)
        if not mono:
            const = int(c)
            continue
        if len(mono) != 1:
            return None
        v, p = mono[0]
        if v not in slot_set or p != 1:
            return None
        coeffs[v] = int(c)
    return coeffs, const


def _find_counter_slot(transition: Tuple["RationalStackValue", ...],
                       slot_vars: List[int]
                       ) -> Optional[int]:
    """Return index ``i`` such that ``transition[i] == slot_vars[i] - 1``
    (the counter slot), or ``None`` if no such slot exists.

    A loop's counter decrements by 1 per iteration and tests against
    zero at the exit. Identifying it lets the classifier separate the
    pure-linear "dependent" slots from the counter-driven dynamics.
    """
    for i, v in enumerate(slot_vars):
        t = transition[i]
        if not isinstance(t, Poly):
            continue
        expected = Poly.variable(v) - Poly.constant(1)
        if t == expected:
            return i
    return None


def _classify_recurrence(
    initial_slots: Tuple["RationalStackValue", ...],
    transition: Tuple["RationalStackValue", ...],
    body_bindings: Dict[int, int],
    prefix_bindings: Optional[Dict[int, int]] = None,
) -> Optional[Tuple[str, "RationalStackValue"]]:
    """Classify the loop transition and build the closed form.

    Returns ``(tier, closed_form_top)`` where ``tier`` is one of
    ``"tier1" | "tier2" | "tier3"`` and ``closed_form_top`` is a
    :class:`Poly` (Tier 1) / :class:`ClosedForm` (Tier 2) /
    :class:`ProductForm` (Tier 3) representing the projected slot
    after the loop terminates. Returns ``None`` if none of the tiers
    match.

    The projected slot is the conventional "output" of the loop — for
    the catalog's four target programs it's either the accumulator
    (sum / factorial / power_of_2) or the second Fibonacci term. We
    pick it as the non-counter slot with the most-complex initial
    state, falling back to the last non-counter slot.
    """
    m = len(transition)
    if m == 0 or m != len(initial_slots):
        return None
    slot_vars = [_SLOT_VAR_BASE + i for i in range(m)]

    # Substitute body-local PUSH bindings so each transition poly is a
    # "clean" expression over slot_vars only.
    substituted: List[Poly] = []
    for t in transition:
        if not isinstance(t, Poly):
            return None
        substituted.append(_substitute_poly(t, body_bindings))

    # Every initial slot is expected to be a Poly over the input PUSH
    # variables (not the slot vars — those are fresh to the body run).
    for s in initial_slots:
        if not isinstance(s, Poly):
            return None
        for v in s.variables():
            if v >= _SLOT_VAR_BASE:
                return None  # slot vars must not appear in init

    counter_idx = _find_counter_slot(tuple(substituted), slot_vars)
    if counter_idx is None:
        return None
    counter_slot_var = slot_vars[counter_idx]
    trip_count: Poly = initial_slots[counter_idx]  # type: ignore[assignment]

    # Identify dependent slots (everything except the counter).
    dep_indices = [i for i in range(m) if i != counter_idx]

    # ── Tier 2 — linear recurrence with constant integer matrix ──
    # Tried first because any purely-linear recurrence (including
    # degenerate single-slot ``value <- k·value``) is cleaner as a
    # :class:`ClosedForm` than as a :class:`ProductForm` of a constant
    # factor. For each dependent slot, the transition must be affine in
    # the dependent slots only (no counter dependence, no higher-degree
    # terms). Build A (|dep| × |dep|) and b (|dep|).
    dep_slot_vars = [slot_vars[i] for i in dep_indices]
    A_rows: List[List[int]] = []
    b_vec: List[int] = []
    linear_ok = True
    for i in dep_indices:
        lin = _extract_linear(substituted[i], dep_slot_vars)
        if lin is None:
            linear_ok = False
            break
        coeffs, const = lin
        row = [coeffs[v] for v in dep_slot_vars]
        A_rows.append(row)
        b_vec.append(const)
    if linear_ok:
        # Sanity-check the counter variable never appears in
        # dependent-slot transitions — if it does, this isn't a
        # constant-coefficient linear recurrence and Tier 2 bails.
        for i in dep_indices:
            for mono in substituted[i].terms:
                for v, p in mono:
                    if v == counter_slot_var:
                        linear_ok = False
                        break
                if not linear_ok:
                    break
            if not linear_ok:
                break
    if linear_ok and dep_indices:
        s0_deps: Tuple[Poly, ...] = tuple(
            initial_slots[i] for i in dep_indices  # type: ignore[misc]
        )
        # Projection: pick the last dependent slot by default — for
        # fibonacci this is ``b`` (= fib(n)); for power_of_2 it's
        # ``value``; for any single-dependent-slot loop it's the sole
        # dependent slot.
        projection = len(dep_indices) - 1
        cf = ClosedForm(
            A=tuple(tuple(row) for row in A_rows),
            b=tuple(b_vec),
            s_0=s0_deps,
            trip_count=trip_count,
            projection=projection,
        )
        return "tier2", cf

    # ── Tier 3 — multiplicative single-accumulator pattern ──
    # Exactly one dependent slot whose transition is
    # ``slot_acc * p(counter_slot_var)`` for a Poly ``p`` that
    # *genuinely* depends on the counter (else Tier 2 would already
    # have caught it). Factorial's body is the canonical example:
    # ``acc <- acc · counter``.
    if len(dep_indices) == 1:
        acc_idx = dep_indices[0]
        acc_var = slot_vars[acc_idx]
        acc_trans = substituted[acc_idx]
        factor_terms: Dict[Monomial, Union[int, Fraction]] = {}
        ok = True
        factor_has_counter = False
        for mono, c in acc_trans.terms.items():
            acc_power = 0
            remainder: List[Tuple[int, int]] = []
            for v, p in mono:
                if v == acc_var:
                    acc_power = p
                elif v == counter_slot_var:
                    remainder.append((v, p))
                    factor_has_counter = True
                else:
                    ok = False
                    break
            if not ok:
                break
            if acc_power != 1:
                ok = False
                break
            factor_terms[tuple(remainder)] = c
        if ok and factor_terms and factor_has_counter:
            factor = Poly(factor_terms)
            acc_init = initial_slots[acc_idx]
            if isinstance(acc_init, Poly):
                init_int = _as_concrete_int(acc_init)
                if init_int is None and prefix_bindings is not None:
                    try:
                        v = acc_init.eval_at(prefix_bindings)
                        if isinstance(v, Fraction) and v.denominator != 1:
                            init_int = None
                        else:
                            init_int = int(v)
                    except KeyError:
                        init_int = None
                if init_int is not None:
                    # Commutative product: iterating counter=upper..1
                    # matches 1..upper. ``eval_at`` walks lower..upper.
                    pf = ProductForm(
                        p=factor,
                        counter_var=counter_slot_var,
                        lower=Poly.constant(1),
                        upper=trip_count,
                        init=int(init_int),
                    )
                    return "tier3", pf

    # ── Tier 1 — affine in dep slots + polynomial driver in counter ──
    # Single dependent slot whose transition is
    # ``dep + p(counter_slot_var)`` where ``p`` is any Poly purely in
    # the counter slot var. Closed form:
    #   dep_N = dep_0 + Σ_{k=0}^{N-1} p(counter_0 - k)
    # For the catalog's ``sum_1_to_n_sym`` this is
    #   acc_N = 0 + Σ_{k=0}^{N-1} (n - k) = n(n+1)/2.
    if len(dep_indices) == 1:
        acc_idx = dep_indices[0]
        acc_var = slot_vars[acc_idx]
        acc_trans = substituted[acc_idx]
        # Split into ``acc_var`` contribution and a pure-counter Poly.
        acc_coef_present = False
        driver_terms: Dict[Monomial, Union[int, Fraction]] = {}
        ok = True
        for mono, c in acc_trans.terms.items():
            acc_power = 0
            counter_only: List[Tuple[int, int]] = []
            other = False
            for v, p in mono:
                if v == acc_var:
                    acc_power = p
                elif v == counter_slot_var:
                    counter_only.append((v, p))
                else:
                    other = True
                    break
            if other:
                ok = False
                break
            if acc_power > 1:
                ok = False
                break
            if acc_power == 1 and counter_only:
                # acc_var * counter^k — Tier 3 territory, not Tier 1.
                ok = False
                break
            if acc_power == 1:
                if c != 1:
                    ok = False
                    break
                acc_coef_present = True
                continue
            driver_terms[tuple(counter_only)] = c
        if ok and acc_coef_present:
            driver = Poly(driver_terms)
            # Symbolic summation: let k ∈ {0, .., N-1} and counter_k =
            # counter_0 - k. Σ_{k=0}^{N-1} p(counter_0 - k). Change
            # variable j = counter_0 - k → as k runs 0..N-1, j runs
            # counter_0..counter_0-N+1 = counter_0..1 (decreasing).
            # For N = counter_0 this is Σ_{j=1}^{counter_0} p(j).
            # Build that sum symbolically by unrolling under Faulhaber
            # — but trip_count is itself symbolic, so we need closed
            # Faulhaber polynomials for each power.
            # Driver is Poly over a single variable counter_slot_var.
            acc_init = initial_slots[acc_idx]
            if not isinstance(acc_init, Poly):
                return None
            closed = _faulhaber_sum(driver, counter_slot_var, trip_count)
            if closed is None:
                return None
            return "tier1", acc_init + closed

    return None


# Faulhaber polynomials up to degree 3. Each entry is a list of Fraction
# coefficients [c_0, c_1, ..., c_{d+1}] so that
#   S_d(n) = Σ_{k=1}^{n} k^d = c_0 + c_1·n + c_2·n² + ... + c_{d+1}·n^{d+1}.
# Only the degrees actually exercised by the catalog are encoded; higher
# degrees raise and bail out of inference.
_FAULHABER: Dict[int, List[Fraction]] = {
    0: [Fraction(0), Fraction(1)],                                 # n
    1: [Fraction(0), Fraction(1, 2), Fraction(1, 2)],              # n(n+1)/2
    2: [Fraction(0), Fraction(1, 6), Fraction(1, 2), Fraction(1, 3)],
    3: [Fraction(0), Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(1, 4)],
}


def _sum_k_power(degree: int, upper: Poly) -> Optional[Poly]:
    """Return ``Σ_{k=1}^{upper} k^degree`` as a Poly in ``upper``.

    ``upper`` is a Poly (typically a single variable ``x_n``). Uses the
    Faulhaber coefficients above; returns ``None`` for degrees beyond
    the encoded table.
    """
    if degree < 0:
        return None
    coefs = _FAULHABER.get(degree)
    if coefs is None:
        return None
    # Build Σ = Σ_j coefs[j] * upper^j.
    total = Poly.constant(0)
    power = Poly.constant(1)
    for j, c in enumerate(coefs):
        if c != 0:
            total = total + Poly({(): c}) * power
        if j < len(coefs) - 1:
            power = power * upper
    return total


def _faulhaber_sum(driver: Poly, counter_var: int,
                   trip_count: Poly) -> Optional[Poly]:
    """Evaluate ``Σ_{j=1}^{trip_count} driver(j)`` symbolically.

    ``driver`` is a Poly in ``counter_var`` only. The identity
    ``Σ_{j=1}^{N} j^d`` is realised by :func:`_sum_k_power` for each
    monomial; the sum is linear in the driver's coefficients.
    """
    total = Poly.constant(0)
    for mono, coeff in driver.terms.items():
        if not mono:
            # Constant term c contributes c·trip_count.
            total = total + Poly({(): coeff}) * trip_count
            continue
        if len(mono) != 1:
            return None
        v, p = mono[0]
        if v != counter_var:
            return None
        s_poly = _sum_k_power(p, trip_count)
        if s_poly is None:
            return None
        total = total + Poly({(): coeff}) * s_poly
    return total


def _try_solve_recurrence(
    prog: List[isa.Instruction],
    stuck_path: "_Path",
    input_mode: str,
    ops: ArithmeticOps,
) -> Optional["RationalStackValue"]:
    """Top-level driver: try to emit a closed form for the loop that
    halted ``stuck_path`` with ``loop_symbolic``.

    Returns the projected closed-form top (:class:`Poly` /
    :class:`ClosedForm` / :class:`ProductForm`) or ``None`` if any
    step of the inference bails. Structured as a best-effort pass:
    any failure falls cleanly through to the existing
    ``loop_symbolic`` behaviour.
    """
    back_edge_pc = stuck_path.pc
    if back_edge_pc < 0 or back_edge_pc >= len(prog):
        return None
    back_edge = prog[back_edge_pc]
    if back_edge.op not in (isa.OP_JZ, isa.OP_JNZ):
        return None
    loop_header_pc = int(back_edge.arg)
    if loop_header_pc < 0 or loop_header_pc > back_edge_pc:
        return None

    # 1. Prefix → initial loop state.
    prefix = _run_prefix(prog, loop_header_pc, input_mode, ops)
    if prefix is None:
        return None
    initial_stack, prefix_bindings = prefix
    n_slots = len(initial_stack)
    if n_slots == 0:
        return None

    # 2. Body → transition on slot vars.
    body_out = _run_body_iteration(
        prog, loop_header_pc, back_edge_pc, n_slots, input_mode, ops,
    )
    if body_out is None:
        return None
    transition, body_bindings = body_out

    # 3. Classify.
    classified = _classify_recurrence(
        tuple(initial_stack), transition, body_bindings,
        prefix_bindings=prefix_bindings,
    )
    if classified is None:
        return None
    _tier, closed_top = classified
    return closed_top


# ─── Reporting helper ─────────────────────────────────────────────

def collapse_report(prog: List[isa.Instruction], *,
                    name: str = "") -> str:
    """Run ``prog`` symbolically and return a one-line collapse summary.

    Example::

        PUSH 5; DUP;ADD;DUP;ADD;DUP;ADD;DUP;ADD  →  9 heads, 1 monomial, top = 16·x0
    """
    r = run_symbolic(prog)
    prefix = f"{name}: " if name else ""
    return (f"{prefix}{r.n_heads} heads → {r.top.n_monomials()} "
            f"monomials, top = {r.top}")


def guarded_to_mermaid(gp: "GuardedPoly") -> str:
    """Render a ``GuardedPoly`` case table as a Mermaid flowchart decision tree.

    Returns valid Mermaid ``flowchart TD`` source. The last case is rendered
    as an implicit ``else`` leaf — its guards are implied by the preceding
    decisions — so a 2-case / single-guard GuardedPoly produces exactly one
    decision diamond and two value leaves.

    Multi-guard cases chain their guards with ``True`` edges; the ``False``
    edge of each case's last guard connects to the next case.
    """
    def _label(g: "Guard") -> str:
        return f"{g.poly} {_REL_SYMBOL[g.relation]} 0"

    def _mq(s: str) -> str:
        return '"' + s.replace('"', "'") + '"'

    lines = ["flowchart TD"]
    ctr = [0]

    def _fresh(prefix: str) -> str:
        ctr[0] += 1
        return f"{prefix}{ctr[0]}"

    cases = list(gp.cases)
    n = len(cases)
    pending: Optional[Tuple[str, str]] = None  # (from_node_id, edge_label)

    for i, (guards, value) in enumerate(cases):
        guards_list = list(guards)

        if i == n - 1:
            # Last case: render as implied else leaf (guards follow by elimination).
            leaf = _fresh("L")
            lines.append(f"    {leaf}[{_mq(repr(value))}]")
            if pending:
                src, lbl = pending
                lines.append(f"    {src} -->|{lbl}| {leaf}")
            break

        if not guards_list:
            # Unconditional middle case (degenerate; shouldn't appear in a valid partition).
            leaf = _fresh("L")
            lines.append(f"    {leaf}[{_mq(repr(value))}]")
            if pending:
                src, lbl = pending
                lines.append(f"    {src} -->|{lbl}| {leaf}")
            pending = None
            continue

        # Chain each guard in the conjunction with True edges between them.
        prev_dec: Optional[str] = None
        for j, g in enumerate(guards_list):
            dec = _fresh("D")
            lines.append(f"    {dec}{{{_mq(_label(g))}}}")
            if j == 0 and pending:
                src, lbl = pending
                lines.append(f"    {src} -->|{lbl}| {dec}")
            elif j > 0 and prev_dec is not None:
                lines.append(f"    {prev_dec} -->|True| {dec}")
            prev_dec = dec

        # True branch of the last guard leads to this case's value leaf.
        leaf = _fresh("L")
        lines.append(f"    {leaf}[{_mq(repr(value))}]")
        lines.append(f"    {prev_dec} -->|True| {leaf}")

        # False branch of the last guard connects to the next case.
        pending = (prev_dec, "False")

    return "\n".join(lines)



__all__ = [
    "run_forking", "ForkingResult",
    "DEFAULT_MAX_PATHS", "DEFAULT_MAX_STEPS",
    "collapse_report", "guarded_to_mermaid",
]
