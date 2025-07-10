import sympy as smp
from sympy.printing.julia import julia_code


def sympy_matrix_to_julia_fill_hermitian(
    M: smp.Matrix,
    func_name: str = "fill_matrix!",
    zero_input: bool = True,
    inplace_add: bool = False,
    input_name: str = "H",
) -> tuple[str, smp.FunctionClass]:
    """
    Generate a Julia function that efficiently fills a Hermitian matrix in-place from a SymPy matrix.

    The generated Julia function optimizes performance through several strategies:
    1. Common subexpression elimination (CSE) applied only to the upper triangle
    2. Pruning of temporary variables not used in the upper triangle
    3. Inlining of temporary variables used only once
    4. Bulk zero-initialization using SIMD loops with type hoisting
    5. Computing only the upper triangle and mirroring via conjugation
    6. Emitting only non-zero assignments

    Args:
        M (smp.Matrix): A SymPy matrix (assumed Hermitian) with symbolic expressions
        func_name (str): Name for the generated Julia function (convention: use bang suffix)

    Returns:
        tuple[str, smp.FunctionClass]:
            - Julia code string for the optimized matrix-filling function
            - SymPy function object representing the call signature
    """
    rows, cols = M.rows, M.cols
    # 1) sort free symbols for deterministic argument order
    syms = sorted(M.free_symbols, key=lambda s: s.name)
    args = ", ".join(s.name for s in syms)

    # 2) run CSE on upper-triangle entries only
    upper = [M[i, j] for i in range(rows) for j in range(i, cols)]
    subs_all, reduced_upper = smp.cse(upper, symbols=smp.numbered_symbols("t"))
    # reconstruct full reduced list with Hermitian mirror
    reduced_all = [None] * (rows * cols)
    idx = 0
    for i in range(rows):
        for j in range(i, cols):
            expr = reduced_upper[idx]
            reduced_all[i * cols + j] = expr
            reduced_all[j * cols + i] = expr.conjugate() if i != j else expr
            idx += 1

    # 3) determine which temps are needed for i ≤ j
    tmp_syms = [t for t, _ in subs_all]
    used_exprs = [
        reduced_all[i * cols + j] for i in range(rows) for j in range(i, cols)
    ]
    needed = set()
    lookup = {t: expr for t, expr in subs_all}

    def mark(expr):
        for s in expr.free_symbols:
            if s in tmp_syms and s not in needed:
                needed.add(s)
                mark(lookup[s])

    for expr in used_exprs:
        mark(expr)
    pruned_subs = [(t, e) for t, e in subs_all if t in needed]

    # 4) count usage for inlining
    usage = {t: 0 for t, _ in pruned_subs}
    all_exprs = [e for _, e in pruned_subs] + used_exprs
    for expr in all_exprs:
        for t in list(usage):
            if t in expr.free_symbols:
                usage[t] += 1
    inline = {t for t, cnt in usage.items() if cnt == 1}
    keep = [(t, e) for t, e in pruned_subs if t not in inline]
    inline_map = {t: lookup[t] for t in inline}

    if inplace_add:
        assignment = "+="
    else:
        assignment = "="

    # 5) emit Julia code
    lines = []
    sig = f"function {func_name}({input_name}" + (", " + args if args else "") + ")"
    lines.append(sig)
    lines.append("    @inbounds begin")
    if zero_input:
        lines.append(f"        zero_matrix!({input_name})")
    # emit kept temporaries
    for t, expr in keep:
        expr_sub = expr.subs(inline_map)
        lines.append(f"        {t} = {julia_code(expr_sub)}")
    # fill non-zero upper triangle and mirror
    for i in range(rows):
        for j in range(i, cols):
            expr = reduced_all[i * cols + j].subs(inline_map)
            if expr != 0:
                code = julia_code(expr)
                lines.append(
                    f"        {input_name}[{i + 1},{j + 1}] {assignment} {code}"
                )
                if i != j:
                    lines.append(
                        f"        {input_name}[{j + 1},{i + 1}] {assignment} conj({input_name}[{i + 1},{j + 1}])"
                    )
    lines.append("    end")
    lines.append("    nothing")
    lines.append("end")
    return "\n".join(lines), smp.Function(func_name)(*syms)
