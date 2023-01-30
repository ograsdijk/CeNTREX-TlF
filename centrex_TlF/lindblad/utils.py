import sympy as smp

__all__ = ["generate_density_matrix_symbolic"]


def recursive_subscript(i: int) -> str:
    # chr(0x2080+i) is unicode for
    # subscript num(i), resulting in x₀₀ for example
    if i < 10:
        return chr(0x2080 + i)
    else:
        return recursive_subscript(i // 10) + chr(0x2080 + i % 10)


def generate_density_matrix_symbolic(
    levels: int,
) -> smp.matrices.dense.MutableDenseMatrix:
    ρ = smp.zeros(levels, levels)
    levels = levels
    for i in range(levels):
        for j in range(i, levels):
            # \u03C1 is unicode for ρ,
            if i == j:
                ρ[i, j] = smp.Symbol(
                    "\u03C1{0},{1}".format(
                        recursive_subscript(i), recursive_subscript(j)
                    )
                )
            else:
                ρ[i, j] = smp.Symbol(
                    "\u03C1{0},{1}".format(
                        recursive_subscript(i), recursive_subscript(j)
                    )
                )
                ρ[j, i] = smp.Symbol(
                    "\u03C1{1},{0}".format(
                        recursive_subscript(i), recursive_subscript(j)
                    )
                )
    return ρ
