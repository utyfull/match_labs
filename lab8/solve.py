import math


def f(x):
    return math.cos(x) + x


def prod(values):
    result = 1.0
    for value in values:
        result *= value
    return result


def lagrange_denominators(xs):
    denoms = []
    for j, xj in enumerate(xs):
        denoms.append(prod(xj - xi for i, xi in enumerate(xs) if i != j))
    return denoms


def lagrange_value(xs, ys, u):
    value = 0.0
    for j, yj in enumerate(ys):
        term = yj
        for i, xi in enumerate(xs):
            if i != j:
                term *= (u - xi) / (xs[j] - xi)
        value += term
    return value


def poly_add(a, b):
    n = max(len(a), len(b))
    result = [0.0] * n
    for i in range(n):
        if i < len(a):
            result[i] += a[i]
        if i < len(b):
            result[i] += b[i]
    return result


def poly_scale(a, scalar):
    return [scalar * value for value in a]


def poly_mul(a, b):
    result = [0.0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] += ai * bj
    return result


def lagrange_polynomial_coefficients(xs, ys):
    result = [0.0]
    for j, yj in enumerate(ys):
        basis = [1.0]
        denom = 1.0
        for i, xi in enumerate(xs):
            if i != j:
                basis = poly_mul(basis, [-xi, 1.0])
                denom *= xs[j] - xi
        result = poly_add(result, poly_scale(basis, yj / denom))
    return result


def divided_differences(xs, ys):
    n = len(xs)
    table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = ys[i]

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i][j - 1] - table[i + 1][j - 1]) / (xs[i] - xs[i + j])

    return table


def newton_value(xs, table, u):
    value = 0.0
    multiplier = 1.0
    for j in range(len(xs)):
        value += table[0][j] * multiplier
        multiplier *= u - xs[j]
    return value


def newton_polynomial_coefficients(xs, table):
    result = [0.0]
    term = [1.0]

    for j in range(len(xs)):
        result = poly_add(result, poly_scale(term, table[0][j]))
        term = poly_mul(term, [-xs[j], 1.0])

    return result


def clean(value, eps=1e-12):
    return 0.0 if abs(value) < eps else value


def fmt(value, digits=10):
    return f"{clean(value):.{digits}f}"


def factor_to_string(xi):
    xi = clean(xi)
    if xi == 0.0:
        return "x"
    if xi > 0.0:
        return f"(x - {fmt(xi, 6)})"
    return f"(x + {fmt(abs(xi), 6)})"


def signed_term(coef, factors):
    coef = clean(coef)
    sign = "-" if coef < 0.0 else "+"
    abs_coef = abs(coef)
    if factors:
        body = f"{fmt(abs_coef, 8)}*" + "*".join(factors)
    else:
        body = fmt(abs_coef, 8)
    return sign, body


def join_signed_terms(terms):
    if not terms:
        return "0"

    parts = []
    for idx, (sign, body) in enumerate(terms):
        if idx == 0:
            parts.append(body if sign == "+" else f"-{body}")
        else:
            parts.append(f" {sign} {body}")
    return "".join(parts)


def polynomial_to_string(coeffs):
    terms = []
    for power, coef in reversed(list(enumerate(coeffs))):
        coef = clean(coef)
        if coef == 0.0:
            continue

        if power == 0:
            body = fmt(abs(coef), 8)
        elif power == 1:
            body = f"{fmt(abs(coef), 8)}*x"
        else:
            body = f"{fmt(abs(coef), 8)}*x^{power}"
        terms.append(("-" if coef < 0.0 else "+", body))

    return join_signed_terms(terms)


def lagrange_product_form(xs, ys):
    denoms = lagrange_denominators(xs)
    terms = []
    for j, yj in enumerate(ys):
        coef = yj / denoms[j]
        factors = [factor_to_string(xi) for i, xi in enumerate(xs) if i != j]
        terms.append(signed_term(coef, factors))
    return "L3(x) = " + join_signed_terms(terms)


def newton_product_form(xs, table):
    terms = []
    for j in range(len(xs)):
        factors = [factor_to_string(xs[i]) for i in range(j)]
        terms.append(signed_term(table[0][j], factors))
    return "N3(x) = " + join_signed_terms(terms)


def print_table(headers, rows):
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    header_line = "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    sep_line = "  ".join("-" * widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def print_case(title, node_names, xs, x_star):
    ys = [f(x) for x in xs]
    exact = f(x_star)
    lagrange_at_star = lagrange_value(xs, ys, x_star)
    diff_table = divided_differences(xs, ys)
    newton_at_star = newton_value(xs, diff_table, x_star)

    print("=" * 78)
    print(title)
    print("Function: y = cos(x) + x")
    print(f"x* = {fmt(x_star, 6)}")
    print()

    rows = []
    for j, (name, xj, yj) in enumerate(zip(node_names, xs, ys)):
        rows.append([str(j), name, fmt(xj, 10), fmt(yj, 10)])
    print_table(["j", "x_j", "x_j numeric", "y_j=f(x_j)"], rows)
    print()

    denoms = lagrange_denominators(xs)
    rows = []
    for j in range(len(xs)):
        rows.append([str(j), fmt(denoms[j], 10), fmt(ys[j] / denoms[j], 10)])
    print("Lagrange coefficients: c_j = y_j / product(x_j - x_i), i != j")
    print_table(["j", "product(x_j-x_i)", "c_j"], rows)
    print()

    lagrange_coeffs = lagrange_polynomial_coefficients(xs, ys)
    print("Lagrange polynomial in product form:")
    print(lagrange_product_form(xs, ys))
    print("Lagrange polynomial in powers of x:")
    print("L3(x) = " + polynomial_to_string(lagrange_coeffs))
    print()

    rows = []
    n = len(xs)
    for i in range(n):
        row = [str(i), node_names[i], fmt(diff_table[i][0], 10)]
        for j in range(1, n):
            row.append(fmt(diff_table[i][j], 10) if i < n - j else "")
        rows.append(row)
    print("Newton divided differences:")
    print_table(["i", "x_i", "Delta^0", "Delta^1", "Delta^2", "Delta^3"], rows)
    print()

    newton_coeffs = newton_polynomial_coefficients(xs, diff_table)
    print("Newton polynomial in product form:")
    print(newton_product_form(xs, diff_table))
    print("Newton polynomial in powers of x:")
    print("N3(x) = " + polynomial_to_string(newton_coeffs))
    print()

    print("Value and interpolation error at x*:")
    print(f"f(x*)      = {fmt(exact, 12)}")
    print(f"L3(x*)     = {fmt(lagrange_at_star, 12)}")
    print(f"N3(x*)     = {fmt(newton_at_star, 12)}")
    print(f"|L3-f|     = {abs(lagrange_at_star - exact):.12e}")
    print(f"|N3-f|     = {abs(newton_at_star - exact):.12e}")
    print(f"|L3-N3|    = {abs(lagrange_at_star - newton_at_star):.12e}")
    print()

    return abs(lagrange_at_star - exact)


def main():
    x_star = 1.0

    case_a_names = ["0", "pi/6", "2*pi/6", "3*pi/6"]
    case_a_xs = [0.0, math.pi / 6.0, 2.0 * math.pi / 6.0, 3.0 * math.pi / 6.0]

    case_b_names = ["0", "pi/6", "pi/4", "pi/2"]
    case_b_xs = [0.0, math.pi / 6.0, math.pi / 4.0, math.pi / 2.0]

    err_a = print_case("Variant 13a", case_a_names, case_a_xs, x_star)
    err_b = print_case("Variant 13b", case_b_names, case_b_xs, x_star)

    print("=" * 78)
    print("Comparison:")
    print(f"error for 13a = {err_a:.12e}")
    print(f"error for 13b = {err_b:.12e}")
    if err_a < err_b:
        print("For x*=1.0, variant 13a gives the smaller interpolation error.")
    elif err_b < err_a:
        print("For x*=1.0, variant 13b gives the smaller interpolation error.")
    else:
        print("Both variants give the same interpolation error at x*=1.0.")


if __name__ == "__main__":
    main()


# The implemented formulas follow sections 4.3 and 4.5:
#
# Lagrange:
# L3(x) = sum_j y_j * product_{i!=j} (x - x_i) / (x_j - x_i).
#
# Newton:
# N3(x) = Delta_0^0
#       + Delta_01^1 * (x - x_0)
#       + Delta_012^2 * (x - x_0) * (x - x_1)
#       + Delta_0123^3 * (x - x_0) * (x - x_1) * (x - x_2).
#
# The interpolation error printed by the program is absolute:
# |P3(x*) - f(x*)|.
