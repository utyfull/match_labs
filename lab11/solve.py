def clean(value, eps=1e-12):
    return 0.0 if abs(value) < eps else value


def fmt(value, digits=10):
    return f"{clean(value):.{digits}f}"


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


def validate_nodes(xs, ys):
    if len(xs) != len(ys):
        raise ValueError("The number of x values must be equal to the number of y values.")
    if len(xs) < 3:
        raise ValueError("At least three nodes are required for the second derivative.")
    for i in range(len(xs) - 1):
        if xs[i + 1] <= xs[i]:
            raise ValueError("Interpolation nodes must be strictly increasing.")


def get_uniform_step(xs, eps=1e-12):
    h = xs[1] - xs[0]
    for i in range(1, len(xs) - 1):
        if abs((xs[i + 1] - xs[i]) - h) > eps:
            raise ValueError("The finite-difference formulas below require equally spaced nodes.")
    return h


def find_node_index(xs, x_star, eps=1e-12):
    for i, x in enumerate(xs):
        if abs(x - x_star) <= eps:
            return i
    raise ValueError("x* must coincide with one of the table nodes for these formulas.")


def divided_differences(xs, ys):
    n = len(xs)
    table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = ys[i]

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (xs[i + j] - xs[i])

    return table


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


def newton_polynomial_coefficients(xs, diff_table):
    result = [0.0]
    term = [1.0]

    for j in range(len(xs)):
        result = poly_add(result, poly_scale(term, diff_table[0][j]))
        term = poly_mul(term, [-xs[j], 1.0])

    return result


def polynomial_derivative(coeffs):
    return [power * coeffs[power] for power in range(1, len(coeffs))]


def polynomial_value(coeffs, x):
    value = 0.0
    power = 1.0
    for coef in coeffs:
        value += coef * power
        power *= x
    return value


def factor_to_string(xi):
    xi = clean(xi)
    if xi == 0.0:
        return "x"
    if xi > 0.0:
        return f"(x - {fmt(xi, 4)})"
    return f"(x + {fmt(abs(xi), 4)})"


def join_signed_terms(terms):
    if not terms:
        return "0"

    parts = []
    for i, (sign, body) in enumerate(terms):
        if i == 0:
            parts.append(body if sign == "+" else f"-{body}")
        else:
            parts.append(f" {sign} {body}")
    return "".join(parts)


def polynomial_to_string(name, coeffs):
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

    return f"{name}(x) = {join_signed_terms(terms)}"


def newton_product_form(xs, diff_table):
    terms = []
    for j in range(len(xs)):
        coef = clean(diff_table[0][j])
        if coef == 0.0:
            continue
        factors = [factor_to_string(xs[i]) for i in range(j)]
        body = fmt(abs(coef), 8)
        if factors:
            body += "*" + "*".join(factors)
        terms.append(("-" if coef < 0.0 else "+", body))

    return "N4(x) = " + join_signed_terms(terms)


def local_difference_derivatives(xs, ys, x_star):
    h = get_uniform_step(xs)
    i = find_node_index(xs, x_star)
    if i == 0 or i == len(xs) - 1:
        raise ValueError("x* must be an internal node for left, right and central formulas.")

    right = (ys[i + 1] - ys[i]) / h
    left = (ys[i] - ys[i - 1]) / h
    central = (ys[i + 1] - ys[i - 1]) / (2.0 * h)
    second = (ys[i + 1] - 2.0 * ys[i] + ys[i - 1]) / (h * h)

    return {
        "h": h,
        "index": i,
        "right": right,
        "left": left,
        "central": central,
        "second": second,
    }


def print_input_data(xs, ys, x_star):
    rows = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        rows.append([str(i), fmt(x, 10), fmt(y, 10)])

    print("Input data:")
    print(f"x* = {fmt(x_star, 10)}")
    print_table(["i", "x_i", "y_i"], rows)
    print()


def print_divided_differences(xs, diff_table):
    n = len(xs)
    rows = []
    for i in range(n):
        row = [str(i), fmt(xs[i], 10)]
        for j in range(n):
            row.append(fmt(diff_table[i][j], 10) if i < n - j else "")
        rows.append(row)

    headers = ["i", "x_i"] + [f"Delta^{j}" for j in range(n)]
    print("Table of divided differences:")
    print_table(headers, rows)
    print()


def print_local_difference_result(local):
    i = local["index"]
    h = local["h"]

    print("Finite-difference formulas for equally spaced nodes:")
    print(f"h = x_i - x_(i-1) = {fmt(h, 10)}, x* = x_{i}")
    print()

    rows = [
        ["right y'(x_i) = (y_(i+1)-y_i)/h", fmt(local["right"], 12)],
        ["left y'(x_i) = (y_i-y_(i-1))/h", fmt(local["left"], 12)],
        ["central y'(x_i) = (y_(i+1)-y_(i-1))/(2h)", fmt(local["central"], 12)],
        ["second y''(x_i) = (y_(i+1)-2y_i+y_(i-1))/h^2", fmt(local["second"], 12)],
    ]
    print_table(["formula", "value"], rows)
    print()


def print_newton_result(xs, diff_table, coeffs, x_star):
    d1_coeffs = polynomial_derivative(coeffs)
    d2_coeffs = polynomial_derivative(d1_coeffs)

    first = polynomial_value(d1_coeffs, x_star)
    second = polynomial_value(d2_coeffs, x_star)

    print("Newton interpolation polynomial built on the whole table:")
    print(newton_product_form(xs, diff_table))
    print(polynomial_to_string("N4", coeffs))
    print()

    print("Derivatives of the Newton interpolation polynomial:")
    print(polynomial_to_string("N4'", d1_coeffs))
    print(polynomial_to_string("N4''", d2_coeffs))
    print()

    rows = [
        ["N4'(x*)", fmt(first, 12)],
        ["N4''(x*)", fmt(second, 12)],
    ]
    print("Result by Program #16 idea: differentiate the Newton polynomial.")
    print_table(["value", "result"], rows)
    print()

    return first, second, d1_coeffs, d2_coeffs


def print_verification(xs, ys, coeffs, local, n4_first, n4_second):
    print("Verification:")
    print()

    rows = []
    max_err = 0.0
    for i, (x, y) in enumerate(zip(xs, ys)):
        err = abs(polynomial_value(coeffs, x) - y)
        max_err = max(max_err, err)
        rows.append([str(i), fmt(x, 6), fmt(y, 10), f"{err:.3e}"])
    print("1) Newton polynomial passes through table nodes: |N4(x_i) - y_i|")
    print_table(["i", "x_i", "y_i", "|N4(x_i) - y_i|"], rows)
    print(f"max |N4(x_i) - y_i| = {max_err:.3e}")
    print()

    print("2) Cross-check between finite-difference and N4 values at x*:")
    rows = [
        ["y'(x*)  central        ",  fmt(local["central"], 12)],
        ["y'(x*)  N4'(x*)        ",  fmt(n4_first, 12)],
        ["|central - N4'(x*)|    ",  f"{abs(local['central'] - n4_first):.3e}"],
        ["y''(x*) second diff    ",  fmt(local["second"], 12)],
        ["y''(x*) N4''(x*)       ",  fmt(n4_second, 12)],
        ["|second - N4''(x*)|    ",  f"{abs(local['second'] - n4_second):.3e}"],
    ]
    print_table(["quantity", "value"], rows)
    print()


def main():
    # Variant data:
    # x* = 0.8
    # i:    0       1       2       3       4
    # x_i:  0.2     0.5     0.8     1.1     1.4
    # y_i:  12.906  5.5273  3.8777  3.2692  3.0319
    xs = [0.2, 0.5, 0.8, 1.1, 1.4]
    ys = [12.906, 5.5273, 3.8777, 3.2692, 3.0319]
    x_star = 0.8

    validate_nodes(xs, ys)

    print("=" * 78)
    print("Lab 11. Numerical differentiation")
    print("Task: find the first and second derivatives of a table-defined function.")
    print()

    print_input_data(xs, ys, x_star)

    local = local_difference_derivatives(xs, ys, x_star)
    print_local_difference_result(local)

    diff_table = divided_differences(xs, ys)
    print_divided_differences(xs, diff_table)

    coeffs = newton_polynomial_coefficients(xs, diff_table)
    first, second, _, _ = print_newton_result(xs, diff_table, coeffs, x_star)

    print_verification(xs, ys, coeffs, local, first, second)

    print("Final answer:")
    print(f"Using central differences: y'({fmt(x_star, 4)}) ~= {fmt(local['central'], 12)}")
    print(f"Using central differences: y''({fmt(x_star, 4)}) ~= {fmt(local['second'], 12)}")
    print(f"Using N4 on all nodes:      y'({fmt(x_star, 4)}) ~= {fmt(first, 12)}")
    print(f"Using N4 on all nodes:      y''({fmt(x_star, 4)}) ~= {fmt(second, 12)}")


if __name__ == "__main__":
    main()
