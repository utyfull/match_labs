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
        raise ValueError("At least three nodes are required for a cubic spline.")
    for i in range(len(xs) - 1):
        if xs[i + 1] <= xs[i]:
            raise ValueError("Interpolation nodes must be strictly increasing.")


def build_c_system(xs, ys):
    # Natural cubic spline:
    # S_i(x) = a_i + b_i*(x-x_{i-1}) + c_i*(x-x_{i-1})^2 + d_i*(x-x_{i-1})^3.
    #
    # The boundary conditions are c_1 = 0 and c_{n+1} = 0, because
    # S''(x_0) = 2*c_1 = 0 and S''(x_n) = 2*c_{n+1} = 0.
    # The internal coefficients c_2, ..., c_n are found from a tridiagonal system.
    n = len(xs) - 1
    h = [xs[i + 1] - xs[i] for i in range(n)]

    lower = []
    diag = []
    upper = []
    rhs = []

    for k in range(1, n):
        lower.append(h[k - 1] if k > 1 else 0.0)
        diag.append(2.0 * (h[k - 1] + h[k]))
        upper.append(h[k] if k < n - 1 else 0.0)
        rhs.append(
            3.0
            * (
                (ys[k + 1] - ys[k]) / h[k]
                - (ys[k] - ys[k - 1]) / h[k - 1]
            )
        )

    return h, lower, diag, upper, rhs


def sweep_method(lower, diag, upper, rhs):
    if not diag:
        return [], [], []

    n = len(diag)
    p = [0.0] * n
    q = [0.0] * n

    p[0] = -upper[0] / diag[0]
    q[0] = rhs[0] / diag[0]

    for i in range(1, n):
        denominator = diag[i] + lower[i] * p[i - 1]
        if abs(denominator) < 1e-15:
            raise ZeroDivisionError("Zero denominator in the sweep method.")
        p[i] = -upper[i] / denominator
        q[i] = (rhs[i] - lower[i] * q[i - 1]) / denominator

    result = [0.0] * n
    result[-1] = q[-1]
    for i in range(n - 2, -1, -1):
        result[i] = p[i] * result[i + 1] + q[i]

    return result, p, q


def calculate_a_coefficients(ys):
    # a_i = y_{i-1}
    return [ys[i] for i in range(len(ys) - 1)]


def calculate_c_coefficients(xs, ys):
    h, lower, diag, upper, rhs = build_c_system(xs, ys)
    internal_c, p, q = sweep_method(lower, diag, upper, rhs)

    # Natural spline boundary conditions:
    # c_1 = 0, c_{n+1} = 0.
    # Internal coefficients c_2, ..., c_n come from the tridiagonal system.
    c = [0.0] * len(xs)
    for i, value in enumerate(internal_c, start=1):
        c[i] = value

    return {
        "h": h,
        "lower": lower,
        "diag": diag,
        "upper": upper,
        "rhs": rhs,
        "p": p,
        "q": q,
        "c": c,
    }


def calculate_b_coefficients(xs, ys, h, c):
    b = []
    for i in range(len(xs) - 1):
        # b_i = (y_i - y_{i-1}) / h_i - h_i * (c_{i+1} + 2*c_i) / 3
        b_i = (ys[i + 1] - ys[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0
        b.append(b_i)
    return b


def calculate_d_coefficients(xs, h, c):
    d = []
    for i in range(len(xs) - 1):
        # d_i = (c_{i+1} - c_i) / (3*h_i)
        d_i = (c[i + 1] - c[i]) / (3.0 * h[i])
        d.append(d_i)
    return d


def natural_cubic_spline_coefficients(xs, ys):
    validate_nodes(xs, ys)

    a = calculate_a_coefficients(ys)
    c_data = calculate_c_coefficients(xs, ys)
    h = c_data["h"]
    c = c_data["c"]
    b = calculate_b_coefficients(xs, ys, h, c)
    d = calculate_d_coefficients(xs, h, c)

    return {
        "h": h,
        "lower": c_data["lower"],
        "diag": c_data["diag"],
        "upper": c_data["upper"],
        "rhs": c_data["rhs"],
        "p": c_data["p"],
        "q": c_data["q"],
        "a": a,
        "b": b,
        "c": c[:-1],
        "d": d,
        "right_boundary_c": c[-1],
    }


def spline_interval_index(xs, x_star):
    if not xs[0] <= x_star <= xs[-1]:
        raise ValueError("x* must belong to the interpolation interval.")

    for i in range(len(xs) - 1):
        if xs[i] <= x_star <= xs[i + 1]:
            return i

    return len(xs) - 2


def spline_value(xs, coeffs, x_star):
    i = spline_interval_index(xs, x_star)
    u = x_star - xs[i]
    value = coeffs["a"][i] + coeffs["b"][i] * u + coeffs["c"][i] * u * u + coeffs["d"][i] * u * u * u
    return value, i, u


def spline_piece_values(xs, coeffs, i, u):
    a = coeffs["a"][i]
    b = coeffs["b"][i]
    c = coeffs["c"][i]
    d = coeffs["d"][i]
    s = a + b * u + c * u * u + d * u * u * u
    s1 = b + 2.0 * c * u + 3.0 * d * u * u
    s2 = 2.0 * c + 6.0 * d * u
    return s, s1, s2


def verify_spline(xs, ys, coeffs):
    n = len(xs) - 1
    node_errors = []
    for i in range(n):
        s_left, _, _ = spline_piece_values(xs, coeffs, i, 0.0)
        node_errors.append(abs(s_left - ys[i]))
    s_last, _, _ = spline_piece_values(xs, coeffs, n - 1, xs[n] - xs[n - 1])
    node_errors.append(abs(s_last - ys[n]))

    _, _, s2_left_boundary = spline_piece_values(xs, coeffs, 0, 0.0)
    _, _, s2_right_boundary = spline_piece_values(xs, coeffs, n - 1, xs[n] - xs[n - 1])

    continuity = []
    for i in range(n - 1):
        h_i = xs[i + 1] - xs[i]
        s_l, s1_l, s2_l = spline_piece_values(xs, coeffs, i, h_i)
        s_r, s1_r, s2_r = spline_piece_values(xs, coeffs, i + 1, 0.0)
        continuity.append((i + 1, abs(s_l - s_r), abs(s1_l - s1_r), abs(s2_l - s2_r)))

    return {
        "node_errors": node_errors,
        "s2_left_boundary": s2_left_boundary,
        "s2_right_boundary": s2_right_boundary,
        "continuity": continuity,
    }


def print_verification(xs, ys, checks):
    print("Verification of the constructed spline:")
    print()

    rows = []
    for i, err in enumerate(checks["node_errors"]):
        rows.append([str(i), fmt(xs[i], 6), fmt(ys[i], 10), f"{err:.3e}"])
    print("1) Spline passes through table nodes: |S(x_i) - y_i|")
    print_table(["i", "x_i", "y_i", "|S(x_i) - y_i|"], rows)
    print()

    print("2) Natural boundary conditions: S''(x_0) = 0 and S''(x_n) = 0")
    rows = [
        ["S''(x_0)", f"{checks['s2_left_boundary']:.3e}"],
        ["S''(x_n)", f"{checks['s2_right_boundary']:.3e}"],
    ]
    print_table(["value", "absolute value"], rows)
    print()

    if checks["continuity"]:
        print("3) Continuity of S, S', S'' at internal nodes")
        rows = []
        for i, ds, ds1, ds2 in checks["continuity"]:
            rows.append([str(i), fmt(xs[i], 6), f"{ds:.3e}", f"{ds1:.3e}", f"{ds2:.3e}"])
        print_table(["i", "x_i", "|S_l-S_r|", "|S'_l-S'_r|", "|S''_l-S''_r|"], rows)
        print()


def print_system(coeffs):
    rows = []
    m = len(coeffs["diag"])
    for row in range(m):
        equation_parts = []
        for col in range(m):
            value = 0.0
            if col == row - 1:
                value = coeffs["lower"][row]
            elif col == row:
                value = coeffs["diag"][row]
            elif col == row + 1:
                value = coeffs["upper"][row]

            if value != 0.0:
                equation_parts.append(f"{fmt(value, 6)}*c_{col + 2}")

        rows.append([str(row + 1), " + ".join(equation_parts), fmt(coeffs["rhs"][row], 10)])

    print("Tridiagonal system for c_2, ..., c_n:")
    print_table(["row", "left side", "right side"], rows)
    print()


def print_sweep_coefficients(coeffs):
    rows = []
    for i, (p_value, q_value) in enumerate(zip(coeffs["p"], coeffs["q"]), start=1):
        rows.append([str(i), f"c_{i + 1}", fmt(p_value, 10), fmt(q_value, 10)])

    print("Forward path coefficients of the sweep method:")
    print_table(["i", "unknown", "p_i", "q_i"], rows)
    print()


def print_spline_coefficients(xs, coeffs):
    rows = []
    for i in range(len(xs) - 1):
        rows.append(
            [
                str(i + 1),
                f"[{fmt(xs[i], 4)}, {fmt(xs[i + 1], 4)}]",
                fmt(coeffs["a"][i], 10),
                fmt(coeffs["b"][i], 10),
                fmt(coeffs["c"][i], 10),
                fmt(coeffs["d"][i], 10),
            ]
        )

    print("Cubic spline coefficients:")
    print_table(["i", "[x_{i-1}, x_i]", "a_i", "b_i", "c_i", "d_i"], rows)
    print()


def print_spline_piece(xs, coeffs, interval_index):
    i = interval_index
    terms = [
        (coeffs["a"][i], ""),
        (coeffs["b"][i], f"*(x - {fmt(xs[i], 6)})"),
        (coeffs["c"][i], f"*(x - {fmt(xs[i], 6)})^2"),
        (coeffs["d"][i], f"*(x - {fmt(xs[i], 6)})^3"),
    ]
    pieces = []
    for j, (coef, tail) in enumerate(terms):
        sign = "-" if coef < 0.0 else "+"
        body = f"{fmt(abs(coef), 10)}{tail}"
        if j == 0:
            pieces.append(body if sign == "+" else f"-{body}")
        else:
            pieces.append(f" {sign} {body}")

    print(f"x* belongs to interval {i + 1}: [{fmt(xs[i], 6)}, {fmt(xs[i + 1], 6)}]")
    print("Spline on this interval:")
    print(f"S_{i + 1}(x) = {''.join(pieces)}")
    print()


def main():
    # Variant 13:
    # x* = 1.5
    # i:    0      1       2       3      4
    # x_i:  0.0    1.0     2.0     3.0    4.0
    # f_i:  1.0    1.5403  1.5839  2.01   3.3464
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    ys = [1.0, 1.5403, 1.5839, 2.01, 3.3464]
    x_star = 1.5

    coeffs = natural_cubic_spline_coefficients(xs, ys)
    value, interval_index, u = spline_value(xs, coeffs, x_star)

    print("=" * 78)
    print("Lab 9. Natural cubic spline interpolation")
    print("Boundary conditions: S''(x_0) = 0, S''(x_n) = 0")
    print(f"x* = {fmt(x_star, 6)}")
    print()

    input_rows = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        input_rows.append([str(i), fmt(x, 10), fmt(y, 10)])
    print("Input data:")
    print_table(["i", "x_i", "f_i"], input_rows)
    print()

    h_rows = [[str(i + 1), fmt(value, 10)] for i, value in enumerate(coeffs["h"])]
    print("Steps h_i = x_i - x_{i-1}:")
    print_table(["i", "h_i"], h_rows)
    print()

    print_system(coeffs)
    print_sweep_coefficients(coeffs)
    print_spline_coefficients(xs, coeffs)
    print_spline_piece(xs, coeffs, interval_index)

    checks = verify_spline(xs, ys, coeffs)
    print_verification(xs, ys, checks)

    print("Value at x*:")
    print(f"u = x* - x_{interval_index} = {fmt(u, 10)}")
    print(f"S({fmt(x_star, 6)}) = {fmt(value, 12)}")


if __name__ == "__main__":
    main()
