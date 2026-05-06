import math


def f(x):
    return (x * x) / (x * x * x - 27.0)


def antiderivative(x):
    return math.log(abs(x * x * x - 27.0)) / 3.0


def clean(value, eps=1e-14):
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


def make_nodes(a, b, h):
    n = int(round((b - a) / h))
    if abs(a + n * h - b) > 1e-9:
        raise ValueError(f"Step h={h} does not divide [a, b] = [{a}, {b}] evenly.")
    xs = [a + i * h for i in range(n + 1)]
    xs[-1] = b
    return xs, n


def left_rectangle_rule(xs, ys):
    total = 0.0
    for i in range(1, len(xs)):
        total += ys[i - 1] * (xs[i] - xs[i - 1])
    return total


def right_rectangle_rule(xs, ys):
    total = 0.0
    for i in range(1, len(xs)):
        total += ys[i] * (xs[i] - xs[i - 1])
    return total


def midpoint_rule(xs, func):
    total = 0.0
    for i in range(1, len(xs)):
        x_mid = 0.5 * (xs[i - 1] + xs[i])
        total += func(x_mid) * (xs[i] - xs[i - 1])
    return total


def trapezoid_rule(xs, ys):
    total = 0.0
    for i in range(1, len(xs)):
        total += 0.5 * (ys[i - 1] + ys[i]) * (xs[i] - xs[i - 1])
    return total


def simpson_rule(xs, ys, func):
    total = 0.0
    for i in range(1, len(xs)):
        h = xs[i] - xs[i - 1]
        x_mid = 0.5 * (xs[i - 1] + xs[i])
        y_mid = func(x_mid)
        total += (h / 6.0) * (ys[i - 1] + 4.0 * y_mid + ys[i])
    return total


def runge_romberg(I_h, I_mh, p, m=2):
    refined = I_h + (I_h - I_mh) / (m ** p - 1.0)
    error = abs(I_h - I_mh) / (m ** p - 1.0)
    return refined, error


def print_node_table(xs, ys, label):
    print(f"Nodes for step {label}:")
    rows = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        rows.append([str(i), fmt(x, 6), fmt(y, 10)])
    print_table(["i", "x_i", "y_i = f(x_i)"], rows)
    print()


def compute_for_step(a, b, h, label):
    xs, n = make_nodes(a, b, h)
    ys = [f(x) for x in xs]

    print(f"--- Step {label} = {fmt(h, 4)} (n = {n} intervals, n+1 = {n + 1} nodes) ---")
    print_node_table(xs, ys, label)

    I_left = left_rectangle_rule(xs, ys)
    I_right = right_rectangle_rule(xs, ys)
    I_mid = midpoint_rule(xs, f)
    I_trap = trapezoid_rule(xs, ys)
    I_simp = simpson_rule(xs, ys, f)

    rows = [
        ["Left rectangle  (p=1)", fmt(I_left, 10)],
        ["Right rectangle (p=1)", fmt(I_right, 10)],
        ["Midpoint        (p=2)", fmt(I_mid, 10)],
        ["Trapezoid       (p=2)", fmt(I_trap, 10)],
        ["Simpson         (p=4)", fmt(I_simp, 10)],
    ]
    print(f"Integral approximations with step {label}:")
    print_table(["method", "value"], rows)
    print()

    return {
        "h": h,
        "n": n,
        "left": I_left,
        "right": I_right,
        "mid": I_mid,
        "trap": I_trap,
        "simp": I_simp,
    }


def print_runge_romberg(res1, res2, exact):
    h1 = res1["h"]
    h2 = res2["h"]
    m = h1 / h2

    print("=" * 78)
    print("Runge-Romberg-Richardson refinement")
    print(f"h1 = {fmt(h1, 4)}, h2 = {fmt(h2, 4)}, partition ratio m = h1/h2 = {fmt(m, 4)}")
    print()
    print("Refined value:  I_refined = I_h2 + (I_h2 - I_h1) / (m^p - 1)")
    print("Error of I_h2:  |R| ~= |I_h2 - I_h1| / (m^p - 1)")
    print()

    methods = [
        ("Left rectangle ", "left",  1),
        ("Right rectangle", "right", 1),
        ("Midpoint       ", "mid",   2),
        ("Trapezoid      ", "trap",  2),
        ("Simpson        ", "simp",  4),
    ]

    rows = []
    for name, key, p in methods:
        I1 = res1[key]
        I2 = res2[key]
        refined, error = runge_romberg(I2, I1, p, m)
        actual_err_h1 = abs(I1 - exact)
        actual_err_h2 = abs(I2 - exact)
        actual_err_ref = abs(refined - exact)
        rows.append([
            name,
            str(p),
            fmt(I1, 8),
            fmt(I2, 8),
            fmt(refined, 8),
            fmt(error, 8),
            fmt(actual_err_h1, 8),
            fmt(actual_err_h2, 8),
            fmt(actual_err_ref, 8),
        ])

    headers = [
        "method", "p",
        "I(h1)", "I(h2)", "I_refined",
        "|R| RRR (h2)",
        "|I(h1)-I*|", "|I(h2)-I*|", "|I_ref-I*|",
    ]
    print_table(headers, rows)
    print()


def main():
    # Variant 13:
    # y = x^2 / (x^3 - 27),  X0 = -2,  Xk = 2,  h1 = 1.0,  h2 = 0.5
    a = -2.0
    b = 2.0
    h1 = 1.0
    h2 = 0.5

    print("=" * 78)
    print("Lab 12. Numerical integration")
    print("Task 3.5: compute F = integral from X0 to Xk of y dx using")
    print("          left/right/midpoint rectangle, trapezoid and Simpson rules.")
    print("          Estimate the accuracy with the Runge-Romberg-Richardson rule.")
    print()
    print("Variant 13:")
    print("  y(x) = x^2 / (x^3 - 27)")
    print(f"  X0 = {fmt(a, 1)}, Xk = {fmt(b, 1)}, h1 = {fmt(h1, 2)}, h2 = {fmt(h2, 2)}")
    print()

    exact = antiderivative(b) - antiderivative(a)
    print("Exact value of the integral (for comparison only):")
    print("  F = (1/3) * ( ln|x^3 - 27| ) evaluated from -2 to 2")
    print(f"    = (1/3) * (ln 19 - ln 35)")
    print(f"    = {fmt(exact, 12)}")
    print()

    res1 = compute_for_step(a, b, h1, "h1")
    res2 = compute_for_step(a, b, h2, "h2")

    print_runge_romberg(res1, res2, exact)

    print("Final answer (using the smaller step h2 = 0.5 and RRR refinement):")
    print(f"  F (Simpson, h2)              ~= {fmt(res2['simp'], 10)}")
    p_simp = 4
    refined_simp, err_simp = runge_romberg(res2['simp'], res1['simp'], p_simp, h1 / h2)
    print(f"  F (Simpson, RRR refined)     ~= {fmt(refined_simp, 10)}")
    print(f"  RRR error estimate (Simpson) ~= {fmt(err_simp, 10)}")
    print(f"  Exact value                  =  {fmt(exact, 10)}")


if __name__ == "__main__":
    main()
