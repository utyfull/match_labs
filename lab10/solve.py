import math
from pathlib import Path

import matplotlib.pyplot as plt


def f(x):
    return math.cos(x) + x


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


def determinant(matrix):
    n = len(matrix)
    a = [row[:] for row in matrix]
    det = 1.0

    for col in range(n):
        pivot = col
        for row in range(col + 1, n):
            if abs(a[row][col]) > abs(a[pivot][col]):
                pivot = row

        if abs(a[pivot][col]) < 1e-15:
            return 0.0

        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            det *= -1.0

        pivot_value = a[col][col]
        det *= pivot_value

        for row in range(col + 1, n):
            factor = a[row][col] / pivot_value
            for j in range(col, n):
                a[row][j] -= factor * a[col][j]

    return det


def replace_column(matrix, rhs, col):
    result = [row[:] for row in matrix]
    for i in range(len(matrix)):
        result[i][col] = rhs[i]
    return result


def solve_by_cramer(matrix, rhs):
    det_a = determinant(matrix)
    if abs(det_a) < 1e-15:
        raise ZeroDivisionError("The normal system determinant is zero.")

    solution = []
    dets = []
    for col in range(len(matrix)):
        det_col = determinant(replace_column(matrix, rhs, col))
        dets.append(det_col)
        solution.append(det_col / det_a)

    return solution, det_a, dets


def build_normal_system(xs, ys, degree):
    size = degree + 1
    matrix = [[0.0] * size for _ in range(size)]
    rhs = [0.0] * size

    for k in range(size):
        for j in range(size):
            matrix[k][j] = sum(x ** (j + k) for x in xs)
        rhs[k] = sum(y * (x ** k) for x, y in zip(xs, ys))

    return matrix, rhs


def polynomial_value(coeffs, x):
    value = 0.0
    power = 1.0
    for coef in coeffs:
        value += coef * power
        power *= x
    return value


def squared_error(xs, ys, coeffs):
    return sum((polynomial_value(coeffs, x) - y) ** 2 for x, y in zip(xs, ys))


def normal_system_residual(matrix, rhs, coeffs):
    n = len(matrix)
    residuals = []
    for k in range(n):
        lhs = sum(matrix[k][j] * coeffs[j] for j in range(n))
        residuals.append(lhs - rhs[k])
    max_abs = max(abs(r) for r in residuals) if residuals else 0.0
    return residuals, max_abs


def node_residuals(xs, ys, coeffs):
    return [polynomial_value(coeffs, x) - y for x, y in zip(xs, ys)]


def polynomial_to_string(name, coeffs):
    terms = []
    for power, coef in enumerate(coeffs):
        coef = clean(coef)
        if coef == 0.0:
            continue

        if power == 0:
            body = fmt(abs(coef), 8)
        elif power == 1:
            body = f"{fmt(abs(coef), 8)}*x"
        else:
            body = f"{fmt(abs(coef), 8)}*x^{power}"

        sign = "-" if coef < 0.0 else "+"
        if not terms:
            terms.append(body if sign == "+" else f"-{body}")
        else:
            terms.append(f" {sign} {body}")

    return f"{name}(x) = {''.join(terms) if terms else '0'}"


def print_normal_system(degree, matrix, rhs):
    rows = []
    for k, row in enumerate(matrix):
        parts = []
        for j, value in enumerate(row):
            parts.append(f"{fmt(value, 6)}*a_{j}")
        rows.append([str(k), " + ".join(parts), fmt(rhs[k], 10)])

    print(f"Normal system for polynomial degree {degree}:")
    print_table(["k", "left side", "right side"], rows)
    print()


def print_solution_details(degree, coeffs, det_a, dets, error):
    print(f"Cramer's rule determinants for degree {degree}:")
    rows = [["det A", fmt(det_a, 10)]]
    for i, det_col in enumerate(dets):
        rows.append([f"det A_{i}", fmt(det_col, 10)])
    print_table(["determinant", "value"], rows)
    print()

    rows = []
    for i, coef in enumerate(coeffs):
        rows.append([f"a_{i}", fmt(coef, 10)])
    print(f"Coefficients for degree {degree}:")
    print_table(["coefficient", "value"], rows)
    print(polynomial_to_string(f"P{degree}", coeffs))
    print(f"Sum of squared errors Phi_{degree} = {fmt(error, 12)}")
    print()


def plot_result(xs, ys, coeffs_by_degree):
    x_min = min(xs) - 0.25
    x_max = max(xs) + 0.25
    steps = 500
    x_plot = [x_min + (x_max - x_min) * i / steps for i in range(steps + 1)]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_original = [f(x) for x in x_plot]
    ax.plot(x_plot, y_original, color="black", linewidth=2, linestyle="--", label="y = cos(x) + x")
    ax.scatter(xs, ys, color="black", s=60, zorder=5, label="table values")

    for degree, coeffs in coeffs_by_degree.items():
        y_plot = [polynomial_value(coeffs, x) for x in x_plot]
        ax.plot(x_plot, y_plot, linewidth=2, label=f"P{degree}(x)")

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.grid(alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Lab10: Least Squares Approximation")
    ax.legend(loc="best")

    out_path = Path(__file__).resolve().parent / "least_squares_plot.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Plot saved to: {out_path}")
    print()


def main():
    # Variant data:
    # i:    0       1      2       3       4      5
    # x_i: -1.0     0.0    1.0     2.0     3.0    4.0
    # y_i: -0.4597  1.0    1.5403  1.5839  2.010  3.3464
    xs = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    ys = [-0.4597, 1.0, 1.5403, 1.5839, 2.010, 3.3464]
    degrees = [1, 2]

    print("=" * 78)
    print("Lab 10. Method of least squares")
    print("The table values are rounded values of y = cos(x) + x.")
    print("Normal system: sum_j a_j * sum_i x_i^(j+k) = sum_i y_i*x_i^k")
    print("Here i = 0..5, so the number of table points is n + 1 = 6.")
    print()

    input_rows = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        input_rows.append([str(i), fmt(x, 10), fmt(y, 10)])
    print("Input data:")
    print_table(["i", "x_i", "y_i"], input_rows)
    print()

    max_power = 2 * max(degrees)
    power_rows = []
    for power in range(max_power + 1):
        power_rows.append([str(power), fmt(sum(x ** power for x in xs), 10)])
    print("Power sums S_k = sum_i x_i^k:")
    print_table(["k", "S_k"], power_rows)
    print()

    coeffs_by_degree = {}
    value_rows = []

    for degree in degrees:
        matrix, rhs = build_normal_system(xs, ys, degree)
        coeffs, det_a, dets = solve_by_cramer(matrix, rhs)
        error = squared_error(xs, ys, coeffs)
        coeffs_by_degree[degree] = coeffs

        print_normal_system(degree, matrix, rhs)
        print_solution_details(degree, coeffs, det_a, dets, error)

    for i, (x, y) in enumerate(zip(xs, ys)):
        row = [str(i), fmt(x, 10), fmt(y, 10)]
        for degree in degrees:
            row.append(fmt(polynomial_value(coeffs_by_degree[degree], x), 10))
        value_rows.append(row)

    print("Calculation results at table points:")
    print_table(["i", "x_i", "y_i", "P1(x_i)", "P2(x_i)"], value_rows)
    print()

    print("Verification:")
    for degree in degrees:
        coeffs = coeffs_by_degree[degree]
        matrix, rhs = build_normal_system(xs, ys, degree)
        _, max_normal_residual = normal_system_residual(matrix, rhs, coeffs)
        residuals = node_residuals(xs, ys, coeffs)
        max_node_residual = max(abs(r) for r in residuals)
        sse = squared_error(xs, ys, coeffs)
        print(f"  degree {degree}:")
        print(f"    max |A*a - b|         = {max_normal_residual:.3e}   (normal system solved)")
        print(f"    max |P{degree}(x_i) - y_i|   = {max_node_residual:.3e}   (best fit residual)")
        print(f"    sum (P{degree}(x_i) - y_i)^2 = {sse:.10e}   (least-squares functional Phi_{degree})")
    print()

    plot_result(xs, ys, coeffs_by_degree)

    e1 = squared_error(xs, ys, coeffs_by_degree[1])
    e2 = squared_error(xs, ys, coeffs_by_degree[2])
    if e2 < e1:
        print("Conclusion: the 2nd degree polynomial gives the smaller squared error.")
    elif e1 < e2:
        print("Conclusion: the 1st degree polynomial gives the smaller squared error.")
    else:
        print("Conclusion: both polynomials give the same squared error.")


if __name__ == "__main__":
    main()
