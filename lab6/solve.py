import math
from pathlib import Path

import matplotlib.pyplot as plt


def left_part(x):
    return math.log(x + 1.0) + 0.5


def right_part(x):
    return 2.0 * x


def f(x):
    return left_part(x) - right_part(x)


def df(x):
    return 1.0 / (x + 1.0) - 2.0


def d2f(x):
    return -1.0 / ((x + 1.0) * (x + 1.0))


def phi(x):
    return (math.log(x + 1.0) + 0.5) / 2.0


def dphi(x):
    return 1.0 / (2.0 * (x + 1.0))


def find_positive_x_axis_intersection(a, b, tol=1e-12, max_iter=1000):
    left = a
    right = b
    f_left = f(left)
    f_right = f(right)

    if f_left * f_right > 0.0:
        raise ValueError("Positive x-axis intersection is not bracketed on the given interval.")

    for _ in range(max_iter):
        mid = (left + right) / 2.0
        f_mid = f(mid)

        if abs(f_mid) <= tol or (right - left) / 2.0 <= tol:
            return mid

        if f_left * f_mid <= 0.0:
            right = mid
            f_right = f_mid
        else:
            left = mid
            f_left = f_mid

    return (left + right) / 2.0


def choose_start_from_positive_x_axis_intersection(a, b, digits=2):
    intersection_x = find_positive_x_axis_intersection(a, b)
    if intersection_x <= 0.0:
        raise ValueError("The selected x-axis intersection is not positive.")

    # For a graphical start, use a rounded value near the root, not the exact root itself.
    return round(intersection_x, digits), intersection_x


def choose_newton_start_by_second_derivative(a, b):
    candidates = [a, b]
    for x0 in candidates:
        if f(x0) * d2f(x0) > 0.0:
            return x0

    raise ValueError("No interval endpoint satisfies f(x0) * f''(x0) > 0 for Newton start.")


def simple_iteration_method(x0, q, eps, max_iter=100000):
    rows = []
    k = 0
    x_prev = x0
    rows.append((k, x_prev, f(x_prev), 0.0))

    factor = q / (1.0 - q)

    while k < max_iter:
        x_next = phi(x_prev)
        k += 1

        delta = abs(x_next - x_prev)
        est = factor * delta
        rows.append((k, x_next, f(x_next), est))

        if est <= eps:
            return x_next, k, rows

        x_prev = x_next

    return x_prev, k, rows


def newton_method(x0, eps, max_iter=100000):
    rows = []
    k = 0
    x_prev = x0
    rows.append((k, x_prev, f(x_prev), 0.0))

    while k < max_iter:
        x_next = x_prev - f(x_prev) / df(x_prev)
        k += 1

        est = abs(x_next - x_prev)
        rows.append((k, x_next, f(x_next), est))

        if est <= eps:
            return x_next, k, rows

        x_prev = x_next

    return x_prev, k, rows


def print_rows(title, rows):
    print(title)
    print("k        x_k           f(x_k)        eps_k")
    for k, xk, fxk, ek in rows:
        print(f"{k:<2d}  {xk:12.8f}  {fxk:12.8f}  {ek:12.8f}")
    print()


def plot_initial_region(a, b, iteration_start_x, newton_start_x):
    left = max(-0.99, a - 0.3)
    right = b + 0.3
    steps = 800

    xs = [left + (right - left) * i / steps for i in range(steps + 1)]
    ys_left = [left_part(x) for x in xs]
    ys_right = [right_part(x) for x in xs]

    iteration_start_y = left_part(iteration_start_x)
    newton_start_y = left_part(newton_start_x)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys_left, color="tab:blue", linewidth=2, label="y1 = ln(x+1) + 0.5")
    ax.plot(xs, ys_right, color="tab:red", linewidth=2, label="y2 = 2x")

    ax.scatter([iteration_start_x], [iteration_start_y], color="black", marker="*", s=180, zorder=5,
               label=f"iteration start ~= ({iteration_start_x:.2f}, {iteration_start_y:.2f})")
    ax.scatter([newton_start_x], [newton_start_y], color="tab:purple", marker="D", s=90, zorder=5,
               label=f"Newton start by f*f''>0 ~= ({newton_start_x:.2f}, {newton_start_y:.2f})")
    ax.annotate("iteration start",
                xy=(iteration_start_x, iteration_start_y),
                xytext=(22, -32),
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.1})
    ax.annotate("Newton start",
                xy=(newton_start_x, newton_start_y),
                xytext=(-100, 36),
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": "tab:purple", "linewidth": 1.1})

    ax.set_title("Lab6: Graphical Start Near Equation Parts Intersection")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    out_path = Path(__file__).resolve().parent / "start_region_plot.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Plot saved to: {out_path}")
    print()


def plot_x_axis_intersection(a, b, intersection_x, iteration_start_x, newton_start_x):
    left = max(-0.99, a - 0.2)
    right = b + 0.2
    steps = 800

    xs = [left + (right - left) * i / steps for i in range(steps + 1)]
    ys = [f(x) for x in xs]
    iteration_start_y = f(iteration_start_x)
    newton_start_y = f(newton_start_x)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, color="tab:green", linewidth=2, label="f(x) = ln(x+1) - 2x + 0.5")
    ax.axhline(0.0, color="black", linewidth=1.1, label="Ox")
    ax.axvline(0.0, color="black", linewidth=0.9)

    ax.scatter([intersection_x], [0.0], color="tab:orange", s=90, zorder=5,
               label=f"positive Ox crossing x ~= {intersection_x:.4f}")
    ax.scatter([iteration_start_x], [iteration_start_y], color="black", marker="*", s=180, zorder=6,
               label=f"iteration start x0 = {iteration_start_x:.2f}")
    ax.scatter([newton_start_x], [newton_start_y], color="tab:purple", marker="D", s=90, zorder=6,
               label=f"Newton start x0 = {newton_start_x:.2f}, f*f''>0")
    ax.annotate("positive crossing",
                xy=(intersection_x, 0.0),
                xytext=(-86, 34),
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": "tab:orange", "linewidth": 1.1})
    ax.annotate("iteration x0",
                xy=(iteration_start_x, iteration_start_y),
                xytext=(22, -34),
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.1})
    ax.annotate("Newton x0",
                xy=(newton_start_x, newton_start_y),
                xytext=(-104, -34),
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": "tab:purple", "linewidth": 1.1})

    ax.set_title("Lab6: Start From Positive Intersection With Ox")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    out_path = Path(__file__).resolve().parent / "x_axis_intersection_plot.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Plot saved to: {out_path}")
    print()


def main():
    # Equation:
    # ln(x + 1) - 2x + 0.5 = 0
    eps = 1e-6

    # f(0) > 0, f(0.5) < 0  =>  [0, 0.5]
    a = 0.0
    b = 0.5

    # Simple iteration setup:
    # x = phi(x) = (ln(x+1) + 0.5)/2
    # q = max |phi'(x)| on [a,b]
    q = max(abs(dphi(a)), abs(dphi(b)))

    # For simple iteration, use a graphical start near the positive root.
    # For Newton, choose the endpoint satisfying f(x0) * f''(x0) > 0.
    start_x, exact_intersection_x = choose_start_from_positive_x_axis_intersection(a, b)
    x0_iter = start_x
    x0_newton = choose_newton_start_by_second_derivative(a, b)

    x_iter, k_iter, rows_iter = simple_iteration_method(x0_iter, q, eps)
    x_newton, k_newton, rows_newton = newton_method(x0_newton, eps)

    print(f"eps = {eps}")
    print(f"Interval for positive root: [{a}, {b}]")
    print(f"Positive intersection of f(x) with Ox: x = {exact_intersection_x:.10f}")
    print(f"Rounded graphical start point for simple iteration: x0 = {x0_iter:.2f}")
    print(f"Newton start by f(x0)*f''(x0)>0: x0 = {x0_newton:.2f}")
    print(f"f(x0)*f''(x0) for Newton start = {f(x0_newton) * d2f(x0_newton):.6e}")
    print(f"q = max|phi'(x)| on [{a}, {b}] = {q:.6f}")
    print()

    print_rows("Simple iteration method:", rows_iter)
    print_rows("Newton method:", rows_newton)

    print(f"Simple iteration root: x = {x_iter:.10f}, iterations = {k_iter}")
    print(f"Newton root:           x = {x_newton:.10f}, iterations = {k_newton}")
    print()

    res_iter = abs(f(x_iter))
    res_newton = abs(f(x_newton))
    delta_roots = abs(x_iter - x_newton)
    in_iter = (a <= x_iter <= b)
    in_newton = (a <= x_newton <= b)

    print("Checks:")
    print(f"|f(x_iter)|   = {res_iter:.6e}")
    print(f"|f(x_newton)| = {res_newton:.6e}")
    print(f"|x_iter - x_newton| = {delta_roots:.6e}")
    print(f"x_iter in [{a}, {b}]   -> {in_iter}")
    print(f"x_newton in [{a}, {b}] -> {in_newton}")
    print()

    plot_initial_region(a, b, x0_iter, x0_newton)
    plot_x_axis_intersection(a, b, exact_intersection_x, x0_iter, x0_newton)

    if k_newton < k_iter:
        print("Conclusion: Newton converged faster.")
    elif k_newton > k_iter:
        print("Conclusion: Simple iteration converged faster.")
    else:
        print("Conclusion: both methods needed the same number of iterations.")


if __name__ == "__main__":
    main()

# Уравнение:
# ln(x+1) - 2x + 0.5 = 0
#
# 1) Где положительный корень:
#    f(0) = 0.5 > 0
#    f(0.5) = ln(1.5) - 1 + 0.5 < 0
#    Значит положительный корень точно между 0 и 0.5.
#
# 2) Метод простой итерации:
#    Переписали как x = phi(x):
#    phi(x) = (ln(x+1) + 0.5)/2.
#    Старт взяли x0 = (a+b)/2 = 0.25.
#    На [0, 0.5]: |phi'(x)| <= 0.5, то есть q=0.5 < 1, сходимость есть.
#    Оценка ошибки на шаге:
#    eps_k = q/(1-q) * |x_k - x_(k-1)|.
#
# 3) Метод Ньютона:
#    x_(k+1) = x_k - f(x_k)/f'(x_k),
#    где f'(x) = 1/(x+1) - 2.
#    Для старта взяли x0 = 0.5 (правая граница),
#    потому что f(x0)*f''(x0) > 0.
#    Тут оценка ошибки просто:
#    eps_k = |x_k - x_(k-1)|.
#
# 4) Что сравниваем:
#    По таблицам видно, как уменьшается eps_k с ростом k.
#    Это и есть зависимость погрешности от количества итераций.
#    Обычно Ньютон заметно быстрее, и у этой задачи тоже.
#
# 5) На текущем запуске при eps=1e-6:
#    - простая итерация: x ~= 0.4282112366, 13 итераций;
#    - Ньютон:           x ~= 0.4282114709, 3 итерации.
#    Разница по скорости хорошо видна по таблицам eps_k.
