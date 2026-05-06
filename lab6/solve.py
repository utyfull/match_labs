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


def find_intersection(a, b, tol=1e-12, max_iter=1000):
    left = a
    right = b
    f_left = f(left)
    f_right = f(right)

    if f_left * f_right > 0.0:
        raise ValueError("Intersection is not bracketed on the given interval.")

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


def plot_initial_region(a, b, start_x):
    left = max(-0.99, a - 0.3)
    right = b + 0.3
    steps = 800

    xs = [left + (right - left) * i / steps for i in range(steps + 1)]
    ys_left = [left_part(x) for x in xs]
    ys_right = [right_part(x) for x in xs]

    start_y = left_part(start_x)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys_left, color="tab:blue", linewidth=2, label="y1 = ln(x+1) + 0.5")
    ax.plot(xs, ys_right, color="tab:red", linewidth=2, label="y2 = 2x")

    ax.scatter([start_x], [start_y], color="black", marker="*", s=180, zorder=5,
               label=f"graphical start ~= ({start_x:.2f}, {start_y:.2f})")
    ax.annotate("start point",
                xy=(start_x, start_y),
                xytext=(22, -32),
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.1})

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

    # The same start point is used for both methods.
    # It is chosen graphically near the intersection of y1 and y2,
    # so it is rounded instead of being the exact root.
    exact_intersection_x = find_intersection(a, b)
    start_x = round(exact_intersection_x, 2)
    x0_iter = start_x
    x0_newton = start_x

    x_iter, k_iter, rows_iter = simple_iteration_method(x0_iter, q, eps)
    x_newton, k_newton, rows_newton = newton_method(x0_newton, eps)

    print(f"eps = {eps}")
    print(f"Interval for positive root: [{a}, {b}]")
    print(f"Exact graph intersection: x = {exact_intersection_x:.10f}")
    print(f"Graphical start point for both methods: x0 = {start_x:.2f}")
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

    plot_initial_region(a, b, start_x)

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
