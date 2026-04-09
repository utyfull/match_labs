import math
from pathlib import Path

import matplotlib.pyplot as plt


def f(x):
    return math.log(x + 1.0) - 2.0 * x + 0.5


def df(x):
    return 1.0 / (x + 1.0) - 2.0


def d2f(x):
    return -1.0 / ((x + 1.0) * (x + 1.0))


def phi(x):
    return (math.log(x + 1.0) + 0.5) / 2.0


def dphi(x):
    return 1.0 / (2.0 * (x + 1.0))


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


def plot_initial_region(a, b, x0_iter, x0_newton, x_iter, x_newton):
    left = max(-0.99, a - 0.3)
    right = b + 0.3
    steps = 800

    xs = [left + (right - left) * i / steps for i in range(steps + 1)]
    ys = [f(x) for x in xs]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, color="tab:red", linewidth=2, label="f(x) = ln(x+1)-2x+0.5")
    ax.axhline(0.0, color="black", linewidth=1)

    # Highlight interval where the positive root and initial guesses are chosen.
    ax.axvspan(a, b, color="gold", alpha=0.25, label=f"start interval [{a}, {b}]")

    ax.axvline(x0_iter, color="tab:blue", linestyle="--", linewidth=1.5, label=f"x0 iter = {x0_iter:.3f}")
    ax.axvline(x0_newton, color="tab:green", linestyle="--", linewidth=1.5, label=f"x0 Newton = {x0_newton:.3f}")

    ax.scatter([x_iter], [0.0], color="tab:blue", s=40, zorder=3, label=f"iter root ~ {x_iter:.6f}")
    ax.scatter([x_newton], [0.0], color="tab:green", s=40, zorder=3, label=f"Newton root ~ {x_newton:.6f}")

    ax.set_title("Lab6: Start Region And Root Location")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    out_path = Path("lab6") / "start_region_plot.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()

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
    x0_iter = (a + b) / 2.0

    # Newton setup:
    # choose x0 so that f(x0)*f''(x0) > 0
    # at x0=b this condition is satisfied for this equation
    x0_newton = b

    x_iter, k_iter, rows_iter = simple_iteration_method(x0_iter, q, eps)
    x_newton, k_newton, rows_newton = newton_method(x0_newton, eps)

    print(f"eps = {eps}")
    print(f"Interval for positive root: [{a}, {b}]")
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

    plot_initial_region(a, b, x0_iter, x0_newton, x_iter, x_newton)

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
