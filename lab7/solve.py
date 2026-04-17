import math
from pathlib import Path

import matplotlib.pyplot as plt


def f1(x1, x2, a):
    return (x1 * x1) / (a * a) + (4.0 * x2 * x2) / (a * a) - 1.0


def f2(x1, x2, a):
    return a * x2 - math.exp(x1) - x1


def phi1(x1, x2, a):
    arg = a * x2 - x1
    if arg <= 1e-12:
        arg = 1e-12
    return math.log(arg)


def phi2(x1, x2, a):
    arg = a * a - x1 * x1
    if arg < 0.0:
        arg = 0.0
    return 0.5 * math.sqrt(arg)


def jacobian(x1, x2, a):
    j11 = 2.0 * x1 / (a * a)
    j12 = 8.0 * x2 / (a * a)
    j21 = -math.exp(x1) - 1.0
    j22 = a
    return j11, j12, j21, j22


def simple_iteration_system(x10, x20, a, eps, max_iter=100000):
    rows = []
    x1, x2 = x10, x20
    rows.append((0, x1, x2, f1(x1, x2, a), f2(x1, x2, a), 0.0))

    k = 0
    while k < max_iter:
        x1n = phi1(x1, x2, a)
        x2n = phi2(x1, x2, a)
        k += 1

        err = max(abs(x1n - x1), abs(x2n - x2))
        rows.append((k, x1n, x2n, f1(x1n, x2n, a), f2(x1n, x2n, a), err))

        if err <= eps:
            return x1n, x2n, k, rows

        x1, x2 = x1n, x2n

    return x1, x2, k, rows


def newton_system(x10, x20, a, eps, max_iter=100000):
    rows = []
    x1, x2 = x10, x20
    rows.append((0, x1, x2, f1(x1, x2, a), f2(x1, x2, a), 0.0))

    k = 0
    while k < max_iter:
        fv1 = f1(x1, x2, a)
        fv2 = f2(x1, x2, a)
        j11, j12, j21, j22 = jacobian(x1, x2, a)

        det = j11 * j22 - j12 * j21

        dx1 = (-fv1 * j22 + j12 * fv2) / det
        dx2 = (-j11 * fv2 + fv1 * j21) / det

        x1n = x1 + dx1
        x2n = x2 + dx2
        k += 1

        err = max(abs(dx1), abs(dx2))
        rows.append((k, x1n, x2n, f1(x1n, x2n, a), f2(x1n, x2n, a), err))

        if err <= eps:
            return x1n, x2n, k, rows

        x1, x2 = x1n, x2n

    return x1, x2, k, rows


def print_rows(title, rows):
    print(title)
    print("k      x1_k         x2_k         f1(x_k)      f2(x_k)      eps_k")
    for k, x1, x2, r1, r2, e in rows:
        print(f"{k:<2d}  {x1:11.7f}  {x2:11.7f}  {r1:11.7f}  {r2:11.7f}  {e:11.7f}")
    print()


def plot_localization(a, x0_iter, y0_iter, x0_newton, y0_newton, xi, yi, xn, yn):
    # Graphical localization of solutions for:
    # x1^2/a^2 + 4*x2^2/a^2 - 1 = 0  (ellipse)
    # a*x2 - e^x1 - x1 = 0          (curve)
    x_min = -2.2
    x_max = 1.2
    steps = 900
    xs = [x_min + (x_max - x_min) * i / steps for i in range(steps + 1)]

    # Ellipse branches
    x_ellipse = []
    y_ellipse_up = []
    y_ellipse_dn = []
    for x in xs:
        v = a * a - x * x
        if v >= 0.0:
            y = 0.5 * math.sqrt(v)
            x_ellipse.append(x)
            y_ellipse_up.append(y)
            y_ellipse_dn.append(-y)

    # Exponential-like curve from second equation
    y_curve = [(math.exp(x) + x) / a for x in xs]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x_ellipse, y_ellipse_up, color="tab:blue", linewidth=2, label="f1=0 (upper ellipse branch)")
    ax.plot(x_ellipse, y_ellipse_dn, color="tab:blue", linewidth=2, linestyle="--", label="f1=0 (lower branch)")
    ax.plot(xs, y_curve, color="tab:red", linewidth=2, label="f2=0: x2=(e^x1+x1)/a")

    # Highlight region where positive solution is searched.
    x_left, x_right = 0.0, 0.8
    y_bottom, y_top = 0.7, 1.1
    ax.axvspan(x_left, x_right, ymin=(y_bottom + 1.2) / 2.7, ymax=(y_top + 1.2) / 2.7, color="gold", alpha=0.20)
    ax.plot([x_left, x_right, x_right, x_left, x_left],
            [y_bottom, y_bottom, y_top, y_top, y_bottom],
            color="goldenrod", linewidth=1.5, label="positive root search box")

    # Initial guesses and found roots.
    ax.scatter([x0_iter], [y0_iter], color="tab:green", s=50, zorder=4, label=f"x0 iter=({x0_iter:.2f},{y0_iter:.2f})")
    ax.scatter([x0_newton], [y0_newton], color="tab:purple", s=50, zorder=4, label=f"x0 Newton=({x0_newton:.2f},{y0_newton:.2f})")
    ax.scatter([xi], [yi], color="black", s=65, marker="x", zorder=5, label=f"iter root≈({xi:.4f},{yi:.4f})")
    ax.scatter([xn], [yn], color="black", s=65, marker="o", facecolors="none", zorder=5, label=f"Newton root≈({xn:.4f},{yn:.4f})")

    ax.axhline(0.0, color="black", linewidth=1)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.grid(alpha=0.25)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-1.2, 1.5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Lab7: Graphical Localization Of Nonlinear System Roots")
    ax.legend(loc="best", fontsize=9)

    out_path = Path("lab7") / "localization_plot.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.show()

    print(f"Plot saved to: {out_path}")
    print()


def main():
    # Variant 13: a = 2
    # System:
    # x1^2/a^2 + 4*x2^2/a^2 - 1 = 0
    # a*x2 - e^x1 - x1 = 0
    a = 2.0
    eps = 1e-6

    # Initial approximation chosen graphically near positive intersection.
    x0_iter, y0_iter = 0.40, 0.95
    x0_newton, y0_newton = 0.50, 0.90

    xi, yi, ki, rows_i = simple_iteration_system(x0_iter, y0_iter, a, eps)
    xn, yn, kn, rows_n = newton_system(x0_newton, y0_newton, a, eps)

    print(f"a = {a}")
    print(f"eps = {eps}")
    print()

    print_rows("Simple iteration method (system):", rows_i)
    print_rows("Newton method (system):", rows_n)

    print(f"Simple iteration root: x1={xi:.10f}, x2={yi:.10f}, iterations={ki}")
    print(f"Newton root:           x1={xn:.10f}, x2={yn:.10f}, iterations={kn}")
    print()

    res_i = max(abs(f1(xi, yi, a)), abs(f2(xi, yi, a)))
    res_n = max(abs(f1(xn, yn, a)), abs(f2(xn, yn, a)))
    delta = max(abs(xi - xn), abs(yi - yn))

    print("Checks:")
    print(f"max(|f1|,|f2|) for simple iteration = {res_i:.6e}")
    print(f"max(|f1|,|f2|) for Newton           = {res_n:.6e}")
    print(f"max difference between roots        = {delta:.6e}")
    print()

    plot_localization(a, x0_iter, y0_iter, x0_newton, y0_newton, xi, yi, xn, yn)

    if kn < ki:
        print("Conclusion: Newton converged faster.")
    elif kn > ki:
        print("Conclusion: Simple iteration converged faster.")
    else:
        print("Conclusion: both methods used the same number of iterations.")


if __name__ == "__main__":
    main()

# Неформально про 7-ю лабу:
#
# 1) Что решаем:
#    f1(x1,x2)=x1^2/a^2 + 4*x2^2/a^2 - 1 = 0
#    f2(x1,x2)=a*x2 - e^x1 - x1 = 0
#    Для варианта 13 берём a=2.
#
# 2) Почему делаем график:
#    по условию старт надо выбрать графически.
#    На графике видно две точки пересечения:
#    - отрицательная (x1<0, x2<0),
#    - положительная (x1>0, x2>0).
#    Нам нужен положительный корень, поэтому старт берем рядом с ним.
#
# 3) Простая итерация для системы:
#    используем вид
#    x1 = ln(a*x2 - x1),
#    x2 = 0.5*sqrt(a^2 - x1^2).
#    На каждой итерации подставляем старые x1,x2 и получаем новые.
#    Остановка: max(|x1(k)-x1(k-1)|, |x2(k)-x2(k-1)|) <= eps.
#
# 4) Метод Ньютона (обобщенный):
#    решаем систему J(xk)*dx = -F(xk),
#    потом x(k+1)=x(k)+dx.
#    Для 2x2 решаем через формулы для определителей.
#    Обычно этот метод сходится быстрее, и здесь это тоже видно.
#
# 5) Что показывают таблицы:
#    - x1_k, x2_k: как идем к корню,
#    - f1, f2: как падает невязка,
#    - eps_k: как падает шаг между итерациями.
#    Это и есть зависимость погрешности от числа итераций.
#
# 6) На текущем запуске при eps=1e-6:
#    - простая итерация: (x1, x2) ~= (0.4249021829, 0.9771717093), 13 итераций;
#    - Ньютон:           (x1, x2) ~= (0.4249023332, 0.9771716849), 4 итерации.
#    То есть метод Ньютона здесь заметно быстрее.
