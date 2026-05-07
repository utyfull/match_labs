import cmath
import math
from pathlib import Path

import matplotlib.pyplot as plt


def f1(x1, x2, a):
    return (x1 * x1) / (a * a) + (4.0 * x2 * x2) / (a * a) - 1.0


def f2(x1, x2, a):
    return a * x2 - math.exp(x1) - x1


def ellipse_upper_part(x1, a):
    arg = a * a - x1 * x1
    if arg < 0.0:
        arg = 0.0
    return 0.5 * math.sqrt(arg)


def curve_part(x1, a):
    return (math.exp(x1) + x1) / a


def reduced_function(x1, a):
    return ellipse_upper_part(x1, a) - curve_part(x1, a)


def find_positive_x_axis_intersection(a, left=0.0, right=0.8, tol=1e-12, max_iter=1000):
    f_left = reduced_function(left, a)
    f_right = reduced_function(right, a)

    if f_left * f_right > 0.0:
        raise ValueError("Positive x-axis intersection is not bracketed on the given interval.")

    for _ in range(max_iter):
        mid = (left + right) / 2.0
        f_mid = reduced_function(mid, a)

        if abs(f_mid) <= tol or (right - left) / 2.0 <= tol:
            return mid

        if f_left * f_mid <= 0.0:
            right = mid
            f_right = f_mid
        else:
            left = mid
            f_left = f_mid

    return (left + right) / 2.0


def choose_start_from_positive_x_axis_intersection(a, digits=2):
    intersection_x = find_positive_x_axis_intersection(a)
    if intersection_x <= 0.0:
        raise ValueError("The selected x-axis intersection is not positive.")

    intersection_y = ellipse_upper_part(intersection_x, a)

    # For a graphical start, use a rounded point near the solution, not the exact solution.
    start_x = round(intersection_x, digits)
    start_y = round(ellipse_upper_part(start_x, a), digits)
    return start_x, start_y, intersection_x, intersection_y


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


def phi_jacobian(x1, x2, a):
    denom = a * x2 - x1
    sqrt_arg = a * a - x1 * x1

    if denom <= 0.0 or sqrt_arg <= 0.0:
        raise ValueError("Phi derivatives are undefined at the selected point.")

    dphi1_dx1 = -1.0 / denom
    dphi1_dx2 = a / denom
    dphi2_dx1 = -x1 / (2.0 * math.sqrt(sqrt_arg))
    dphi2_dx2 = 0.0

    return dphi1_dx1, dphi1_dx2, dphi2_dx1, dphi2_dx2


def spectral_radius_2x2(m11, m12, m21, m22):
    trace = m11 + m22
    det = m11 * m22 - m12 * m21
    discriminant = trace * trace - 4.0 * det

    lambda1 = (trace + cmath.sqrt(discriminant)) / 2.0
    lambda2 = (trace - cmath.sqrt(discriminant)) / 2.0

    return max(abs(lambda1), abs(lambda2))


def check_simple_iteration_convergence(a, center_x, center_y, radius=0.05, samples=11):
    max_rho = 0.0
    worst_point = (center_x, center_y)

    for ix in range(samples):
        x1 = center_x - radius + 2.0 * radius * ix / (samples - 1)
        for iy in range(samples):
            x2 = center_y - radius + 2.0 * radius * iy / (samples - 1)
            m11, m12, m21, m22 = phi_jacobian(x1, x2, a)
            rho = spectral_radius_2x2(m11, m12, m21, m22)

            if rho > max_rho:
                max_rho = rho
                worst_point = (x1, x2)

    return max_rho < 1.0, max_rho, worst_point


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


def plot_localization(a, start_x, start_y):
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
            y = ellipse_upper_part(x, a)
            x_ellipse.append(x)
            y_ellipse_up.append(y)
            y_ellipse_dn.append(-y)

    # Exponential-like curve from second equation
    y_curve = [curve_part(x, a) for x in xs]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x_ellipse, y_ellipse_up, color="tab:blue", linewidth=2, label="f1=0 (upper ellipse branch)")
    ax.plot(x_ellipse, y_ellipse_dn, color="tab:blue", linewidth=2, linestyle="--", label="f1=0 (lower branch)")
    ax.plot(xs, y_curve, color="tab:red", linewidth=2, label="f2=0: x2=(e^x1+x1)/a")

    # The start point for both methods is chosen graphically near the curve intersection.
    ax.scatter([start_x], [start_y], color="black", marker="*", s=190, zorder=6,
               label=f"graphical start ~= ({start_x:.2f},{start_y:.2f})")
    ax.annotate("start point",
                xy=(start_x, start_y),
                xytext=(24, -30),
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.1})

    ax.axhline(0.0, color="black", linewidth=1)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.grid(alpha=0.25)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-1.2, 1.5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Lab7: Graphical Start Near System Curves Intersection")
    ax.legend(loc="best", fontsize=9)

    out_path = Path(__file__).resolve().parent / "localization_plot.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Plot saved to: {out_path}")
    print()


def plot_x_axis_intersection(a, intersection_x, start_x):
    x_min = -0.2
    x_max = 0.8
    steps = 800

    xs = [x_min + (x_max - x_min) * i / steps for i in range(steps + 1)]
    ys = [reduced_function(x, a) for x in xs]
    start_y = reduced_function(start_x, a)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, color="tab:green", linewidth=2,
            label="g(x1) = upper ellipse branch - curve")
    ax.axhline(0.0, color="black", linewidth=1.1, label="Ox")
    ax.axvline(0.0, color="black", linewidth=0.9)

    ax.scatter([intersection_x], [0.0], color="tab:orange", s=90, zorder=5,
               label=f"positive Ox crossing x1 ~= {intersection_x:.4f}")
    ax.scatter([start_x], [start_y], color="black", marker="*", s=180, zorder=6,
               label=f"rounded start x1 = {start_x:.2f}")
    ax.annotate("positive crossing",
                xy=(intersection_x, 0.0),
                xytext=(-92, 34),
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": "tab:orange", "linewidth": 1.1})
    ax.annotate("start x1",
                xy=(start_x, start_y),
                xytext=(22, -34),
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.1})

    ax.set_title("Lab7: Start From Positive Intersection With Ox")
    ax.set_xlabel("x1")
    ax.set_ylabel("g(x1)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    out_path = Path(__file__).resolve().parent / "x_axis_intersection_plot.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Plot saved to: {out_path}")
    print()


def main():
    # Variant 13: a = 2
    # System:
    # x1^2/a^2 + 4*x2^2/a^2 - 1 = 0
    # a*x2 - e^x1 - x1 = 0
    a = 2.0
    eps = 1e-6

    # The same start point is used for both methods.
    # It is chosen graphically near the positive intersection of the reduced
    # function g(x1) with Ox, so it is rounded instead of being the exact root.
    start_x, start_y, exact_start_x, exact_start_y = choose_start_from_positive_x_axis_intersection(a)
    x0_iter, y0_iter = start_x, start_y
    x0_newton, y0_newton = start_x, start_y

    converges, max_rho, worst_point = check_simple_iteration_convergence(a, x0_iter, y0_iter)

    xi, yi, ki, rows_i = simple_iteration_system(x0_iter, y0_iter, a, eps)
    xn, yn, kn, rows_n = newton_system(x0_newton, y0_newton, a, eps)

    print(f"a = {a}")
    print(f"eps = {eps}")
    print(f"Positive intersection of g(x1) with Ox: x1 = {exact_start_x:.10f}")
    print(f"Matching system point: ({exact_start_x:.10f}, {exact_start_y:.10f})")
    print(f"Rounded graphical start point for both methods: x0 = ({start_x:.2f}, {start_y:.2f})")
    print("Simple iteration convergence pre-check:")
    print("rho(Phi'(x)) < 1 in local start neighborhood ->", converges)
    print(f"max rho(Phi') ~= {max_rho:.6f} at ({worst_point[0]:.4f}, {worst_point[1]:.4f})")
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

    plot_localization(a, start_x, start_y)
    plot_x_axis_intersection(a, exact_start_x, start_x)

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
