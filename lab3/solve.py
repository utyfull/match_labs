def to_jacobi_form(a, b):
    n = len(a)
    alpha = [[0.0] * n for _ in range(n)]
    beta = [0.0] * n

    for i in range(n):
        diag = a[i][i]
        beta[i] = b[i] / diag
        for j in range(n):
            if i != j:
                alpha[i][j] = -a[i][j] / diag

    return alpha, beta


def vector_diff_norm_inf(x, y):
    mx = 0.0
    for i in range(len(x)):
        d = abs(x[i] - y[i])
        if d > mx:
            mx = d
    return mx


def matrix_norm_inf(m):
    mx = 0.0
    for row in m:
        s = 0.0
        for v in row:
            s += abs(v)
        if s > mx:
            mx = s
    return mx


def simple_iteration(alpha, beta, eps, max_iter=100000):
    n = len(beta)
    q = matrix_norm_inf(alpha)
    factor = q / (1.0 - q) if q < 1.0 else 1.0

    x_old = beta[:]
    k = 0

    while k < max_iter:
        k += 1
        x_new = [0.0] * n
        for i in range(n):
            s = beta[i]
            for j in range(n):
                s += alpha[i][j] * x_old[j]
            x_new[i] = s

        diff = vector_diff_norm_inf(x_new, x_old)
        est = factor * diff if q < 1.0 else diff
        if est <= eps:
            return x_new, k, est

        x_old = x_new

    return x_old, k, est


def seidel_iteration(alpha, beta, eps, max_iter=100000):
    n = len(beta)
    q = matrix_norm_inf(alpha)
    factor = q / (1.0 - q) if q < 1.0 else 1.0

    x_old = beta[:]
    k = 0

    while k < max_iter:
        k += 1
        x_new = x_old[:]

        for i in range(n):
            s = beta[i]
            for j in range(n):
                if j < i:
                    s += alpha[i][j] * x_new[j]
                elif j > i:
                    s += alpha[i][j] * x_old[j]
            x_new[i] = s

        diff = vector_diff_norm_inf(x_new, x_old)
        est = factor * diff if q < 1.0 else diff
        if est <= eps:
            return x_new, k, est

        x_old = x_new

    return x_old, k, est


def mat_vec_mul(a, x):
    n = len(a)
    m = len(a[0])
    res = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(m):
            s += a[i][j] * x[j]
        res[i] = s
    return res


def residual_norm_inf(a, x, b):
    ax = mat_vec_mul(a, x)
    mx = 0.0
    for i in range(len(b)):
        d = abs(ax[i] - b[i])
        if d > mx:
            mx = d
    return mx


def print_vector(name, v, digits=6):
    print(name)
    print("[" + ", ".join(f"{x:.{digits}f}" for x in v) + "]")


def main():
    # 24*x1 - 7*x2 - 4*x3 + 4*x4 = -190
    # -3*x1 - 9*x2 - 2*x3 - 2*x4 = -12
    # 3*x1 + 7*x2 + 24*x3 + 9*x4 = 155
    # x1 - 6*x2 - 2*x3 - 15*x4 = -17
    a = [
        [24.0, -7.0, -4.0, 4.0],
        [-3.0, -9.0, -2.0, -2.0],
        [3.0, 7.0, 24.0, 9.0],
        [1.0, -6.0, -2.0, -15.0],
    ]
    b = [-190.0, -12.0, 155.0, -17.0]
    eps = 1e-4

    alpha, beta = to_jacobi_form(a, b)
    q = matrix_norm_inf(alpha)

    x_jacobi, k_jacobi, est_jacobi = simple_iteration(alpha, beta, eps)
    x_seidel, k_seidel, est_seidel = seidel_iteration(alpha, beta, eps)

    print(f"eps = {eps}")
    print(f"q = ||alpha||_inf = {q:.6f}")
    print()

    print_vector("Jacobi (simple iteration) solution:", x_jacobi)
    print(f"Jacobi iterations: {k_jacobi}")
    print(f"Jacobi final estimate: {est_jacobi:.6e}")
    print(f"Jacobi residual ||Ax-b||_inf: {residual_norm_inf(a, x_jacobi, b):.6e}")
    print()

    print_vector("Seidel solution:", x_seidel)
    print(f"Seidel iterations: {k_seidel}")
    print(f"Seidel final estimate: {est_seidel:.6e}")
    print(f"Seidel residual ||Ax-b||_inf: {residual_norm_inf(a, x_seidel, b):.6e}")
    print()

    if k_seidel < k_jacobi:
        print("Conclusion: Seidel converged faster for this system.")
    elif k_seidel > k_jacobi:
        print("Conclusion: Jacobi converged faster for this system.")
    else:
        print("Conclusion: both methods needed the same number of iterations.")


if __name__ == "__main__":
    main()

#
# 1) Сначала приводим Ax=b к виду x = beta + alpha*x.
#    Это обычный шаг Якоби:
#    beta[i] = b[i]/a[i][i], alpha[i][j] = -a[i][j]/a[i][i], i != j.
#    На диагонали alpha[i][i] = 0.
#
# 2) Дальше считаем q = ||alpha||_inf (макс. сумма модулей по строке).
#    Для этой задачи q = 0.791667 < 1, значит сходимость есть.
#    И можно использовать оценку:
#    eps_k = q/(1-q) * ||x(k)-x(k-1)||_inf.
#
# 3) Метод простых итераций (Якоби):
#    x_new берется ТОЛЬКО из x_old:
#    x_new[i] = beta[i] + sum(alpha[i][j] * x_old[j]).
#    То есть вся новая итерация строится на старых значениях.
#
# 4) Метод Зейделя:
#    Когда считаем x_new[i], уже используем свежие x_new[0..i-1].
#    Остальные (i+1..n-1) пока берем из x_old.
#    Поэтому он обычно быстрее (и в этой задаче тоже).
#
# 5) Что взяли как старт:
#    x(0) = beta (как в методичке).
#    Для нашей системы:
#    beta = [-7.9167, 1.3333, 6.4583, 1.1333]
#
# 6) Пример первых шагов (приближенно):
#    Якоби:
#    x(1) ~= [-6.6403, 2.2852, 6.6340, -0.7889]
#    x(2) ~= [-6.0130, 2.2478, 6.9177, -1.1080]
#
#    Зейдель:
#    x(1) ~= [-6.6403, 1.8597, 6.3209, -0.8960]
#    x(2) ~= [-6.1714, 2.1849, 6.9285, -1.0759]
#
# 7) Когда останавливаемся:
#    как только текущая оценка ошибки <= eps.
#    В коде это est <= eps.
#
# 8) В конце печатаем:
#    - решение каждым методом,
#    - число итераций,
#    - финальную оценку ошибки,
#    - невязку ||Ax-b||_inf.
#    По итерациям сразу видно, кто сошелся быстрее.
