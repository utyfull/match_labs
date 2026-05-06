def lu_decomposition_with_pivoting(a):
    n = len(a)
    u = [row[:] for row in a]
    l = [[0.0] * n for _ in range(n)]
    p = list(range(n))  # перестановка строк (матрица P)
    swaps = 0

    for i in range(n):
        l[i][i] = 1.0  # единичная диагональ L

    for k in range(n):
        # 1) выбор главного элемента в столбце k (partial pivoting)
        pivot_row = k
        pivot_abs = abs(u[k][k])
        for i in range(k + 1, n):
            cur = abs(u[i][k])
            if cur > pivot_abs:
                pivot_abs = cur
                pivot_row = i

        if pivot_row != k:
            # 2) перестановка строк в U, P и уже посчитанной части L
            u[k], u[pivot_row] = u[pivot_row], u[k]
            p[k], p[pivot_row] = p[pivot_row], p[k]
            for j in range(k):
                l[k][j], l[pivot_row][j] = l[pivot_row][j], l[k][j]
            swaps += 1

        # 3) зануление элементов под ведущим в столбце k
        for i in range(k + 1, n):
            factor = u[i][k] / u[k][k]
            l[i][k] = factor
            u[i][k] = 0.0
            for j in range(k + 1, n):
                u[i][j] -= factor * u[k][j]

    return l, u, p, swaps


def permute_vector(v, p):
    return [v[p[i]] for i in range(len(v))]


def forward_substitution(l, b):
    n = len(l)
    z = [0.0] * n
    for i in range(n):
        # прямой ход: решаем L z = b
        s = b[i]
        for j in range(i):
            s -= l[i][j] * z[j]
        z[i] = s / l[i][i]
    return z


def backward_substitution(u, z):
    n = len(u)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        # обратный ход: решаем U x = z
        s = z[i]
        for j in range(i + 1, n):
            s -= u[i][j] * x[j]
        x[i] = s / u[i][i]
    return x


def solve_lu(l, u, p, b):
    # решение через LU с перестановкой: P*A = L*U
    pb = permute_vector(b, p)
    z = forward_substitution(l, pb)
    x = backward_substitution(u, z)
    return x


def determinant_from_u(u, swaps):
    # det(A) = (-1)^swaps * произведение диагонали U
    det_u = 1.0
    for i in range(len(u)):
        det_u *= u[i][i]
    return det_u if swaps % 2 == 0 else -det_u


def inverse_from_lu(l, u, p):
    n = len(l)
    inv = [[0.0] * n for _ in range(n)]

    for col in range(n):
        # колонка обратной: решаем A x = e_col
        e = [0.0] * n
        e[col] = 1.0
        x = solve_lu(l, u, p, e)
        for row in range(n):
            inv[row][col] = x[row]

    return inv


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


def mat_mul(a, b):
    n = len(a)
    m = len(b[0])
    p = len(b)
    c = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(p):
                s += a[i][k] * b[k][j]
            c[i][j] = s
    return c


def identity_matrix(n):
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        m[i][i] = 1.0
    return m


def matrix_diff_norm_inf(a, b):
    mx = 0.0
    for i in range(len(a)):
        for j in range(len(a[0])):
            d = abs(a[i][j] - b[i][j])
            if d > mx:
                mx = d
    return mx


def matrix_norm_inf(a):
    mx = 0.0
    for row in a:
        s = 0.0
        for v in row:
            s += abs(v)
        if s > mx:
            mx = s
    return mx


def vector_norm_inf(v):
    mx = 0.0
    for x in v:
        if abs(x) > mx:
            mx = abs(x)
    return mx


def residual_norm_inf(a, x, b):
    ax = mat_vec_mul(a, x)
    mx = 0.0
    for i in range(len(b)):
        d = abs(ax[i] - b[i])
        if d > mx:
            mx = d
    return mx


def permute_matrix_rows(a, p):
    return [a[p[i]][:] for i in range(len(p))]


def print_vector(name, v, digits=6):
    print(name)
    print("[" + ", ".join(f"{x:.{digits}f}" for x in v) + "]")


def print_matrix(name, m, digits=6):
    print(name)
    for row in m:
        print("[" + "  ".join(f"{x:.{digits}f}" for x in row) + "]")


def main():
    # -6*x1 - 5*x2 - 3*x3 - 8*x4 = 101
    #  5*x1 - 1*x2 - 5*x3 - 4*x4 = 51
    # -6*x1 + 0*x2 + 5*x3 + 5*x4 = -53
    # -7*x1 - 2*x2 + 8*x3 + 5*x4 = -63
    a = [
        [-6.0, -5.0, -3.0, -8.0],
        [5.0, -1.0, -5.0, -4.0],
        [-6.0, 0.0, 5.0, 5.0],
        [-7.0, -2.0, 8.0, 5.0],
    ]
    b = [101.0, 51.0, -53.0, -63.0]

    l, u, p, swaps = lu_decomposition_with_pivoting(a) # l - нижняя треугольная, u - верхняя треугольная, p - перестановка строк, swaps - количество перестановок
    x = solve_lu(l, u, p, b) # решение слау в два шага: 1) решаем lz = pb, 2) решаем ux = z
    det_a = determinant_from_u(u, swaps) # определитель матрицы A
    inv_a = inverse_from_lu(l, u, p) # обратная матрица от A
    ax = mat_vec_mul(a, x) # проверка решения: A*x
    residual = residual_norm_inf(a, x, b)
    # Относительная невязка — стабильнее абсолютной, показывает масштаб ошибки
    a_norm = matrix_norm_inf(a)
    x_norm = vector_norm_inf(x)
    b_norm = vector_norm_inf(b)
    rel_residual = residual / (a_norm * x_norm + b_norm) if (a_norm * x_norm + b_norm) != 0.0 else 0.0

    # Проверка корректности LU: P*A должно совпадать с L*U
    pa = permute_matrix_rows(a, p)
    lu = mat_mul(l, u)
    lu_err = matrix_diff_norm_inf(pa, lu)

    # Минимальный по модулю pivot в U — индикатор плохой обусловленности
    min_pivot = min(abs(u[i][i]) for i in range(len(u)))
    i_mat = identity_matrix(len(a))
    ainv_left = mat_mul(a, inv_a)
    ainv_right = mat_mul(inv_a, a)
    # Проверяем корректность обратной с двух сторон
    inv_err_left = matrix_diff_norm_inf(ainv_left, i_mat)
    inv_err_right = matrix_diff_norm_inf(ainv_right, i_mat)

    print_matrix("L:", l)
    print()
    print_matrix("U:", u)
    print()
    print("Permutation p (rows):", p)
    print()
    print_vector("Solution x:", x)
    print()
    print(f"det(A) = {det_a:.6f}")
    print()
    print_matrix("A^(-1):", inv_a)
    print()
    print_vector("Check A*x:", ax)
    print_vector("Right part b:", b)
    print(f"Residual ||Ax-b||_inf = {residual:.6e}")
    print(f"Relative residual = {rel_residual:.6e}")
    print()
    print(f"Inverse check ||A*A^(-1)-I||_inf = {inv_err_left:.6e}")
    print(f"Inverse check ||A^(-1)*A-I||_inf = {inv_err_right:.6e}")
    print()
    print(f"Check ||P*A - L*U||_inf = {lu_err:.6e}")
    print(f"Min |pivot| in U = {min_pivot:.6e}")
    if min_pivot < 1e-12:
        print("Warning: very small pivot -> matrix may be singular or ill-conditioned.")


if __name__ == "__main__":
    main()

# Небольшой пример (3x3):
#
# A = [
#   [0, 2, 1],
#   [2, 2, 3],
#   [4, 1, 8],
# ]
#
# В самом начале:
# U = A
# L = I
# p = [0, 1, 2]
# swaps = 0
#
# Шаг k = 0:
# 1) Смотрим 0-й столбец. Самый большой по модулю элемент — 4 (строка 2).
#    Меняем строки 0 и 2.
#    p -> [2, 1, 0], swaps -> 1
# 2) Зануляем элементы под главным:
#    - для строки 1: factor = 2/4 = 0.5, пишем L[1][0] = 0.5
#    - для строки 2: factor = 0/4 = 0.0, пишем L[2][0] = 0.0
#
# Шаг k = 1:
# 1) Теперь смотрим 1-й столбец (начиная со строки 1) и снова выбираем лучший pivot.
#    Если он не в строке 1 — снова делаем swap в U и p.
# 2) Важный момент: нужно также поменять уже заполненную часть L
#    (столбцы 0..k-1) для этих строк.
# 3) Зануляем U[2][1] и сохраняем коэффициент в L[2][1].
#
# В итоге:
# - U становится верхнетреугольной
# - L хранит все коэффициенты зануления (на диагонали единицы)
# - p хранит, как мы переставляли строки

#
# 1) Как из L, U, p получить решение x (функция solve_lu)
# Пример 2x2:
# L = [
#   [1,   0],
#   [0.5, 1],
# ]
# U = [
#   [4, 1],
#   [0, 2.5],
# ]
# p = [1, 0]
# b = [6, 9]
#
# Что делаем:
# - сначала переставляем правую часть по p:
#   pb = [b[1], b[0]] = [9, 6]
# - решаем Lz = pb (прямой ход):
#   z0 = 9
#   z1 = 6 - 0.5 * 9 = 1.5
# - решаем Ux = z (обратный ход):
#   x1 = 1.5 / 2.5 = 0.6
#   x0 = (9 - 1 * 0.6) / 4 = 2.1
# В итоге x = [2.1, 0.6]
#
# 2) Как считается det(A) (функция determinant_from_u)
# Формула:
# det(A) = (-1)^swaps * (u00 * u11 * ... * unn)
# То есть:
# - перемножаем диагональ U
# - если swap-ов нечетное число -> меняем знак
#
# Для примера выше:
# diag(U) = [4, 2.5], swaps = 1
# det(A) = - (4 * 2.5) = -10
#
# 3) Как строится обратная матрица (функция inverse_from_lu)
# Идея простая:
# - берем e0 = [1, 0, 0, ...], e1 = [0, 1, 0, ...], и т.д.
# - для каждого ei решаем A * xi = ei через тот же solve_lu
# - каждый найденный xi — это очередной столбец A^(-1)
# Для 2x2 делаем 2 раза, для 4x4 — 4 раза.
#
# 4) Проверка решения (функция mat_vec_mul)
# После того как нашли x, считаем A*x.
# Если A*x совпадает с b (или очень близко из-за округления),
# значит решили систему правильно.
