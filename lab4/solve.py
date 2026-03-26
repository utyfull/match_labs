import math

def identity_matrix(n):
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        m[i][i] = 1.0
    return m


def transpose(a):
    n = len(a)
    m = len(a[0])
    t = [[0.0] * n for _ in range(m)]
    for i in range(n):
        for j in range(m):
            t[j][i] = a[i][j]
    return t


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


def offdiag_norm(a):
    n = len(a)
    s = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            s += a[i][j] * a[i][j]
    return math.sqrt(s)


def max_offdiag_index(a):
    n = len(a)
    p, q = 0, 1
    mx = abs(a[p][q])
    for i in range(n):
        for j in range(i + 1, n):
            cur = abs(a[i][j])
            if cur > mx:
                mx = cur
                p, q = i, j
    return p, q


def jacobi_rotation_method(a, eps, max_iter=10000):
    n = len(a)
    ak = [row[:] for row in a]
    v = identity_matrix(n)

    history = [offdiag_norm(ak)]
    rotations = []

    k = 0
    while history[-1] > eps and k < max_iter:
        p, q = max_offdiag_index(ak)
        phi = 0.5 * math.atan2(2.0 * ak[p][q], ak[p][p] - ak[q][q])
        c = math.cos(phi)
        s = math.sin(phi)

        u = identity_matrix(n)
        u[p][p] = c
        u[q][q] = c
        u[p][q] = -s
        u[q][p] = s

        ak = mat_mul(transpose(u), mat_mul(ak, u))
        v = mat_mul(v, u)

        rotations.append((p, q, phi))
        history.append(offdiag_norm(ak))
        k += 1

    eigenvalues = [ak[i][i] for i in range(n)]
    return eigenvalues, v, ak, k, history, rotations


def get_column(m, j):
    return [m[i][j] for i in range(len(m))]


def mat_vec_mul(a, x):
    n = len(a)
    m = len(a[0])
    y = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(m):
            s += a[i][j] * x[j]
        y[i] = s
    return y


def vec_sub(x, y):
    return [x[i] - y[i] for i in range(len(x))]


def vec_norm2(x):
    s = 0.0
    for v in x:
        s += v * v
    return math.sqrt(s)


def vec_dot(x, y):
    s = 0.0
    for i in range(len(x)):
        s += x[i] * y[i]
    return s


def print_vector(name, v, digits=6):
    print(name)
    print("[" + ", ".join(f"{x:.{digits}f}" for x in v) + "]")


def print_matrix(name, m, digits=6):
    print(name)
    for row in m:
        print("[" + "  ".join(f"{x:.{digits}f}" for x in row) + "]")


def main():
    # [ 8   0  -2]
    # [ 0   5   4]
    # [-2   4  -6]
    a = [
        [8.0, 0.0, -2.0],
        [0.0, 5.0, 4.0],
        [-2.0, 4.0, -6.0],
    ]
    eps = 1e-5

    eigenvalues, eigenvectors, a_final, k, history, rotations = jacobi_rotation_method(a, eps)

    print(f"eps = {eps}")
    print(f"iterations = {k}")
    print()

    print_matrix("Final near-diagonal matrix A^(k):", a_final)
    print()
    print_vector("Eigenvalues (diag A^(k)):", eigenvalues)
    print()
    print_matrix("Eigenvectors matrix V (columns are eigenvectors):", eigenvectors)
    print()

    print("Error history t(A^(k)) by iterations:")
    for i in range(len(history)):
        print(f"k={i:2d}: t={history[i]:.6e}")
    print()

    print("Rotation pairs (p, q, phi) by iterations:")
    for i, (p, q, phi) in enumerate(rotations, start=1):
        print(f"k={i:2d}: p={p}, q={q}, phi={phi:.6f}")
    print()

    print("Eigenpair residuals ||A*v - lambda*v||_2:")
    for j in range(len(eigenvalues)):
        vj = get_column(eigenvectors, j)
        av = mat_vec_mul(a, vj)
        lv = [eigenvalues[j] * x for x in vj]
        r = vec_sub(av, lv)
        print(f"lambda_{j+1}: {vec_norm2(r):.6e}")
    print()

    print("Orthogonality check (dot products of different eigenvectors):")
    n = len(eigenvalues)
    for i in range(n):
        for j in range(i + 1, n):
            vi = get_column(eigenvectors, i)
            vj = get_column(eigenvectors, j)
            print(f"(v{i+1}, v{j+1}) = {vec_dot(vi, vj):.6e}")


if __name__ == "__main__":
    main()

#
# 1) Что хотим:
#    взять симметричную матрицу A и "докрутить" ее до почти диагональной.
#    Диагональ в конце = собственные значения.
#    Накопленная матрица поворотов V = собственные векторы (по столбцам).
#
# 2) Один шаг итерации:
#    - ищем самый большой по модулю внедиагональный элемент a[p][q];
#    - считаем угол:
#      phi = 0.5 * atan2(2*a[p][q], a[p][p] - a[q][q]);
#    - строим поворот U в плоскости (p, q);
#    - обновляем A: A <- U^T * A * U;
#    - обновляем V: V <- V * U.
#
# 3) Как меряем текущую "ошибку недиагональности":
#    t(A) = sqrt(sum(a[i][j]^2, i<j)).
#    Чем ближе t(A) к нулю, тем более диагональная матрица.
#    Это и есть метрика зависимости погрешности от числа итераций.
#
# 4) Для нашей матрицы:
#    A = [[8, 0, -2], [0, 5, 4], [-2, 4, -6]]
#    начальная t(A^(0)) = sqrt(0^2 + (-2)^2 + 4^2) = sqrt(20) ~= 4.4721
#    дальше на каждой итерации t(A^(k)) падает:
#    k=0: 4.4721
#    k=1: 2.0000
#    k=2: 0.6185
#    k=3: 0.0752
#    k=4: 0.0210
#    k=5: 0.0001
#    k=6: 0.000000146
#    ... и так до eps.
#
# 5) Почему работает:
#    каждый поворот убивает один "крупный" внедиагональный элемент
#    и в целом делает A более диагональной.
#    Для симметричной матрицы метод стабильно сходится.
#
# 6) Что смотреть в выводе:
#    - iterations: сколько шагов понадобилось;
#    - history t(A^(k)): видно скорость сходимости;
#    - residual ||A*v - lambda*v||: проверка качества найденной пары;
#    - dot(v_i, v_j): проверка ортогональности векторов.
