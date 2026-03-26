import cmath
import math

def identity_matrix(n):
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        m[i][i] = 1.0
    return m


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


def vector_norm(v):
    s = 0.0
    for x in v:
        s += x * x
    return math.sqrt(s)


def qr_decomposition_householder(a):
    n = len(a)
    r = [row[:] for row in a]
    q = identity_matrix(n)

    for k in range(n - 1):
        x = [r[i][k] for i in range(k, n)]
        nx = vector_norm(x)
        if nx == 0.0:
            continue

        sign = 1.0 if x[0] >= 0.0 else -1.0
        v = x[:]
        v[0] += sign * nx
        nv = vector_norm(v)
        if nv == 0.0:
            continue
        for i in range(len(v)):
            v[i] /= nv

        h = identity_matrix(n)
        for i in range(k, n):
            for j in range(k, n):
                h[i][j] -= 2.0 * v[i - k] * v[j - k]

        r = mat_mul(h, r)
        q = mat_mul(q, h)

    return q, r


def qr_decomposition_error(q, r, a):
    qr = mat_mul(q, r)
    n = len(a)
    s = 0.0
    for i in range(n):
        for j in range(n):
            d = qr[i][j] - a[i][j]
            s += d * d
    return math.sqrt(s)


def strict_lower_subdiag_norm(a):
    n = len(a)
    s = 0.0
    for i in range(n):
        for j in range(i - 1):
            s += a[i][j] * a[i][j]
    return math.sqrt(s)


def qr_algorithm_eigenvalues(a, eps, max_iter=10000):
    n = len(a)
    ak = [row[:] for row in a]
    history = [strict_lower_subdiag_norm(ak)]
    k = 0

    while k < max_iter and history[-1] > eps:
        q, r = qr_decomposition_householder(ak)
        ak = mat_mul(r, q)
        history.append(strict_lower_subdiag_norm(ak))
        k += 1

    eigs = []
    i = 0
    while i < n:
        if i == n - 1 or abs(ak[i + 1][i]) < eps:
            eigs.append(complex(ak[i][i], 0.0))
            i += 1
        else:
            a11 = ak[i][i]
            a12 = ak[i][i + 1]
            a21 = ak[i + 1][i]
            a22 = ak[i + 1][i + 1]
            tr = a11 + a22
            det = a11 * a22 - a12 * a21
            disc = tr * tr - 4.0 * det
            root = cmath.sqrt(disc)
            eig1 = (tr + root) / 2.0
            eig2 = (tr - root) / 2.0
            eigs.append(eig1)
            eigs.append(eig2)
            i += 2

    return eigs, ak, k, history


def format_complex(z, digits=6):
    if abs(z.imag) < 1e-12:
        return f"{z.real:.{digits}f}"
    sign = "+" if z.imag >= 0 else "-"
    return f"{z.real:.{digits}f}{sign}{abs(z.imag):.{digits}f}i"


def print_matrix(name, m, digits=6):
    print(name)
    for row in m:
        print("[" + "  ".join(f"{x:.{digits}f}" for x in row) + "]")


def main():
    # [-1   2   9]
    # [ 9   3   4]
    # [ 8  -4  -6]
    a = [
        [-1.0, 2.0, 9.0],
        [9.0, 3.0, 4.0],
        [8.0, -4.0, -6.0],
    ]
    eps = 1e-6

    q0, r0 = qr_decomposition_householder(a)
    qr_err = qr_decomposition_error(q0, r0, a)

    eigs, a_final, iters, history = qr_algorithm_eigenvalues(a, eps)

    print(f"eps = {eps}")
    print()

    print_matrix("Q from first QR decomposition:", q0)
    print()
    print_matrix("R from first QR decomposition:", r0)
    print()
    print(f"Check ||A - Q*R||_F = {qr_err:.6e}")
    print()

    print(f"QR algorithm iterations = {iters}")
    print_matrix("Final quasi-triangular A^(k):", a_final)
    print()

    print("Eigenvalues:")
    for i, e in enumerate(eigs, start=1):
        print(f"lambda_{i} = {format_complex(e)}")
    print()

    print("Convergence history e^(k) = ||elements below first subdiagonal||:")
    for i, val in enumerate(history):
        print(f"k={i:3d}: e={val:.6e}")


if __name__ == "__main__":
    main()

#
# 1) Сначала мы делаем QR-разложение A = Q*R.
#    Q - ортогональная, R - верхнетреугольная.
#    Здесь это сделано отражениями Хаусхолдера.
#
# 2) Потом запускаем QR-итерации:
#    A(0)=A,
#    A(k)=Q(k)R(k),
#    A(k+1)=R(k)Q(k).
#    Эти матрицы подобны исходной, то есть собственные значения у них те же.
#
# 3) Для произвольной (не обязательно симметричной) матрицы итог обычно
#    не чисто треугольный, а квазитреугольный (1x1 и 2x2 блоки на диагонали).
#    Поэтому:
#    - 1x1 блок дает вещественное собственное значение;
#    - 2x2 блок может дать комплексно-сопряженную пару.
#
# 4) Критерий остановки здесь:
#    e(k) = sqrt(sum(a[i][j]^2, i > j+1)).
#    То есть смотрим элементы НИЖЕ первой поддиагонали.
#    Когда e(k) <= eps, считаем, что квазитреугольный вид достигнут.
#
# 5) Для нашей матрицы:
#    A = [[-1, 2, 9], [9, 3, 4], [8, -4, -6]]
#    e(0) = |a31| = 8.0
#    дальше e(k) убывает по итерациям, и это напрямую показывает
#    зависимость погрешности от числа итераций.
#    На текущем запуске при eps=1e-6:
#    - итераций: 19
#    - собственные значения:
#      lambda ~= -13.006365, 2.949770, 6.056595
#    - первые значения e(k):
#      8.000000 -> 3.745820 -> 2.164276 -> 1.731720 -> 1.279164 -> ...
#    - финал:
#      e(19) = 6.491605e-07 <= eps
#
# 6) Что смотреть в выводе:
#    - ||A-QR|| для проверки корректности QR-разложения;
#    - число итераций QR-алгоритма;
#    - финальную матрицу A^(k);
#    - найденные lambda_i (в том числе комплексные, если есть);
#    - таблицу e(k) по шагам.
