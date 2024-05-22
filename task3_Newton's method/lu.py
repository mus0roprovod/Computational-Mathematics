import numpy as np
from numpy import matlib

def meow(A):
    a = A.copy()
    rows, cols = a.shape
    p = matlib.identity(rows)
    q = matlib.identity(rows)

    for i in range(rows):
        row = i
        col = i
        c = a[i, i]
        for j in range(i, rows):
            for k in range(i, cols):
                if abs(a[j, k]) > abs(c):
                    c = a[j, k]
                    row = j
                    col = k
        if row != i or col != i:
            a[[i, row]] = a[[row, i]]
            p[[i, row]] = p[[row, i]]
            for j in range(rows):
                a[j, i], a[j, col] = a[j, col], a[j, i]
                q[j, i], q[j, col] = q[j, col], q[j, i]
        for j in range(i + 1, cols):
            if a[i, i]:
                a[j, i] /= a[i, i]
            else:
                a[j, i] = 0
            for k in range(i + 1, cols):
                a[j, k] -= a[i, k] * a[j, i]

    L = np.tril(a)
    U = np.triu(a)
    for i in range(rows):
        L[i, i] = 1
    return L, U, p, q

def woof(p, l, u, q, b):
    rows, cols = l.shape

    count_operations = 0
    b = p @ b
    count_operations += 190

    y = np.zeros((rows, 1), dtype=float)
    for i in range(rows):
        y[i, 0] = b[i] - sum([l[i, j] * y[j, 0] for j in range(i)])
        count_operations += 2 * i

    r = np.zeros((rows, 1), dtype=float)
    for i in reversed(range(rows)):
        r[i] = (y[i] - sum([r[k] * u[i, k] for k in range(i + 1, cols)])) / u[i, i]
        count_operations += 2 * (cols - i)

    r = q @ r

    return r, count_operations
