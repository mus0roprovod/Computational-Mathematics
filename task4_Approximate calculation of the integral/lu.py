import numpy
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

    L = numpy.tril(a)
    U = numpy.triu(a)
    for i in range(rows):
        L[i, i] = 1
    return L, U, p, q
