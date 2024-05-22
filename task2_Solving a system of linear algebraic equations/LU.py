import numpy as np
from numpy import linalg

# I. LU - разложения
row = 3
col = 3
k = 0
random_int_matrix = (np.random.randint(-100, 100, size=(row, col))) / 1.0  # матрица А
m_u_l = random_int_matrix.copy()  # матрица ul с результатом
m_row = np.eye(row)  # матрица отвечающая за строчки
m_col = np.eye(row)  # матрица отвечающая за столбцы
diag_cur_row = 0
diag_cur_col = 0
zero = False

while k != row - 1:
    # 1. найти в матрице максимальный эдемент
    sub_m_u_l = m_u_l[diag_cur_row:, diag_cur_col:]
    kof_row, kof_col = np.unravel_index(np.argmax(np.abs(sub_m_u_l)), sub_m_u_l.shape)  # find max in sub mul
    kof_row += k
    kof_col += k
    # 2. перенести его на рассматриваемой части матрицы
    if (kof_row != diag_cur_row) or (kof_col != diag_cur_col):  # надо переместить ведущий на диагональ
        m_u_l[[kof_row, diag_cur_row], :] = m_u_l[[diag_cur_row, kof_row], :]
        m_row[[kof_row, diag_cur_row], :] = m_row[[diag_cur_row, kof_row], :]
        m_u_l[:, [kof_col, diag_cur_col]] = m_u_l[:, [diag_cur_col, kof_col]]
        m_col[:, [kof_col, diag_cur_col]] = m_col[:, [diag_cur_col, kof_col]]

    # 3. Метод Гауса: делим элеметры всего столбца на элемент ведущий - вездесущий
    if m_u_l[diag_cur_row, diag_cur_col] != 0:
        m_u_l[diag_cur_row + 1:, diag_cur_col] = m_u_l[diag_cur_row + 1:, diag_cur_col] / m_u_l[
            diag_cur_row, diag_cur_col]
    else:
        print("Деление на ноль")
        zero = True
        break
    # 3.1 в каждой строчке умножить на элемент, который находится на певрой позиции

    kof_row_elements = m_u_l[diag_cur_row, diag_cur_row + 1:]
    kof_col_elements = m_u_l[diag_cur_row + 1:, diag_cur_col].reshape((row - k - 1, 1))

    if (np.any(kof_row_elements == 0)) or (np.any(kof_col_elements == 0)):
        print("Деление на ноль")
        zero = True
        break
    m_u_l[diag_cur_row + 1:, diag_cur_col + 1:] -= (kof_row_elements * kof_col_elements)

    diag_cur_row += 1
    diag_cur_col += 1
    k += 1


if not zero:
    U = np.triu(m_u_l)
    L = np.tril(m_u_l)
    np.fill_diagonal(L, 1)
    PAQ = np.matmul(m_row, random_int_matrix)
    PAQ = np.matmul(PAQ, m_col)
    LU = np.matmul(linalg.inv(m_row), L)
    LU = np.matmul(LU, U)
    LU = np.matmul(LU, linalg.inv(m_col))
    ans = np.abs(random_int_matrix - LU)

    print(random_int_matrix)
    print(LU)
    print()
    print(ans)

    # a) Определитель матрицы А
    det_A = np.prod(np.diag(U)) * np.linalg.det(m_col) * np.linalg.det(m_row)
    real_det_A = np.linalg.det(random_int_matrix)
    print(det_A)
    print(real_det_A)


    # б) Решение СЛАУ
    # 1. Сгенерировать столбец x и сгенировать бешки по этим иксам
    x_real = (np.random.randint(-100, 100, size=(row, 1))) / 1.0
    b = np.matmul(random_int_matrix, x_real)

    # TODO матрица А
    # 2. Найти реешние Ly = b, где у = Ux
    b_slae = np.matmul(m_row, b) #!!!
    y_solv = np.zeros_like(b)
    for i in range(len(b)):
        y_solv[i] = b_slae[i] - np.dot(L[i, :i], y_solv[:i])

    # 3. Найти Ux = y, выразить иксы = решение
    x_solv = np.zeros_like(y_solv)

    for i in range(len(y_solv) - 1, -1, -1):   # идем с конца, реверсом до 0 элемента
        x_solv[i] = (y_solv[i] - np.dot(U[i, i + 1:], x_solv[i + 1:])) / U[i, i]

    x_solv = np.matmul(m_col, x_solv)

    # print(x_real)
    # print(x_solv)

    # в) Нахождение обратной матрицы
    # сделать все тоже самое что и в прошлом пункте, но вместо бешки теперь единичная матрица

    my_reverse = np.zeros_like(random_int_matrix)

    for i in range(row):
        e = np.zeros(row)
        e[i] = 1

        y_rev = np.zeros_like(e)
        for j in range(row):
            y_rev[j] = (e[j] - np.dot(L[j, :j], y_rev[:j])) / L[j, j]

        x_rev = np.zeros_like(y_rev)
        for j in range(row - 1, -1, -1):
            x_rev[j] = (y_rev[j] - np.dot(U[j, j + 1:], x_rev[j + 1:])) / U[j, j]

        my_reverse[:, i] = x_rev

    # print(np.linalg.inv(PAQ))
    # print()
    # print(my_reverse)

    # г) Число обусловленности

    # 0. Нужно найти нормы, для этого буду использовать бесконечную норму

    norm_A = 0
    # 1. Нужно пройтись по строчкам в матрице PAQ
    for row in PAQ:
        # 2. Найти максимальную сумму по абс в строчке это и будет норма
        row_sum = np.sum(np.abs(row))
        if row_sum > norm_A:
            norm_A = row_sum
    # 3. Аналогично
    norm_rev_A = 0

    for row in my_reverse:
        row_sum = np.sum(np.abs(row))
        if row_sum > norm_rev_A:
            norm_rev_A = row_sum

    # 4. Число обсуловленности - их произведение
    num_condition = norm_rev_A * norm_A

    # print(num_condition)
    # print()
    # print(np.linalg.cond(PAQ, p=np.inf))