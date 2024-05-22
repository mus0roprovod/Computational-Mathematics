import numpy as np

# III QR - разложение
# а) Само разложение
#  0. Q до самого начала это просто единичная матрица, точно так же как и R
row = 4
col = 4

random_int_matrix = (np.random.randint(-100, 100, size=(row, col))) / 1.0  # матрица А
Q = np.identity(row)
R = random_int_matrix.copy()
diag_cur_row = 0
diag_cur_col = 0
k = 1

while k != row:
    #  1. Проходимся по столбцам, которые находятся под диагональю
    sub_m = R[diag_cur_row:, diag_cur_col].copy()
    #  2. Находим норму этого столбца
    norm_col = np.linalg.norm(sub_m, 2)
    #  3. Вычитаю норму этого столбца из первого элемента столбца
    sub_m[0] -= norm_col
    #  4. Нормирую полученный вектор
    norm_col = np.linalg.norm(sub_m)
    if np.any(norm_col != 0):
        sub_m /= norm_col
    #  5. Умножаю его на его самого ток транспорнированного и умножаю на два
    sub_m = np.dot(sub_m.reshape(-1, 1), sub_m.reshape(1, -1))
    #  6. Вычитаю из правого нижнего минора единичной матрицы
    e = np.eye(row)
    e[diag_cur_row:, diag_cur_col:] -= 2 * sub_m
    #  7. Умножаю Q справа на матрицу из 6 п. и слева на R
    R = e.dot(R)
    Q = Q.dot(e)
    diag_cur_col += 1
    diag_cur_row += 1
    k += 1


QR = Q.dot(R)
print(QR)
print()
print(random_int_matrix)
print(np.abs(random_int_matrix - QR))
# б) Решение СЛАУ
# 1. Сгенерировать столбец x и сгенировать бешки по этим иксам
x_real = (np.random.randint(-100, 100, size=(row, 1))) / 1.0
b = np.matmul(random_int_matrix, x_real)
# 2. Найти реешние Qy = b, где у = Rx как y = Q^-1*b
y_solv = np.dot(np.transpose(Q), b)
# 3. Найти решение Rx = y как в первом пункте
x_solv = np.zeros_like(y_solv)
for i in range(len(y_solv) - 1, -1, -1):  # идем с конца, реверсом до 0 элемента
    x_solv[i] = (y_solv[i] - np.dot(R[i, i + 1:], x_solv[i + 1:])) / R[i, i]

print(x_real)
print(x_solv)