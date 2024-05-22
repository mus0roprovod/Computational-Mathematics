import numpy as np
import random

#  II. Работа с вырожденными матрицами
row = 4
col = 4
k = 0
random_int_matrix = (np.random.randint(-10, 10, size=(row, col))) / 1.0  # матрица А

# a) Нахождение ранга матрицы
#  0. Сгенерировать вырожденные матрицы путем клонирования рандомного числа строчек
random_integer_np = np.random.randint(2, row + 1)
for i in range(random_integer_np):
    random_int_matrix[i, :] = random_int_matrix[0, :]

#1. Вставляю ЛУ алгоритм из первого пункта
m_u_l = random_int_matrix.copy()  # матрица ul с результатом
m_row = np.eye(row)  # матрица отвечающая за строчки
m_col = np.eye(row)  # матрица отвечающая за столбцы
diag_cur_row = 0
diag_cur_col = 0
zero = False

while k != row - 1:
    # 1.1 найти в матрице максимальный эдемент
    sub_m_u_l = m_u_l[diag_cur_row:, diag_cur_col:]
    kof_row, kof_col = np.unravel_index(np.argmax(np.abs(sub_m_u_l)), sub_m_u_l.shape)  # find max in sub mul
    kof_row += k
    kof_col += k
    # 1.2 перенести его на рассматриваемой части матрицы
    if (kof_row != diag_cur_row) or (kof_col != diag_cur_col):  # надо переместить ведущий на диагональ
        m_u_l[[kof_row, diag_cur_row], :] = m_u_l[[diag_cur_row, kof_row], :]
        m_row[[kof_row, diag_cur_row], :] = m_row[[diag_cur_row, kof_row], :]
        m_u_l[:, [kof_col, diag_cur_col]] = m_u_l[:, [diag_cur_col, kof_col]]
        m_col[:, [kof_col, diag_cur_col]] = m_col[:, [diag_cur_col, kof_col]]

    # 1.3 Метод Гауса: делим элеметры всего столбца на элемент ведущий - вездесущий
    if m_u_l[diag_cur_row, diag_cur_col] != 0:
        m_u_l[diag_cur_row + 1:, diag_cur_col] = m_u_l[diag_cur_row + 1:, diag_cur_col] / m_u_l[
            diag_cur_row, diag_cur_col]
    else: # !!! Главное отличие здесь, когда если у нас элемент равен нулю, то мы просто обнуляем всю строчку
        m_u_l[diag_cur_row + 1:, diag_cur_col] = 0

    # 1.3.1 в каждой строчке умножить на элемент, который находится на певрой позиции
    kof_row_elements = m_u_l[diag_cur_row, diag_cur_row + 1:]
    kof_col_elements = m_u_l[diag_cur_row + 1:, diag_cur_col].reshape((row - k - 1, 1))
    m_u_l[diag_cur_row + 1:, diag_cur_col + 1:] -= (kof_row_elements * kof_col_elements)
    diag_cur_row += 1
    diag_cur_col += 1
    k += 1

U =  np.triu(m_u_l)
L = np.tril(m_u_l)
np.fill_diagonal(L, 1)
np.fill_diagonal(L, 1)
PAQ = np.matmul(m_row, random_int_matrix)
PAQ = np.matmul(PAQ, m_col)
LU = np.matmul(L, U)

#  2. Посчитаем нулевые строчки в матрице
num_zero_rows = 0
for i in U:
    if np.all(i == 0):
        num_zero_rows += 1
# !!! TODO там где ноль ноль и справа


#  3. Тогда ранг матрицы - это строчки - нулевые строчки
my_rank = row - num_zero_rows
# print(np.linalg.matrix_rank(random_int_matrix, tol=None))
# print()
# print(my_rank)

# б) Проверка системы на совместимость и если все окей, то ее частное решение

# 1. Нам необходимо избавиться от зависимых строчек в A, поэтому применим метод Гаусса
A = np.copy(random_int_matrix)
for i in range(row):
    if A[i, i] == 0:
        for k in range(i + 1, row):
            if A[k, i] != 0:
                A[[i, k]] = A[[k, i]]
                break
        else:
            continue

    for j in range(i + 1, row):
        koef = A[j, i] / A[i, i]
        A[j, i:] -= koef * A[i, i:]

# 2. После метода Гаусса просто обрезаем первые строчки соотв. ранку и сгенерим x и получим б
A = A[:my_rank]
x_real = (np.random.randint(1, 5, size=((np.shape(A))[1], 1))) / 1.0
b = np.matmul(A, x_real)


#  3. Проверим, что матрица А имеет такой же ранк, что и расширенная матрица Ab
Ab = np.hstack([A, b.reshape(-1, 1)])
rank_Ab = np.linalg.matrix_rank(Ab)
rows, cols = Ab.shape
print(Ab)
# 2. Если ранки совпали, то тогда система совместна и можно найти решение
if my_rank == rank_Ab:
    # 3. Гаус-жордан
    for i in range(min(rows, cols - 1)):
        Ab[i, :-1] = Ab[i, :-1] / Ab[i, i]
        for j in range(i - 1, -1, -1):
            koef = Ab[j, i]
            Ab[j, :-1] -= koef * Ab[i, :-1]
            Ab[j, -1] -= koef * Ab[i, -1]

    a = Ab[:, :-1]
    b = Ab[:, -1]
    b = b.reshape(-1, 1)
    d = row - my_rank
# 4. Теперь осталось буквально вставить частное и общее решение
    print('Общее решение:')
    for i in range(my_rank):
        print(f'x_{i + 1} = ', end='')
        for j in range(d):
            print(f'({a[i][col - d + j]:.4f}) * x_{col - d + j + 1} + ', end='')
        print(f'({b[i][0]:.4f})')
    print('Частное решение:')
    for i in range(my_rank):
        print(f'x_{i + 1} = ', end='')
        print(f'{b[i][0]:.4f}')
    print()
