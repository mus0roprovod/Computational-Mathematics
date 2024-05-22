import numpy as np
from numpy.random import randint, choice
from numpy import tril, triu
row = 4
rel_tol = 1e-5

# 0. генерим нач. матрицу и икс
A = (np.random.randint(-10, 10, size=(row, row))) / 1.0
x = (np.random.randint(-10, 10, size=(row, 1))) / 1.0

# 1. Генерим матрицу именно с  большиим диагональным преобладанием, а то зейдель будет сходится тыщу лет
for i in range(row):
    A[i, i] = np.sum(np.abs(A[:, i])) * randint(2, 10) * choice([1, -1])
for i in range(row):
    row_sum_except_diagonal = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
    if np.abs(A[i, i]) <= row_sum_except_diagonal:
        print('не робит')
        exit()

# 2. считаем бешку
b = A.dot(x)

# а) Метод якоби
# 0. Оценка метода Якоби
D = np.diag(np.diag(A))  # диагональная матрица
R = A - D  # закинули все остальное

D_inv = np.linalg.inv(D)
B_jacobi = -D_inv.dot(R)
g_jacobi = D_inv.dot(b)

# Оценка количества итераций для методом простых итерация см википедию
norm_B_jacobi = np.linalg.norm(B_jacobi, ord=np.inf) # норма альфа
g_norm = np.linalg.norm(g_jacobi, ord=np.inf) # норма бета
est = (norm_B_jacobi / (1 - norm_B_jacobi)) * g_norm # справа штука
k_jacobi = 1 # ищем кашку
while est >= rel_tol:
    est *= norm_B_jacobi #повышаем степень к
    k_jacobi += 1

real_count = 0
# Вычисление решения с использованием метода Якоби
x_jacobi = np.zeros_like(b) #начальное знаечение равно нулю
for _ in range(k_jacobi):
    real_count += 1
    x_jacobi = g_jacobi + B_jacobi.dot(x_jacobi)
    if np.linalg.norm(A.dot(x_jacobi) - b, ord=np.inf) <= rel_tol:
        break

# Метод Зейделя
L = tril(A, -1) #нижний
U = triu(A, 1) #верхний
L_plus_D_inv = np.linalg.inv(D + L)
# все точно такое же просто д и эл как в
B_seidel = -L_plus_D_inv.dot(U)
g_seidel = L_plus_D_inv.dot(b)

# Оценка количества итераций для метода Зейделя
norm_B_seidel = np.linalg.norm(B_seidel, ord=np.inf)
est_seidel = (norm_B_seidel / (1 - norm_B_seidel)) * g_norm
k_seidel = 1
while est_seidel >= rel_tol:
    est_seidel *= norm_B_seidel
    k_seidel += 1

real_count1 = 0
# Вычисление решения с использованием метода Зейделя
x_seidel = np.zeros_like(b)
for _ in range(k_seidel):
    real_count1 += 1
    for i in range(row):
        sum1 = sum(A[i, j] * x_seidel[j] for j in range(i)) #то что слева
        sum2 = sum(A[i, j] * x_seidel[j] for j in range(i+1, row)) # то что справа
        x_seidel[i] = (b[i] - sum1 - sum2) / A[i, i] # мы берем б вычитаем то что справа, вычитаем то что слева и делим на лиагональ
    if np.linalg.norm(A.dot(x_seidel) - b, ord=np.inf) <= rel_tol:
        break

print(f"настоящие иксы {x}")
print('Якоби:')
print(f"априорная ошибка: {k_jacobi}, а оказалось {real_count}")
print()
print(x_jacobi)
print()
print('Зейделя:')
print(f"априорная ошибка: {k_seidel}, а оказалось {real_count1}")
print()
print(x_seidel)
