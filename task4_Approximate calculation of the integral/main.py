import numpy as np
from SQF_GQF import *

a = 1.7
b = 3.2
alpha = 0
beta = 1/4
max_derivative = 129.429
rating_above = 3.9902
real_ans = 11.83933565874812191864851199716726555747 # 2

# 1. Нютона-Котса
# a) построить ИКФ с весовой ф. p(x) = (x - a) ** -alpha * (b - x) ** - beta
# 0. Точки - начало отрезка, середина и конец
# 1. Вычисляем момент в этих точках
# 2. Высчитываем матрицу А, которая содержит степени точек х_1 - х_3
# 3. Потом там чет решаем и все

print("`•.,¸,.•*¯`•.,¸,.•*`•.,¸,.•*¯`•.,¸,.•*IQF*¯`•.,¸,.•*`•.,¸,.•*¯`•.,¸,.•*`•.,¸,.•*¯`•.,¸,.•*")
ans_IQF = IQF(a, b)
print(f"Получили решение: {ans_IQF}")
print(f"Методическая погрешность: {rating_above}")
print(f"На деле же: {real_ans - ans_IQF}")
print()

print("`•.,¸,.•*¯`•.,¸,.•*`•.,¸,.•*¯`•.,¸,.•*SQF + IQF*¯`•.,¸,.•*`•.,¸,.•*¯`•.,¸,.•*`•.,¸,.•*¯`•.,¸,.•*")
# Б) Теперь считаем ИКФ на каждом шаге и суммируют
S, H, counter = SQF(a, b)
ans_SQF = S[-1]
print(f"Получили решение : {ans_SQF} за вот столько {counter} итераций")
print(f"Наша погрешность: {np.abs(real_ans - ans_SQF)}")
print()

print("`•.,¸,.•*¯`•.,¸,.•*`•.,¸,.•*¯`•.,¸,.•*SQF + OPT_K*¯`•.,¸,.•*`•.,¸,.•*¯`•.,¸,.•*`•.,¸,.•*¯`•.,¸,.•*")
# В) Теперь посчитаем k оптимально
k_opt = opt_k0(S, H)
print("Надо шагов опт ", k_opt)
S, H, counter = SQF(a, b, 1, k_opt)
ans_SQF_K = S[-1]
print(f"Получили решение : {ans_SQF_K} за вот столько {counter} итераций")
print(f"Наша погрешность: {np.abs(real_ans - ans_SQF_K)}")
print()

print("`•.,¸,.•*¯`•.,¸,.•*`•.,¸,.•*¯`•.,¸,.•*GQF*¯`•.,¸,.•*`•.,¸,.•*¯`•.,¸,.•*`•.,¸,.•*¯`•.,¸,.•*")
# # Г) А что там по гауссу?!
ans_GQF = GQF(a, b)
print(f"Получили решение : {ans_GQF}")
print(f"Наша погрешность: {np.abs(real_ans - ans_GQF)}")
print()

print("`•.,¸,.•*¯`•.,¸,.•*`•.,¸,.•*¯`•.,¸,.•*SQF + GQF*¯`•.,¸,.•*`•.,¸,.•*¯`•.,¸,.•*`•.,¸,.•*¯`•.,¸,.•*")
S, H1, counter = SQF(a, b, 0)
ans_SQF_GQF = S[-1]
print(f"Получили решение : {ans_SQF_GQF} за вот столько итераций {counter}")
print(f"Наша погрешность: {np.abs(real_ans - ans_SQF_GQF)}")
print()

print("`•.,¸,.•*¯`•.,¸,.•*`•.,¸,.•*¯`•.,¸,.•*SQF + GQF + K_OPT*¯`•.,¸,.•*`•.,¸,.•*¯`•.,¸,.•*`•.,¸,.•*¯`•.,¸,.•*")
S, H2, counter = SQF(a, b, 0, k_opt)
ans_SQF_GQF_K = S[-1]
print(f"Получили решение : {ans_SQF_GQF_K} за вот столько итераций {counter}")
print(f"Наша погрешность: {np.abs(real_ans - ans_SQF_GQF_K)}")




# print(SQF(a, b, 2, 0))
# print(SQF(a, b, k_opt, 0 ))










