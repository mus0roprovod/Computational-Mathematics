import math

c_f = 1e-6 / 3.0

def my_artan(x):
    c_u = 1e-6 / 0.42
    argv = x #сумма ряда
    n = x #числитель учитывая знак ()
    x_2 = -(x * x) #мы всегда домножаем на квадрат
    k = 1 # 2k + 1

    while True:
        n *= x_2
        k += 2
        cur = n / k #текущий шаг
        argv += cur
        if abs(cur) < c_u:
            return argv

def my_sqrt(x):
    c_w = 1e-6 / 1.2
    prev = 1.0

    while True:
        argv = 0.5 * (prev + x/prev)

        if abs(argv - prev) < c_w:
            return argv

        prev = argv


def my_e(x):
    c_v = 2 * (1e-6 / 0.18)
    argv = 1
    k = 1
    cur = 1

    while True:
        cur *= x / k
        argv += cur
        k += 1

        if abs(cur) < c_v:
            return argv


def my_ans(x):
    return my_sqrt(1 + my_artan(0.8 * x + 0.2)) / my_e(2 * x + 1)


def real_ans(x):
    return math.sqrt(1 + math.atan(0.8 * x + 0.2)) / math.exp(2 * x + 1)


start = int(0.1 * 100)
end = int(0.2 * 100)
step = int(0.01 * 100)

for x in range(start, end + 1, step):
    x /= 100
    per_ans = real_ans(x)
    aprx_ans = my_ans(x)
    error = abs(per_ans - aprx_ans)

    print(f'x = {x} \t z(x) = {round(per_ans ,10)} \t\t z`(x) = {round(aprx_ans, 10)}\t\t {error}')

