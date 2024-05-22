from IQF import *
a = 1.7
b = 3.2
alpha = 0
beta = 1/4
max_derivative = 129.429
rating_above = 3.9901
real_ans = 11.83933565874812191864851199716726555747# 2

def p(x):
    return ((x - a) ** (-alpha)) * ((b - x) ** (-beta))

def f(x):
    return 3 * np.cos(0.5 * x) * np.exp(x / 4) + 5 * np.sin(2.5 * x) * np.exp( - x / 3) + 2 * x

def GQF(a, b):
    # 1. Надо посчитать моменты

    m_0 = (-4/3 * (16/5 - b)**(3/4)) - (-4/3 * (16/5 - a)**(3/4))
    m_1 = (-4/105 * (16/5 - b)**(3/4) * (64 + 15*b)) - (-4/105 * (16/5 - a)**(3/4) * (64 + 15*a))
    m_2 = (-(4*(16/5-b)**(3/4)*(8192 + 1920*b + 525*b**2))/5775)  - (-(4*(16/5-a)**(3/4)*(8192 + 1920*a + 525*a**2))/5775)
    m_3 = (-(4*(16/5-b)**(3/4) * (524288 + 122880*b + 33600*b**2 + 9625*b**3))/144375) - (-(4*(16/5 - a)**(3/4) * (524288 + 122880 * a + 33600*a**2 + 9625*a**3))/144375)
    m_4 = (-(4*(16/5-b)**(3/4) * (134217728 + 31457280*b + 8601600*b**2 + 2464000*b**3 + 721875*b**4))/13715625) - (-(4*(16/5-a)**(3/4) * (134217728 + 31457280 * a + 8601600 *a **2 + 2464000*a**3 + 721875*a**4))/13715625)
    m_5 =  (-(4*(16/5-b)**(3/4) * (8589934592 + 2013265920*b + 550502400*b**2 + 157696000*b**3 + 46200000*b**4 + 13715625*b**5))/315459375) - ((-(4*(16/5-a)**(3/4) * (8589934592 + 2013265920*a + 550502400*a**2 + 157696000*a**3 + 46200000*a**4 + 13715625*a**5))/315459375))
    mu = - 1 * np.array([m_3, m_4, m_5])

    #2. Решим системку
    M = np.array([[
        m_0, m_1, m_2
    ], [
        m_1, m_2, m_3
    ], [
        m_2, m_3, m_4
    ]])
    a = np.flip(np.linalg.solve(M,  mu))

    # 3. Решаем уравнение
    w = np.array([1])
    w = np.append(w, a)
    x = np.roots(w)

    # 4. Решаем другую систему
    powers = np.arange(3)[:, None]
    a = x ** powers
    mu = np.array([m_0, m_1, m_2])
    A = np.linalg.solve(a,  mu)
    A_1, A_2, A_3 = A
    x_1, x_2, x_3 = x

    return A_1 * f(x_1) + A_2 * f(x_2) + A_3 * f(x_3)



def SQF(a, b, flag = 1, k1 = 2):
    S = np.array([]) # массивчик с значениями интегралов
    H = np.array([]) # массивчик с шагами
    C = 0
    counter = 0
    k = k1
    while True:
        counter += 1
        #0. Разбиваем а б на k кусочков и в каждом кусочке считаем икф
        size_of_step =(b - a) / k
        sum  = 0
        for i in range(k):
            segment_start = a + i * size_of_step
            segment_end = segment_start + size_of_step
            sum += IQF(segment_start, segment_end) if flag == 1 else GQF(segment_start, segment_end)

        #1. Все полученные интегральчики будем складывать в массивчик
        S = np.append(S, sum)
        H = np.append(H, size_of_step)

        r = len(S) - 1

        # 2. Посчитать методом Эээээээ м-ку

        if r >= 2:
            m = -np.log(abs(S[r] - S[r - 1]) / abs(S[r-1] - S[r-2])) / np.log(2)

        else:
            if flag == 0:
                m = 6
            else:
                m = 3

        # 3. Методом Ричердсона посчитаем цешки
        A = -1 * np.ones((r + 1, r + 1))
        for i in range(r):
            A[:, i] = H ** (i + m)
        C = np.linalg.solve(A, -S)

        # 4. Вычислим ошибки
        error = abs(real_ans - S[r]) if r == 0 else abs(C[r] - S[r])
        print(k, ": ", m, error, abs(real_ans - S[r]) )
        if error < 1e-8:
            break
        k *= 2
    return S, H, counter

def opt_k0(S, H):
    # 1. Посчитать ЭМ
    m = -np.log(abs(S[2] - S[1]) / abs(S[1] - S[0])) / np.log(2)
    print(m)
    # 2. Воспользуемся формулой 32
    RH_2 =  (S[1] - S[0]) / ((np.power(2, m)) - 1)

    # 3. Воскользоваться формулой 34
    H_opt = (H[1] * (1e-8 / np.abs(RH_2)) ** (1 / m)) * 0.95
    # 4. Посчитать k_opt
    return int(np.ceil((b - a) / H_opt))