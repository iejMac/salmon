import numpy as np
import pulp as plp


# https://stackoverflow.com/a/279586
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


# https://or.stackexchange.com/a/1174
@static_vars(var_id=0)
def min2_lp(lp, a, b, M):
    X = plp.LpVariable(f'min2_X_{min2_lp.var_id}')
    y = plp.LpVariable(f'min2_y_{min2_lp.var_id}', cat=plp.LpBinary)
    min2_lp.var_id += 1

    lp += b - a <= M * y
    lp += a - b <= M * (1 - y)
    lp += X <= a
    lp += X <= b
    lp += X >= a - M * (1 - y)
    lp += X >= b - M * y

    return X


def min_lp(lp, *args, M):
    if len(args) == 1:
        return args[0]

    X = min2_lp(lp, args[0], args[1], M)

    for a in args[2:]:
        X = min2_lp(lp, X, a, M)

    return X


# https://or.stackexchange.com/a/712
@static_vars(var_id=0)
def max2_lp(lp, a, b, M):
    X = plp.LpVariable(f'max2_X_{max2_lp.var_id}')
    y = plp.LpVariable(f'max2_y_{max2_lp.var_id}', cat=plp.LpBinary)
    max2_lp.var_id += 1

    lp += a - b <= M * y
    lp += b - a <= M * (1 - y)
    lp += X >= a
    lp += X >= b
    lp += X <= a + M * (1 - y)
    lp += X <= b + M * y

    return X


def max_lp(lp, *args, M):
    if len(args) == 1:
        return args[0]

    X = max2_lp(lp, args[0], args[1], M)

    for a in args[2:]:
        X = max2_lp(lp, X, a, M)

    return X


def find_c_adam(a, b, alpha, u, omega, M=10):
    solver = plp.PULP_CBC_CMD(msg=False)
    assert len(a) == len(b) == len(alpha) == len(u) == len(omega)
    n = len(a)

    lp = plp.LpProblem('min_c', plp.LpMinimize)
    c = plp.LpVariable.dicts('c', range(n))
    r = plp.LpVariable.dicts('r', range(n))
    lp.c = c
    lp.r = r

    # stability at initialization
    lp += a[0] + b[0] == 0
    for i in range(1, n - 1):
        lp += a[i] + b[i] == 0.5
    lp += a[n - 1] + b[n - 1] >= 0.5

    # stable activations during training
    lp += r[0] == a[0] + c[0]
    lp += r[0] >= 0

    for i in range(1, n - 1):
        x1 = plp.LpVariable(f'min_x1_{i}')
        x2 = plp.LpVariable(f'min_x2_{i}')
        x3 = plp.LpVariable(f'min_x3_{i}')

        lp += x1 == a[i] + c[i] - alpha[i]
        lp += x2 == a[i] + c[i] + r[i - 1] - u[i]
        lp += x3 == 0.5 + r[i - 1] - omega[i]
        lp += r[i] == min_lp(lp, x1, x2, x3, M=M)
        lp += r[i] >= 0

    # stable logits during training
    x1 = plp.LpVariable(f'min_x1_{n - 1}')
    x2 = plp.LpVariable(f'min_x2_{n - 1}')
    x3 = plp.LpVariable(f'min_x3_{n - 1}')

    lp += x1 == a[n - 1] + b[n - 1] + r[n - 2] - omega[n - 1]
    lp += x2 == a[n - 1] + c[n - 1] - alpha[n - 1]
    lp += x3 == a[n - 1] + c[n - 1] + r[n - 2] - u[n - 1]
    lp += r[n - 1] == min_lp(lp, x1, x2, x3, M=M)
    lp += r[n - 1] >= 0

    # feature learning
    # lp += r[n - 2] == 0

    lp += plp.lpSum(c)
    lp.solve(solver)

    if lp.status != plp.LpStatusOptimal:
        return n * [np.nan], n * [np.nan]

    return [lp.c[i].varValue for i in range(n)], [lp.r[i].varValue for i in range(n)]


def find_c_sgd(a, b, alpha, u, omega, M=10):
    solver = plp.PULP_CBC_CMD(msg=False)
    assert len(a) == len(b) == len(alpha) == len(u) == len(omega)
    n = len(a)

    lp = plp.LpProblem('min_c', plp.LpMinimize)
    c = plp.LpVariable.dicts('c', range(n))
    r = plp.LpVariable.dicts('r', range(n))
    g = plp.LpVariable.dicts('g', range(n - 1))
    lp.c = c
    lp.r = r

    # stability at initialization
    lp += a[0] + b[0] == 0
    for i in range(1, n - 1):
        lp += a[i] + b[i] == 0.5
    lp += a[n - 1] + b[n - 1] >= 0.5

    # stable activations during training
    lp += r[0] == g[0] + a[0] + c[0]
    lp += r[0] >= 0

    for i in range(n - 1):
        # NOTE: gradient scale (potential) mistake
        # lp += g[i] == max_lp(lp, a[n - 1] + b[n - 1], 2 * a[n - 1] + c[n - 1], M=M) + a[i]
        lp += g[i] == min_lp(lp, a[n - 1] + b[n - 1], 2 * a[n - 1] + c[n - 1], M=M) + a[i]

    for i in range(1, n - 1):
        x1 = plp.LpVariable(f'min_x1_{i}')
        x2 = plp.LpVariable(f'min_x2_{i}')
        x3 = plp.LpVariable(f'min_x3_{i}')

        lp += x1 == g[i] + a[i] + c[i] - alpha[i]
        lp += x2 == g[i] + a[i] + c[i] + r[i - 1] - u[i]
        lp += x3 == 0.5 + r[i - 1] - omega[i]
        lp += r[i] == min_lp(lp, x1, x2, x3, M=M)
        lp += r[i] >= 0

    # stable logits during training
    x1 = plp.LpVariable(f'min_x1_{n - 1}')
    x2 = plp.LpVariable(f'min_x2_{n - 1}')
    x3 = plp.LpVariable(f'min_x3_{n - 1}')

    lp += x1 == a[n - 1] + b[n - 1] + r[n - 2] - omega[n - 1]
    lp += x2 == 2 * a[n - 1] + c[n - 1] - alpha[n - 1]
    lp += x3 == 2 * a[n - 1] + c[n - 1] + r[n - 2] - u[n - 1]
    lp += r[n - 1] == min_lp(lp, x1, x2, x3, M=M)
    lp += r[n - 1] >= 0

    # feature learning
    # lp += r[n - 2] == 0

    lp += plp.lpSum(c)
    lp.solve(solver)

    if lp.status != plp.LpStatusOptimal:
        return n * [np.nan], n * [np.nan]

    return [lp.c[i].varValue for i in range(n)], [lp.r[i].varValue for i in range(n)]