p = [0, 0]
F = { 0.3: 1.1, 1.6: 1.4, 1.9: 2.0, 2.5: 2.1 }
f = lambda x, p: p[0] * x + p[1]

def grad(x, p):
    return [ 2*(f(x, p) - F[x])*x,
             2*(f(x, p) - F[x]) ]

def train(p):
    s = [0, 0]
    for i in F:
        x = i
        dps = grad(x, p)
        s[0] += dps[0]/len(F)
        s[1] += dps[1]/len(F)
    return s

def update(p, a=0.1):
    dp = train(p)
    for i in range(len(p)):
        p[i] -= a*dp[i]

for i in range(100_000):
    update(p)

for i in F:
    x, y = i, F[i]
    print(f'f({x}) ~ {f(x, p):0f} ({y})')
print(f'f(x)   ~ {p[0]:0f}x + {p[1]:0f}')
