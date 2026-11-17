"""Offline check of p7 teacher dumps: alignment, softness of soft shift targets, rel-KD scale."""
import pickle, sys, numpy as np
data = pickle.load(open('/DATA/rashmi_2521cs34/data/IEMOCAP.pkl', 'rb'), encoding='latin1')
labels, train = data[2], list(data[10])
sen_of = lambda e: {1: 0, 3: 0, 5: 0, 2: 1, 0: 2, 4: 2}[e]
held, fit = train[:12], train[12:]
T = [pickle.load(open(p, 'rb')) for p in sys.argv[1:]]
sm = lambda z, t: (lambda x: x / x.sum(-1, keepdims=True))(np.exp((z - z.max(-1, keepdims=True)) / t))
t0 = T[0]
for name, vs in (('fitted 108', fit), ('held-out 12', held)):
    y = np.concatenate([labels[v] for v in vs]); ys = np.array([sen_of(e) for e in y])
    pe = np.concatenate([t0[v]['emo'].argmax(-1) for v in vs]); ps = np.concatenate([t0[v]['sen'].argmax(-1) for v in vs])
    print(f'{name}: emo acc {100*(pe==y).mean():.1f}  sen acc {100*(ps==ys).mean():.1f}')
for tau in (1, 2, 4):
    same, diff, maxp = [], [], []
    for v in fit:
        P = sm(t0[v]['sen'].astype(np.float64), tau); ys = np.array([sen_of(e) for e in labels[v]])
        maxp.append(P.max(-1))
        for a in range(0, len(ys), 19):
            Q, s = P[a:a+19], ys[a:a+19]
            D = 1 - Q @ Q.T; m = ~np.eye(len(s), dtype=bool); eq = s[:, None] == s[None, :]
            same.append(D[m & eq]); diff.append(D[m & ~eq])
    print(f'tau {tau}: mean max-prob {np.concatenate(maxp).mean():.3f}; differ target on same-sent pairs {np.concatenate(same).mean():.3f}, on diff-sent pairs {np.concatenate(diff).mean():.3f}')
def cosm(H):
    H = H.astype(np.float64); H = H / np.linalg.norm(H, axis=1, keepdims=True); return H @ H.T
off = [cosm(t0[v]['H'])[~np.eye(len(labels[v]), dtype=bool)].mean() for v in fit]
print(f'teacher H mean off-diagonal cosine {np.mean(off):.3f}')
if len(T) > 1:
    mse = [((cosm(T[0][v]['H']) - cosm(T[1][v]['H'])) ** 2).mean() for v in fit]
    print(f'rel-KD MSE between two independent teachers (seed scale): {np.mean(mse):.2e}')
