import numpy as np
import pandas as pd


def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x))

def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))

# ---------- numeric generator ----------
def simulate_numeric(
    m=1000,
    n=15,
    dists=None,
    base_rate=0.5,
    strength=7.5,
    seed=None
):
    rng = np.random.default_rng(seed)
    if dists is None:
        means  = np.linspace(-0.5, 0.5, n)
        sigmas = np.linspace(0.6, 1.8, n)
        dists  = list(map(tuple, np.c_[means, sigmas]))

    mus  = np.array([mu for mu, s in dists], dtype=float)
    sigs = np.array([s  for mu, s in dists], dtype=float)

    X = rng.normal(loc=mus, scale=sigs, size=(m, n))
    Z = (X - mus) / sigs
    score = strength * Z.mean(axis=1)
    p = sigmoid(logit(base_rate) + score)
    y = rng.binomial(n=1, p=p)

    df = pd.DataFrame({f"X{i+1}": X[:, i] for i in range(n)})
    df["y"] = y
    return df

# ---------- categorical generator ----------
def simulate_categorical(
    m=1000,
    n=15,
    k_min=2,
    k_max=5,
    base_rate=0.5,
    strength=7.5,
    seed=None,
    imbalance=False,
    dirichlet_alpha=1.0
):
    rng = np.random.default_rng(seed)
    Ks = rng.integers(k_min, k_max + 1, size=n)

    # Per-feature level effects: mean 0, std 1 (equal per-feature impact)
    effects = []
    for Kj in Ks:
        w = rng.normal(0.0, 1.0, size=Kj)
        w = w - w.mean()
        std = w.std()
        effects.append(w if std == 0 else w / std)

    # Sample categories per feature
    Xcats = np.empty((m, n), dtype=int)
    for j, Kj in enumerate(Ks):
        if imbalance:
            probs = rng.dirichlet(np.full(Kj, dirichlet_alpha, dtype=float))
        else:
            probs = np.full(Kj, 1.0 / Kj, dtype=float)
        Xcats[:, j] = rng.choice(Kj, size=m, p=probs)

    # Score from chosen levels
    contribs = np.empty((m, n), dtype=float)
    for j in range(n):
        contribs[:, j] = effects[j][Xcats[:, j]]
    score = strength * contribs.mean(axis=1)

    p = sigmoid(logit(base_rate) + score)
    y = rng.binomial(n=1, p=p)

    # DataFrame with categorical dtypes
    data = {}
    for j, Kj in enumerate(Ks):
        s = pd.Series(Xcats[:, j], name=f"X{j+1}", dtype="category")
        s = s.astype(pd.CategoricalDtype(categories=list(range(Kj))))
        data[f"X{j+1}"] = s
    df = pd.DataFrame(data)
    df["y"] = y
    return df

# ---------- mixed generator: 8 numeric + 7 categorical (total 15) ----------
def simulate_mixed(
    m=1000,
    num_features=8,
    cat_features=7,
    num_dists=None,
    k_min=2,
    k_max=5,
    base_rate=0.5,
    strength=7.5,
    seed=None,
    cat_imbalance=False,
    dirichlet_alpha=1.0
):
    assert num_features >= 0 and cat_features >= 0
    n_total = num_features + cat_features
    rng = np.random.default_rng(seed)

    # --- numeric part ---
    if num_features > 0:
        if num_dists is None:
            means  = np.linspace(-0.5, 0.5, num_features)
            sigmas = np.linspace(0.6, 1.8, num_features)
            num_dists = list(map(tuple, np.c_[means, sigmas]))
        mus  = np.array([mu for mu, s in num_dists], dtype=float)
        sigs = np.array([s  for mu, s in num_dists], dtype=float)
        X_num = rng.normal(loc=mus, scale=sigs, size=(m, num_features))
        Z_num = (X_num - mus) / sigs
        contrib_num = Z_num
    else:
        X_num = np.empty((m, 0))
        contrib_num = np.empty((m, 0))

    # --- categorical part ---
    if cat_features > 0:
        Ks = rng.integers(k_min, k_max + 1, size=cat_features)
        effects = []
        for Kj in Ks:
            w = rng.normal(0.0, 1.0, size=Kj)
            w = w - w.mean()
            std = w.std()
            effects.append(w if std == 0 else w / std)
        Xcats = np.empty((m, cat_features), dtype=int)
        for j, Kj in enumerate(Ks):
            if cat_imbalance:
                probs = rng.dirichlet(np.full(Kj, dirichlet_alpha, dtype=float))
            else:
                probs = np.full(Kj, 1.0 / Kj, dtype=float)
            Xcats[:, j] = rng.choice(Kj, size=m, p=probs)
        contrib_cat = np.empty((m, cat_features), dtype=float)
        for j in range(cat_features):
            contrib_cat[:, j] = effects[j][Xcats[:, j]]
    else:
        Ks = np.array([], dtype=int)
        Xcats = np.empty((m, 0), dtype=int)
        contrib_cat = np.empty((m, 0))

    all_contribs = np.concatenate([contrib_num, contrib_cat], axis=1)
    score = strength * all_contribs.mean(axis=1)
    p = sigmoid(logit(base_rate) + score)
    y = rng.binomial(n=1, p=p)


    data = {}

    for i in range(num_features):
        data[f"X{i+1}"] = X_num[:, i]

    for j in range(cat_features):
        col_idx = num_features + j + 1
        s = pd.Series(Xcats[:, j], name=f"X{col_idx}", dtype="category")
        s = s.astype(pd.CategoricalDtype(categories=list(range(Ks[j]))))
        data[f"X{col_idx}"] = s

    df = pd.DataFrame(data)
    df["y"] = y
    return df


if __name__ == "__main__":

    
    data_size = 500
    
    if data_size == 1000:
        
        
        seed_numeric = 1001
        seed_categ   = 2002
        seed_mixed   = 3003



        df_num = simulate_numeric(m=1000, n=16, base_rate=0.5, strength=25, seed=seed_numeric)
        df_num.to_csv("s_1000_num.csv", index=False)


        df_cat = simulate_categorical(m=1000, n=16, k_min=2, k_max=5,
                                      base_rate=0.5, strength=25,
                                      seed=seed_categ, imbalance=False)
        df_cat.to_csv("s_1000_cat.csv", index=False)


        df_mix = simulate_mixed(m=1000, num_features=8, cat_features=8,
                                k_min=2, k_max=5, base_rate=0.5, strength=25,
                                seed=seed_mixed, cat_imbalance=False)
        df_mix.to_csv("s_1000_mix.csv", index=False)
        
    
    elif data_size == 500:
        
        seed_numeric = 4004
        seed_categ   = 5005
        seed_mixed   = 6006


        df_num = simulate_numeric(m=500, n=15, base_rate=0.5, strength=25, seed=seed_numeric)
        df_num.to_csv("s_500_num.csv", index=False)


        df_cat = simulate_categorical(m=500, n=15, k_min=2, k_max=5,
                                      base_rate=0.5, strength=25,
                                      seed=seed_categ, imbalance=False)
        df_cat.to_csv("s_500_cat.csv", index=False)


        df_mix = simulate_mixed(m=500, num_features=8, cat_features=7,
                                k_min=2, k_max=5, base_rate=0.5, strength=25,
                                seed=seed_mixed, cat_imbalance=False)
        df_mix.to_csv("s_500_mix.csv", index=False)        

