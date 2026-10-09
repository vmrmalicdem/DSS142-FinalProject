# Tier 2 cells for 02_abt_to_models.ipynb. Each "# %% [cell]" block is one new code cell.
# Where to insert:  T2-A, T2-B, T2-C  -> right after the "INC" table cell (index ~51), BEFORE the negative-control cell.
#                   T2-D              -> right after the existing Q2 results/figure cells (needs HOUR_Z, seg helpers, holm, block_idx).
# Nothing here has been executed. Fixed seeds are used everywhere, so cell order does not change results.

# %% [cell] T2-A: do held-out blocks fall outside the training range of each sensor?
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import make_pipeline

SENS = GROUPS_Q3["indoor_outdoor"]
rows = []
for k, (tr, te) in enumerate(OUTER):
    for s in SENS:
        a = abt[s].to_numpy(float); lo, hi = a[tr].min(), a[tr].max()
        rows.append(dict(fold=k, sensor=s,
                         share_outside=float(((a[te] < lo) | (a[te] > hi)).mean()),
                         shift_in_train_sd=(a[te].mean() - a[tr].mean()) / a[tr].std()))
RANGE = pd.DataFrame(rows)
P1 = RANGE.pivot(index="sensor", columns="fold", values="share_outside").loc[SENS]
P2 = RANGE.pivot(index="sensor", columns="fold", values="shift_in_train_sd").loc[SENS]

fig, ax = plt.subplots(1, 2, figsize=(11, 0.28 * len(SENS) + 2), sharey=True)
im0 = ax[0].imshow(P1.values, aspect="auto", cmap="Reds", vmin=0, vmax=max(0.5, P1.values.max()))
ax[0].set_title("Share of test block outside\nthe training range"); plt.colorbar(im0, ax=ax[0])
lim = np.abs(P2.values).max()
im1 = ax[1].imshow(P2.values, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
ax[1].set_title("Test-block mean minus train mean\n(in training SDs)"); plt.colorbar(im1, ax=ax[1])
for a_ in ax:
    a_.set_xticks(range(N_OUTER)); a_.set_xlabel("outer test block")
ax[0].set_yticks(range(len(SENS))); ax[0].set_yticklabels(SENS)
plt.tight_layout(); plt.show()
print("Mean share of test rows outside the training range, by block:\n", P1.mean().round(3).to_string())
print("Mean |shift| in training SDs, by block:\n", P2.abs().mean().round(2).to_string())

# %% [cell] T2-B: baselines (hour-mean, ridge) and de-seasonalised sensors, same outer blocks and inner CV as Q3
abtd = abt.copy(); DES = []
for s in SENS:                                           # deviation from the sensor's own trailing 1-day mean (uses only the past)
    abtd[s + "_dev"] = abt[s] - abt[s].rolling(144, min_periods=1).mean(); DES.append(s + "_dev")

def eval_group(name, cols, kind, df=abtd):
    out = []
    for k, (tr, te) in enumerate(OUTER):
        if kind == "hour_mean":
            mu = pd.Series(Y_LOG[tr]).groupby(df["hour"].to_numpy()[tr]).mean()
            pred = df["hour"].iloc[te].map(mu).to_numpy()
        else:
            cv = [(np.searchsorted(tr, a), np.searchsorted(tr, b)) for a, b in blocks(tr, N_INNER, BUFFER)]
            if kind == "ridge":
                cat = [c for c in cols if c in CAT]; num = [c for c in cols if c not in CAT]
                pipe = Pipeline([("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                                                             ("num", StandardScaler(), num)])), ("model", Ridge())])
                grid = {"model__alpha": [1, 10, 100, 1000, 10000]}
            else:
                pipe = make_pipe("GBM", cols); grid = MODELS["GBM"][1]
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="r2").fit(df.iloc[tr][cols], Y_LOG[tr])
            pred = gs.predict(df.iloc[te][cols])
        out.append(dict(name=name, fold=k, r2=r2_score(Y_LOG[te], pred)))
    return out

SPECS = [("hour mean (typical value for that hour)", None, "hour_mean"),
         ("ridge: temporal", CAT, "ridge"),
         ("ridge: raw sensors only", SENS, "ridge"),
         ("ridge: temporal + raw sensors", CAT + SENS, "ridge"),
         ("ridge: temporal + de-seasonalised sensors", CAT + DES, "ridge"),
         ("GBM: temporal + de-seasonalised sensors", CAT + DES, "GBM")]
BASE = pd.DataFrame([r for nm, cols, kd in SPECS for r in eval_group(nm, cols, kd)])
display(BASE.groupby("name", sort=False).r2.agg(mean="mean", min="min", max="max"))
print("For reference, the main Q3 run (mean out-of-sample R2 by model and group):")
display(Q3[Q3.run == "main"].groupby(["model", "group"]).r2.mean().unstack().round(3))

# %% [cell] T2-C: does environment add anything beyond hour + weekday?  Fit time first, then model its leftover errors with sensors.
def resid_stage(sensor_cols, kind, df=abtd):
    rows = []; S = df[sensor_cols].to_numpy(float)
    for k, (tr, te) in enumerate(OUTER):
        enc = OneHotEncoder(handle_unknown="ignore").fit(df.iloc[tr][CAT])
        lin = LinearRegression().fit(enc.transform(df.iloc[tr][CAT]), Y_LOG[tr])
        yhat_tr = lin.predict(enc.transform(df.iloc[tr][CAT])); yhat_te = lin.predict(enc.transform(df.iloc[te][CAT]))
        r_tr, r_te = Y_LOG[tr] - yhat_tr, Y_LOG[te] - yhat_te
        est = (make_pipeline(StandardScaler(), Ridge(alpha=1000)) if kind == "ridge" else
               HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, learning_rate=0.05, random_state=SEED))
        rh = est.fit(S[tr], r_tr).predict(S[te])
        rows.append(dict(sensors="raw" if sensor_cols is SENS else "de-seasonalised", model=kind, fold=k,
                         r2_time_only=r2_score(Y_LOG[te], yhat_te), r2_time_plus_env=r2_score(Y_LOG[te], yhat_te + rh),
                         gain_on_leftover=1 - ((r_te - rh) ** 2).sum() / (r_te ** 2).sum()))   # >0: sensors explain part of what time of day missed
    return rows
RES = pd.DataFrame([r for cols in (SENS, DES) for kd in ("ridge", "GBM") for r in resid_stage(cols, kd)])
G = RES.groupby(["sensors", "model"]).agg(time_only=("r2_time_only", "mean"), time_plus_env=("r2_time_plus_env", "mean"),
                                           mean_gain=("gain_on_leftover", "mean"), min_gain=("gain_on_leftover", "min"),
                                           folds_positive=("gain_on_leftover", lambda s: int((s > 0).sum())))
display(G.round(4))
print(f"Finding: sensors improve on hour+weekday in {int((RES.gain_on_leftover > 0).sum())} of {len(RES)} (fold x model x sensor-set) fits.")

# %% [cell] T2-D: Q2 v2 - kink beyond a straight line, hour-of-day control inside the model, wider bootstrap, CI width, edge share, detectable effect
N_BOOT_V2 = 5000      # replicates; each is generated in chunks so memory stays small. Expect a few minutes.

def _F(R, Ht, hh, nn, k):
    q = (R @ Ht.T) ** 2 / hh; rr = (R * R).sum(1)[:, None]
    return q / ((rr - q) / (nn - k - 1))

def seg_test_v2(x, y, direction, seed_key, Zhour):
    nn = len(x); mu, sd = x.mean(), x.std(); xs = (x - mu) / sd
    grid = np.quantile(x, np.linspace(GRID_LO, GRID_HI, GRID_N)); ts = (grid - mu) / sd
    p5, p95 = np.percentile(xs, [5, 95]); dist = (p95 - ts) if direction == "upper" else (ts - p5)
    H = hinge(xs, ts, direction)
    X0 = np.c_[np.ones(nn), xs, Zhour]; Q = np.linalg.qr(X0)[0]; k = X0.shape[1]      # null: straight line + hour of day
    r = y - Q @ (Q.T @ y); Ht = H - (H @ Q) @ Q.T; hh = (Ht * Ht).sum(1)              # residualise on the controls (FWL)
    beta, sse = fit_seg(xs, r, Ht, np.ones((1, nn))); g, b, slope, pct = pick(beta, sse, dist, direction)
    Fo = _F(r[None, :], Ht, hh, nn, k)[0]
    rng_v = np.random.default_rng(seed_key); sup, taus, edge = [], [], []
    for s in range(0, N_BOOT_V2, CHUNK):
        B = min(CHUNK, N_BOOT_V2 - s)
        ip = block_idx(nn, BLOCK, B, rng_v); W = np.zeros((B, nn))
        for i in range(B): W[i] = np.bincount(ip[i], minlength=nn)
        _, ss = fit_seg(xs, r, Ht, W); gb = ss.argmin(1)                                # breakpoint CI: resample (r, Ht) blocks
        taus.append(grid[gb]); edge.append((gb == 0) | (gb == GRID_N - 1))
        Es = r[block_idx(nn, BLOCK, B, rng_v)]; Es = Es - (Es @ Q) @ Q.T                # null: resampled residuals, search repeated
        sup.append(_F(Es, Ht, hh, nn, k).max(1))
    sup = np.concatenate(sup); taus = np.concatenate(taus); edge = np.concatenate(edge)
    out = dict(tau_hat=grid[g[0]], interior=bool(0 < g[0] < GRID_N - 1), kink_slope_per_sd=slope[0], pct_beyond_trend=pct[0],
               supF=Fo.max(), p_raw=(1 + (sup >= Fo.max()).sum()) / (len(sup) + 1),
               tau_lo=np.percentile(taus, 2.5), tau_hi=np.percentile(taus, 97.5), edge_share=edge.mean(),
               ci_width_frac=(np.percentile(taus, 97.5) - np.percentile(taus, 2.5)) / (grid[-1] - grid[0]))
    return out, sup, (r, Ht, hh, nn, k, dist)

def q2_run_v2(y_all, tag, variables=Q_VARS):
    t0 = time.perf_counter(); rows, SUP, CTX = [], {}, {}
    for vi, v in enumerate(variables):
        x = abt[v].to_numpy(float)
        for di, d in enumerate(("upper", "lower")):
            o, sup, ctx = seg_test_v2(x, y_all, d, [SEED, vi, di], HOUR_Z)
            rows.append(dict(variable=v, direction=d, run=tag, **o)); SUP[(v, d)] = sup; CTX[(v, d)] = ctx
    D = pd.DataFrame(rows); nt = len(D); D["p_holm"] = holm(D.p_raw)
    g0 = GRID_N // 2; mde = []                                                         # detectable kink at the median breakpoint
    for (v, d) in zip(D.variable, D.direction):
        crit = np.quantile(SUP[(v, d)], 1 - ALPHA / nt); r, Ht, hh, nn, k, dist = CTX[(v, d)]
        mde.append(next((p for p in range(1, 101) if _F((r + np.log1p(p / 100) / dist[g0] * Ht[g0])[None, :], Ht, hh, nn, k).max() >= crit), np.nan))
    D["detectable_pct_at_extreme"] = mde
    D["c1_holm"] = D.p_holm < ALPHA; D["c2_interior"] = D.interior; D["c3_slope_pos"] = D.kink_slope_per_sd > 0
    D["c4_pct10"] = D.pct_beyond_trend >= 10
    D["supported"] = D.c1_holm & D.c2_interior & D.c3_slope_pos & D.c4_pct10
    D["stable_interval"] = (D.ci_width_frac <= 0.5) & (D.edge_share <= 0.2)
    print(f"[{tag}] {nt} tests, {N_BOOT_V2} replicates, {time.perf_counter() - t0:.0f} s | min attainable Holm p = {nt / (N_BOOT_V2 + 1):.4f}")
    return D

Q2V2 = q2_run_v2(Y_LOG, "main_v2")
display(Q2V2[["variable", "direction", "tau_hat", "tau_lo", "tau_hi", "ci_width_frac", "edge_share", "pct_beyond_trend",
              "p_raw", "p_holm", "supported", "stable_interval", "detectable_pct_at_extreme"]].round(4))
ns = int(Q2V2.supported.sum())
print(f"Supported thresholds (kink beyond straight line, hour controlled, Holm p<0.05, interior, +slope, >=10%): {ns} of {len(Q2V2)}")
print(f"Median detectable effect: a kink that changes appliance energy by ~{Q2V2.detectable_pct_at_extreme.median():.0f}% "
      f"(range {Q2V2.detectable_pct_at_extreme.min():.0f}-{Q2V2.detectable_pct_at_extreme.max():.0f}%) between a mid-range breakpoint and the P95/P5 extreme "
      "would just have been flagged at the Holm level. Smaller kinks than that cannot be ruled out.")
