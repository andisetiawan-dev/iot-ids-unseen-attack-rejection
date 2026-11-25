# %% [markdown]
# # Notebook C (v4): LOGO + Selective Classification + OOD, versi diperbaiki
#
# Perbaikan dari notebook B:
# 1. `Attack_label` dibuang dari fitur (bocor: 0 = Benign, 1 = semua serangan).
# 2. Baris duplikat dan kolom konstan dibuang.
# 3. Grup serangan mengikuti taksonomi paper Ferrag et al. (Password = Malware, Uploading = Injection).
# 4. Train / validation / test dipisah. Threshold dikalibrasi di validation data seen-class saja (TPR 95%), OOD tidak dipakai untuk memilih threshold.
# 5. Metrik OOD standar (AUROC, AUPR, FPR@95TPR), multi-seed, uji Wilcoxon.
# 6. Baseline OOD tambahan: Energy, Mahalanobis, kNN, Isolation Forest, plus temperature scaling.
# 7. LOGO tingkat grup (5 skenario) dan tingkat jenis serangan (14 skenario).

# %%
import os, glob, time, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import logsumexp, softmax
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score,
                             average_precision_score, confusion_matrix, classification_report)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')

QUICK = os.environ.get('QUICK') == '1'
OUT = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.environ.get('OUT', 'out')
os.makedirs(OUT, exist_ok=True)
SEEDS = [0] if QUICK else [0, 1, 2, 3, 4]
CW_SEEDS = SEEDS[:3]
TYPE_SEEDS = SEEDS[:3]
MODELS = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest']
if QUICK:
    MODELS = ['XGBoost', 'LightGBM']
TYPE_MODELS = ['XGBoost', 'LightGBM']
print('QUICK =', QUICK, '| seeds =', SEEDS)

# %% [markdown]
# ## Data

# %%
hits = glob.glob('/kaggle/input/**/edge_iiotset_clean.csv', recursive=True)
path = hits[0] if hits else os.environ['DATA']
df = pd.read_csv(path)
print(path, df.shape)

FERRAG = {
    'Normal': 'Benign',
    'DDoS_UDP': 'DoS_DDoS', 'DDoS_ICMP': 'DoS_DDoS', 'DDoS_HTTP': 'DoS_DDoS', 'DDoS_TCP': 'DoS_DDoS',
    'Port_Scanning': 'Information_Gathering', 'Vulnerability_scanner': 'Information_Gathering',
    'Fingerprinting': 'Information_Gathering',
    'MITM': 'MITM',
    'SQL_injection': 'Injection', 'XSS': 'Injection', 'Uploading': 'Injection',
    'Backdoor': 'Malware', 'Password': 'Malware', 'Ransomware': 'Malware',
}
df['group'] = df['Attack_type'].map(FERRAG)
assert df['group'].notna().all(), df.loc[df['group'].isna(), 'Attack_type'].unique()

NON_FEATURE = ['Attack_label', 'Attack_type', 'Attack_group', 'group']
feat_all = [c for c in df.columns if c not in NON_FEATURE]
n0 = len(df)
df = df.drop_duplicates(subset=feat_all + ['Attack_type']).reset_index(drop=True)
FEATS = [c for c in feat_all if df[c].nunique() > 1]
print(f'duplikat dibuang: {n0 - len(df)} | kolom konstan dibuang: {len(feat_all) - len(FEATS)} | fitur: {len(FEATS)}')

X = df[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e15, 1e15).values.astype(np.float32)
GROUPS = [g for g in df['group'].unique() if g != 'Benign']
TYPES = [t for t in df['Attack_type'].unique() if t != 'Normal']
print(df['group'].value_counts().to_dict())
print(df['Attack_type'].value_counts().to_dict())

# %% [markdown]
# ## Model dan skor OOD

# %%
def make(name, seed):
    if name == 'XGBoost':
        return xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric='mlogloss',
                                 random_state=seed, n_jobs=-1, verbosity=0)
    if name == 'LightGBM':
        # min_child_weight=1: default 1e-3 bikin LightGBM multiclass diverge pada 14 kelas (akurasi jatuh ke 0.1-0.5)
        return lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, min_child_weight=1,
                                  random_state=seed, n_jobs=-1, verbose=-1)
    if name == 'CatBoost':
        return CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1,
                                  random_seed=seed, verbose=0, thread_count=-1)
    return RandomForestClassifier(n_estimators=200, max_depth=12, random_state=seed, n_jobs=-1)


def raw_logits(name, m, A):
    if name == 'XGBoost':
        return m.predict(A, output_margin=True)
    if name == 'LightGBM':
        return m.predict(A, raw_score=True)
    if name == 'CatBoost':
        return m.predict(A, prediction_type='RawFormulaVal')
    return None


def fit_temperature(z, y):
    Ts = np.logspace(-0.5, 1, 40)
    nll = [-np.mean(np.log(softmax(z / T, axis=1)[np.arange(len(y)), y] + 1e-12)) for T in Ts]
    return Ts[int(np.argmin(nll))]


def entropy(p):
    return -(p * np.log(p + 1e-12)).sum(1)


def model_scores(name, m, sets, yva):
    """Skor OOD berbasis model. Makin tinggi = makin mungkin OOD. sets = [val, test, ood]."""
    probs = [m.predict_proba(s) for s in sets]
    out = {'MSP': [1 - p.max(1) for p in probs], 'Entropy': [entropy(p) for p in probs]}
    zs = [raw_logits(name, m, s) for s in sets]
    if zs[0] is not None:
        out['Energy'] = [-logsumexp(z, axis=1) for z in zs]
        T = fit_temperature(zs[0], yva)
        pT = [softmax(z / T, axis=1) for z in zs]
        out['MSP_T'] = [1 - p.max(1) for p in pT]
        out['Entropy_T'] = [entropy(p) for p in pT]
    return out, probs


def feature_scores(Ztr, ytr, sets, seed):
    """Skor OOD berbasis ruang fitur, tidak bergantung pada model klasifikasi."""
    out = {}
    classes = np.unique(ytr)
    mu = np.stack([Ztr[ytr == k].mean(0) for k in classes])
    Xc = Ztr - mu[np.searchsorted(classes, ytr)]
    P = np.linalg.inv(Xc.T @ Xc / len(Xc) + 1e-3 * np.eye(Ztr.shape[1]))

    def maha(Z):
        d = np.stack([np.einsum('ij,jk,ik->i', Z - m_, P, Z - m_) for m_ in mu])
        return d.min(0)
    out['Mahalanobis'] = [maha(Z) for Z in sets]

    sub = np.random.RandomState(seed).choice(len(Ztr), min(15000, len(Ztr)), replace=False)
    nn = NearestNeighbors(n_neighbors=5, algorithm='brute', n_jobs=-1).fit(Ztr[sub])
    out['kNN'] = [nn.kneighbors(Z)[0][:, -1] for Z in sets]

    iso = IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1).fit(Ztr)
    out['IsolationForest'] = [-iso.score_samples(Z) for Z in sets]
    return out


COVS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]


def metrics(sva, ste, so, correct_te):
    thr = np.quantile(sva, 0.95)          # threshold dari validation seen-class (TPR 95%)
    acc = ste <= thr
    y = np.r_[np.zeros(len(ste)), np.ones(len(so))]
    s = np.r_[ste, so]
    order = np.argsort(ste, kind='stable')
    cum = np.cumsum(correct_te[order]) / np.arange(1, len(order) + 1)
    row = {
        'coverage': acc.mean(),
        'sel_acc': correct_te[acc].mean() if acc.any() else np.nan,
        'ood_rej': (so > thr).mean(),
        'auroc': roc_auc_score(y, s),
        'aupr_ood': average_precision_score(y, s),
        'fpr95': (so <= np.quantile(ste, 0.95)).mean(),
        'aurc': 1 - cum.mean(),
    }
    for q in COVS:
        row[f'selacc_cov{int(q * 100)}'] = cum[int(np.ceil(q * len(cum))) - 1]
    return row


# %% [markdown]
# ## E0-E1: closed-world (cek kebocoran Attack_label, confusion matrix, F1 per kelas)

# %%
le_g = LabelEncoder().fit(df['group'])
y_g = le_g.transform(df['group'])
cw_rows, oof = [], np.zeros(len(df), dtype=int)

for name in MODELS:
    for seed in CW_SEEDS:
        f1s, accs = [], []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y_g):
            m = make(name, seed).fit(X[tr], y_g[tr])
            p = m.predict(X[te]).ravel().astype(int)
            f1s.append(f1_score(y_g[te], p, average='macro'))
            accs.append(accuracy_score(y_g[te], p))
            if name == 'XGBoost' and seed == CW_SEEDS[0]:
                oof[te] = p
        cw_rows.append({'model': name, 'seed': seed, 'f1_macro': np.mean(f1s), 'acc': np.mean(accs)})
    print(name, 'selesai')
cw = pd.DataFrame(cw_rows)
cw_sum = cw.groupby('model').agg(f1_mean=('f1_macro', 'mean'), f1_std=('f1_macro', 'std'),
                                 acc_mean=('acc', 'mean')).round(4)
cw_sum.to_csv(f'{OUT}/cw_summary.csv')
print(cw_sum)

# ablation: efek kebocoran Attack_label, dan setting 15-kelas
Xleak = np.c_[X, df['Attack_label'].values.astype(np.float32)]
leak_f1 = np.mean([f1_score(y_g[te], make('XGBoost', 0).fit(Xleak[tr], y_g[tr]).predict(Xleak[te]).ravel(), average='macro')
                   for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y_g)])
y15 = LabelEncoder().fit_transform(df['Attack_type'])
f15, a15 = [], []
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y15):
    p = make('XGBoost', 0).fit(X[tr], y15[tr]).predict(X[te]).ravel()
    f15.append(f1_score(y15[te], p, average='macro')); a15.append(accuracy_score(y15[te], p))
abl = pd.DataFrame([
    {'setting': '6 grup, tanpa Attack_label (benar)', 'f1_macro': cw_sum.loc['XGBoost', 'f1_mean']},
    {'setting': '6 grup, dengan Attack_label (bocor)', 'f1_macro': leak_f1},
    {'setting': '15 kelas (Attack_type), XGBoost', 'f1_macro': np.mean(f15)},
]).round(4)
abl.to_csv(f'{OUT}/cw_ablation.csv', index=False)
print(abl)

cm = confusion_matrix(y_g, oof)
pd.DataFrame(cm, index=le_g.classes_, columns=le_g.classes_).to_csv(f'{OUT}/cw_confusion.csv')
rep = pd.DataFrame(classification_report(y_g, oof, target_names=le_g.classes_, output_dict=True)).T.round(4)
rep.to_csv(f'{OUT}/cw_per_class.csv')
print(rep)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
cmn = cm / cm.sum(1, keepdims=True)
ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(len(le_g.classes_)), le_g.classes_, rotation=45, ha='right')
ax.set_yticks(range(len(le_g.classes_)), le_g.classes_)
for i in range(len(cmn)):
    for j in range(len(cmn)):
        ax.text(j, i, f'{cmn[i, j]:.2f}', ha='center', va='center', color='white' if cmn[i, j] > .5 else 'black', fontsize=8)
ax.set_title('Closed-world XGBoost: confusion matrix (dinormalisasi per baris)')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_cw_confusion.png', dpi=150); plt.close(fig)

# %% [markdown]
# ## E2-E6: LOGO (fungsi umum untuk tingkat grup dan tingkat jenis serangan)

# %%
def logo(col, helds, models, seeds, level):
    lab = df[col].values
    rows = []
    for held in helds:
        ood_idx = np.where(lab == held)[0]
        seen_idx = np.where(lab != held)[0]
        for seed in seeds:
            t0 = time.time()
            tr, rest = train_test_split(seen_idx, test_size=0.4, stratify=lab[seen_idx], random_state=seed)
            va, te = train_test_split(rest, test_size=0.5, stratify=lab[rest], random_state=seed)
            le = LabelEncoder().fit(lab[tr])
            ytr, yva, yte = le.transform(lab[tr]), le.transform(lab[va]), le.transform(lab[te])
            qt = QuantileTransformer(n_quantiles=500, output_distribution='normal',
                                     subsample=100000, random_state=seed).fit(X[tr])
            Ztr, Zva, Zte, Zo = [qt.transform(X[i]).astype(np.float64) for i in (tr, va, te, ood_idx)]
            fs = feature_scores(Ztr, ytr, [Zva, Zte, Zo], seed)
            sets = [X[va], X[te], X[ood_idx]]
            for name in models:
                m = make(name, seed).fit(X[tr], ytr)
                ms, probs = model_scores(name, m, sets, yva)
                pred = probs[1].argmax(1)
                correct = (pred == yte).astype(float)
                if correct.mean() < 0.8:
                    print(f'!! PERINGATAN: classifier {name} gagal ({held}, seed {seed}): acc={correct.mean():.3f}')
                base = {'level': level, 'held': held, 'model': name, 'seed': seed,
                        'f1_seen': f1_score(yte, pred, average='macro'), 'acc_seen': correct.mean(),
                        'conf_unseen': probs[2].max(1).mean() * 100}
                for meth, (sva, ste, so) in {**ms, **fs}.items():
                    rows.append({**base, 'method': meth, **metrics(sva, ste, so, correct)})
            print(f'[{level}] {held:24s} seed={seed} {time.time() - t0:5.1f}s')
    return pd.DataFrame(rows)


runs_g = logo('group', GROUPS, MODELS, SEEDS, 'group')
runs_g.to_csv(f'{OUT}/logo_group_runs.csv', index=False)

# %%
runs_t = logo('Attack_type', TYPES, TYPE_MODELS, TYPE_SEEDS, 'type')
runs_t.to_csv(f'{OUT}/logo_type_runs.csv', index=False)

# %% [markdown]
# ## E3-E5: ringkasan, uji signifikansi, kurva risk-coverage

# %%
MET = ['auroc', 'aupr_ood', 'fpr95', 'aurc', 'coverage', 'sel_acc', 'ood_rej']
METHODS = ['MSP', 'Entropy', 'Energy', 'MSP_T', 'Entropy_T', 'Mahalanobis', 'kNN', 'IsolationForest']


def summarize(runs, path):
    g = runs.groupby(['model', 'method'])[MET].agg(['mean', 'std'])
    g.columns = [f'{a}_{b}' for a, b in g.columns]
    g = g.round(4).reset_index()
    g.to_csv(path, index=False)
    return g


sum_g = summarize(runs_g, f'{OUT}/logo_group_summary.csv')
sum_t = summarize(runs_t, f'{OUT}/logo_type_summary.csv')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print('== LOGO grup (XGBoost) =='); print(sum_g[sum_g.model == 'XGBoost'].to_string(index=False))
print('== LOGO tipe (XGBoost) =='); print(sum_t[sum_t.model == 'XGBoost'].to_string(index=False))

per_held = (runs_g[runs_g.model == 'XGBoost'].groupby(['held', 'method'])[['auroc', 'ood_rej', 'sel_acc', 'coverage']]
            .mean().round(4).reset_index())
per_held.to_csv(f'{OUT}/logo_group_per_heldout_xgb.csv', index=False)

# uji Wilcoxon berpasangan pada AUROC (pasangan = held-out x seed), tiap metode vs MSP dan vs metode terbaik
def wilcox(runs, level):
    rows = []
    for model, d in runs.groupby('model'):
        piv = d.pivot_table(index=['held', 'seed'], columns='method', values='auroc')
        best = piv.mean().idxmax()
        for ref in ['MSP', best]:
            for meth in piv.columns:
                if meth == ref:
                    continue
                diff = (piv[meth] - piv[ref]).dropna()
                p = wilcoxon(diff).pvalue if (diff != 0).any() and len(diff) > 5 else np.nan
                rows.append({'level': level, 'model': model, 'method': meth, 'vs': ref,
                             'mean_auroc_diff': diff.mean(), 'n_pairs': len(diff), 'p_wilcoxon': p})
    return pd.DataFrame(rows).round(5)


stats = pd.concat([wilcox(runs_g, 'group'), wilcox(runs_t, 'type')])
stats.to_csv(f'{OUT}/stats_wilcoxon.csv', index=False)
print(stats[(stats.model == 'XGBoost')].to_string(index=False))

# gambar 1: heatmap AUROC per held-out (XGBoost)
piv = per_held.pivot(index='method', columns='held', values='auroc').reindex(METHODS)
fig, ax = plt.subplots(figsize=(8, 4.5))
im = ax.imshow(piv.values, cmap='viridis', vmin=0.5, vmax=1)
ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=30, ha='right'); ax.set_yticks(range(piv.shape[0]), piv.index)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        ax.text(j, i, f'{piv.values[i, j]:.2f}', ha='center', va='center', color='white', fontsize=8)
fig.colorbar(im, ax=ax, label='AUROC'); ax.set_title('LOGO grup, XGBoost: AUROC deteksi unseen (rata-rata seed)')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_auroc_heatmap_group.png', dpi=150); plt.close(fig)

# gambar 2: AUROC per metode, per model
fig, ax = plt.subplots(figsize=(9, 4))
w = 0.8 / runs_g.model.nunique()
for k, (model, d) in enumerate(runs_g.groupby('model')):
    s = d.groupby('method')['auroc'].agg(['mean', 'std']).reindex(METHODS)
    ax.bar(np.arange(len(METHODS)) + k * w, s['mean'], w, yerr=s['std'], label=model, capsize=2)
ax.set_xticks(np.arange(len(METHODS)) + w, METHODS, rotation=30, ha='right'); ax.set_ylim(0.4, 1)
ax.set_ylabel('AUROC (mean ± std)'); ax.legend(fontsize=8); ax.set_title('LOGO grup: AUROC per metode OOD')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_auroc_methods_group.png', dpi=150); plt.close(fig)

# gambar 3: risk-coverage (XGBoost, grup)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
d = runs_g[runs_g.model == 'XGBoost']
for meth in METHODS:
    dm = d[d.method == meth]
    ax.plot(COVS, [dm[f'selacc_cov{int(q * 100)}'].mean() for q in COVS], marker='o', ms=3, label=meth)
ax.set_xlabel('Coverage (data seen yang diterima)'); ax.set_ylabel('Selective accuracy')
ax.legend(fontsize=7); ax.set_title('Risk-coverage, LOGO grup, XGBoost')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_risk_coverage_group.png', dpi=150); plt.close(fig)

# gambar 4: AUROC per jenis serangan (XGBoost)
dt = runs_t[runs_t.model == 'XGBoost'].groupby(['held', 'method'])['auroc'].mean().unstack().reindex(columns=METHODS)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(dt.values, cmap='viridis', vmin=0.5, vmax=1, aspect='auto')
ax.set_xticks(range(dt.shape[1]), dt.columns, rotation=30, ha='right'); ax.set_yticks(range(dt.shape[0]), dt.index)
for i in range(dt.shape[0]):
    for j in range(dt.shape[1]):
        ax.text(j, i, f'{dt.values[i, j]:.2f}', ha='center', va='center', color='white', fontsize=7)
fig.colorbar(im, ax=ax, label='AUROC'); ax.set_title('LOGO per jenis serangan, XGBoost: AUROC')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_auroc_heatmap_type.png', dpi=150); plt.close(fig)

# %%
conf = runs_g.drop_duplicates(['held', 'model', 'seed']).groupby(['held', 'model'])['conf_unseen'].mean().unstack().round(2)
conf.to_csv(f'{OUT}/logo_group_conf_unseen.csv')
print('Rata-rata max-probability pada serangan unseen (%):'); print(conf)
print('SELESAI. File di', OUT)
