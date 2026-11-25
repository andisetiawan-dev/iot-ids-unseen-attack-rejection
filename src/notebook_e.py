# %% [markdown]
# # Notebook E: benchmark OOD pada IDS IoT, dataset ketiga TON_IoT + MI per-pohon untuk random forest
#
# Tambahan dari notebook C:
# 1. Dataset penuh (bukan subsampel), preprocessing lebih bersih, kolom port dibuang (diuji ulang di sensitivity analysis).
# 2. Dataset kedua: CIC-IoT-2023 (7 grup serangan) untuk uji generalisasi.
# 3. Skor OOD berbasis ensemble (mutual information antar anggota ensemble).
# 4. Metrik mixed-stream: error rate di antara sampel yang diterima pada lalu lintas campuran seen + unseen.
# 5. Eksperimen pseudo-OOD: bisakah metode OOD terbaik dipilih tanpa data unseen asli?

# %%
import os, glob, time, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import logsumexp, softmax
from scipy.stats import wilcoxon, spearmanr
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

DATASET = os.environ.get('DATASET', 'ton')   # mk_nb mengganti baris ini: 'edge', 'cic', atau 'ton'
QUICK = os.environ.get('QUICK') == '1'
OUT = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.environ.get('OUT', 'out')
os.makedirs(OUT, exist_ok=True)
SEEDS = [0] if QUICK else [0, 1, 2, 3, 4]
CW_SEEDS = SEEDS
TYPE_SEEDS = SEEDS[:3]
MAXN = 4000 if QUICK else 30000        # batas jumlah sampel val / test / OOD per run (hemat waktu kNN)
MODELS = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'XGBEns']
if QUICK:
    MODELS = ['XGBoost', 'XGBEns', 'RandomForest']
TYPE_MODELS = ['XGBoost', 'LightGBM']
RUN_TYPE = DATASET == 'edge' and not QUICK
print('DATASET =', DATASET, '| QUICK =', QUICK, '| seeds =', SEEDS)

# %% [markdown]
# ## Data

# %%
def finalize(df, feats):
    n0 = len(df)
    df = df.drop_duplicates(subset=feats + ['Attack_type']).reset_index(drop=True)
    keep = [c for c in feats if df[c].nunique() > 1]
    print(f'duplikat dibuang: {n0 - len(df)} | kolom konstan dibuang: {len(feats) - len(keep)} | fitur: {len(keep)}')
    return df, keep


if DATASET == 'edge':
    FERRAG = {
        'Normal': 'Benign',
        'DDoS_UDP': 'DoS_DDoS', 'DDoS_ICMP': 'DoS_DDoS', 'DDoS_HTTP': 'DoS_DDoS', 'DDoS_TCP': 'DoS_DDoS',
        'Port_Scanning': 'Information_Gathering', 'Vulnerability_scanner': 'Information_Gathering',
        'Fingerprinting': 'Information_Gathering',
        'MITM': 'MITM',
        'SQL_injection': 'Injection', 'XSS': 'Injection', 'Uploading': 'Injection',
        'Backdoor': 'Malware', 'Password': 'Malware', 'Ransomware': 'Malware',
    }
    BENIGN_TYPE = 'Normal'
    sub_path = 'Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv'
    cands = [f'/kaggle/input/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot/{sub_path}',
             f'/kaggle/input/edgeiiotset-cyber-security-dataset-of-iot-iiot/{sub_path}']
    path = next((c for c in cands if os.path.exists(c)), os.environ.get('DATA'))
    print('input:', os.listdir('/kaggle/input') if os.path.isdir('/kaggle/input') else '-', '| path:', path, flush=True)
    raw = pd.read_csv(path, dtype=str)   # semua string dulu, konversi manual (hindari crash dtype campuran)
    print(path, raw.shape, flush=True)
    raw['Attack_type'] = raw['Attack_type'].astype(str).str.strip()
    # identitas / timestamp / payload mentah dibuang; port disimpan terpisah untuk sensitivity analysis
    ID_COLS = ['frame.time', 'ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4', 'arp.dst.proto_ipv4',
               'http.file_data', 'http.request.full_uri', 'icmp.transmit_timestamp', 'http.request.uri.query',
               'tcp.options', 'tcp.payload', 'mqtt.msg']
    PORT_COLS = [c for c in ['tcp.srcport', 'tcp.dstport', 'udp.port'] if c in raw.columns]
    raw = raw.drop(columns=[c for c in ID_COLS if c in raw.columns])
    feats = [c for c in raw.columns if c not in ['Attack_label', 'Attack_type']]
    for c in list(feats):
        if raw[c].dtype == object or str(raw[c].dtype).startswith('str'):
            num = pd.to_numeric(raw[c], errors='coerce')
            if (num.notna() | raw[c].isna()).mean() > 0.99:
                raw[c] = num
            elif raw[c].nunique() <= 50:
                raw[c] = pd.factorize(raw[c].astype(str))[0]
            else:
                print('drop kolom kardinalitas tinggi:', c)
                feats.remove(c)
                raw = raw.drop(columns=[c])
    raw['group'] = raw['Attack_type'].map(FERRAG)
    assert raw['group'].notna().all(), raw.loc[raw['group'].isna(), 'Attack_type'].unique()
    PORT_COLS = [c for c in PORT_COLS if c in raw.columns]
    feats_np = [c for c in feats if c not in PORT_COLS]
    df, FEATS_ALL = finalize(raw, feats_np + PORT_COLS)
    FEATS = [c for c in FEATS_ALL if c not in PORT_COLS]
elif DATASET == 'ton':
    TON = {'normal': 'Benign', 'ddos': 'DoS_DDoS', 'dos': 'DoS_DDoS', 'scanning': 'Information_Gathering',
           'mitm': 'MITM', 'injection': 'Injection', 'xss': 'Injection',
           'backdoor': 'Malware', 'password': 'Malware', 'ransomware': 'Malware'}
    BENIGN_TYPE = 'normal'
    hits = glob.glob('/kaggle/input/**/train_test_network.csv', recursive=True)
    path = hits[0] if hits else os.environ['DATA']
    raw = pd.read_csv(path, dtype=str)
    print(path, raw.shape, flush=True)
    raw['Attack_type'] = raw['type'].astype(str).str.strip().str.lower()
    # IP = identitas mesin penyerang/korban di testbed, 'label' dan 'type' = target; semuanya dibuang
    raw = raw.drop(columns=['src_ip', 'dst_ip', 'label', 'type'])
    PORT_COLS = ['src_port', 'dst_port']
    feats = [c for c in raw.columns if c != 'Attack_type']
    for c in list(feats):
        num = pd.to_numeric(raw[c], errors='coerce')
        if (num.notna() | raw[c].isna()).mean() > 0.99:
            raw[c] = num
        elif raw[c].nunique() <= 50:
            raw[c] = pd.factorize(raw[c].astype(str))[0]
        else:
            print('drop kolom kardinalitas tinggi:', c)
            feats.remove(c)
            raw = raw.drop(columns=[c])
    raw['group'] = raw['Attack_type'].map(TON)
    assert raw['group'].notna().all(), raw.loc[raw['group'].isna(), 'Attack_type'].unique()
    PORT_COLS = [c for c in PORT_COLS if c in raw.columns]
    feats_np = [c for c in feats if c not in PORT_COLS]
    df, FEATS_ALL = finalize(raw, feats_np + PORT_COLS)
    FEATS = [c for c in FEATS_ALL if c not in PORT_COLS]
else:
    BENIGN_TYPE = 'BenignTraffic'
    hits = [h for h in glob.glob('/kaggle/input/**/train.csv', recursive=True) if 'CICIOT' in h.upper()]
    path = hits[0] if hits else os.environ['DATA']
    raw = pd.read_csv(path)
    print(path, raw.shape)
    raw = raw.rename(columns={'label': 'Attack_type'})
    CAP = 800 if QUICK else 4000
    raw = pd.concat([g.sample(min(len(g), 30000 if t == BENIGN_TYPE else CAP), random_state=0)
                     for t, g in raw.groupby('Attack_type')]).reset_index(drop=True)

    def cic_group(t):
        if t == 'BenignTraffic': return 'Benign'
        if t.startswith('DDoS'): return 'DDoS'
        if t.startswith('DoS'): return 'DoS'
        if t.startswith('Mirai'): return 'Mirai'
        if t.startswith('Recon') or t == 'VulnerabilityScan': return 'Recon'
        if t in ('MITM-ArpSpoofing', 'DNS_Spoofing'): return 'Spoofing'
        if t == 'DictionaryBruteForce': return 'BruteForce'
        return 'WebBased'
    raw['group'] = raw['Attack_type'].map(cic_group)
    feats = [c for c in raw.columns if c not in ['Attack_type', 'group']]
    PORT_COLS = []
    df, FEATS = finalize(raw, feats)

X = df[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e15, 1e15).values.astype(np.float32)
GROUPS = [g for g in df['group'].unique() if g != 'Benign']
TYPES = [t for t in df['Attack_type'].unique() if t != BENIGN_TYPE]
print(df['group'].value_counts().to_dict())
print(df['Attack_type'].value_counts().to_dict())

# %% [markdown]
# ## Model dan skor OOD

# %%
class XGBEns:
    """Ensemble 8 XGBoost dengan subsampling beda; prediksi utama = 4 anggota pertama; MI = disagreement antar anggota."""
    def __init__(self, seed, n=8):
        self.ms = [xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
                                     colsample_bytree=0.8, eval_metric='mlogloss', random_state=seed * 100 + i,
                                     n_jobs=-1, verbosity=0) for i in range(n)]

    def fit(self, A, y):
        for m in self.ms:
            m.fit(A, y)
        return self

    def member_probas(self, A):
        return np.stack([m.predict_proba(A) for m in self.ms])

    def predict_proba(self, A):
        return self.member_probas(A)[:4].mean(0)

    def predict(self, A):
        return self.predict_proba(A).argmax(1)


def make(name, seed):
    if name == 'XGBEns':
        return XGBEns(seed)
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
    out = {}
    if isinstance(m, XGBEns):
        mp = [m.member_probas(s) for s in sets]
        probs = [a[:4].mean(0) for a in mp]
        for n in (2, 4, 8):
            out[f'MI_n{n}'] = [entropy(a[:n].mean(0)) - np.mean([entropy(q) for q in a[:n]], axis=0) for a in mp]
            out[f'Entropy_n{n}'] = [entropy(a[:n].mean(0)) for a in mp]
        out['MI'] = out['MI_n4']
    else:
        probs = [m.predict_proba(s) for s in sets]
    out.update({'MSP': [1 - p.max(1) for p in probs], 'Entropy': [entropy(p) for p in probs]})
    if name == 'RandomForest':   # disagreement antar pohon: H(rata-rata) - rata-rata H(pohon)
        mit = []
        for s in sets:
            s32, ps, hs = s.astype(np.float32), 0, 0
            for t in m.estimators_:
                pt = t.predict_proba(s32)
                ps, hs = ps + pt, hs + entropy(pt)
            mit.append(entropy(ps / len(m.estimators_)) - hs / len(m.estimators_))
        out['MI_trees'] = mit
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
    for tq in (0.90, 0.99):
        th = np.quantile(sva, tq)
        a2 = ste <= th
        row[f'coverage_t{int(tq * 100)}'] = a2.mean()
        row[f'sel_acc_t{int(tq * 100)}'] = correct_te[a2].mean() if a2.any() else np.nan
        row[f'ood_rej_t{int(tq * 100)}'] = (so > th).mean()
    # error rate di antara sampel yang diterima pada lalu lintas campuran (r = porsi unseen)
    a_id, e_id, a_o = row['coverage'], 1 - row['sel_acc'], 1 - row['ood_rej']
    for r in (0.01, 0.05, 0.10):
        row[f'mixerr_r{int(r * 100)}'] = ((1 - r) * a_id * e_id + r * a_o) / ((1 - r) * a_id + r * a_o)
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

# ablation: setting jenis serangan, dan efek kolom port
y15 = LabelEncoder().fit_transform(df['Attack_type'])
f15 = []
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y15):
    p = make('XGBoost', 0).fit(X[tr], y15[tr]).predict(X[te]).ravel()
    f15.append(f1_score(y15[te], p, average='macro'))
abl = [{'setting': 'grup, XGBoost, tanpa kolom port', 'f1_macro': cw_sum.loc['XGBoost', 'f1_mean']},
       {'setting': 'jenis serangan, XGBoost', 'f1_macro': np.mean(f15)}]
Xp = None
if PORT_COLS:
    Xp = df[FEATS_ALL].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e15, 1e15).values.astype(np.float32)
    fp = [f1_score(y_g[te], make('XGBoost', 0).fit(Xp[tr], y_g[tr]).predict(Xp[te]).ravel(), average='macro')
          for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y_g)]
    abl.append({'setting': 'grup, XGBoost, DENGAN kolom port', 'f1_macro': np.mean(fp)})
abl = pd.DataFrame(abl).round(4)
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
NN_ROWS, DUMP = [], {}


def fuse(a, b):
    """Rata-rata persentil (terhadap distribusi validation seen) dari dua skor OOD."""
    ra, rb = np.sort(a[0]), np.sort(b[0])
    return [(np.searchsorted(ra, a[i]) / len(ra) + np.searchsorted(rb, b[i]) / len(rb)) / 2 for i in range(3)]


def sub(idx, rng):
    return idx if len(idx) <= MAXN else rng.choice(idx, MAXN, replace=False)


def pseudo_rows(held, seed, tr, va, te, lab, qt, Xm):
    """Sembunyikan satu grup seen sebagai 'unseen palsu', ukur AUROC tiap metode (tanpa menyentuh OOD asli)."""
    rng = np.random.RandomState(seed)
    cand = [g for g in np.unique(lab[tr]) if g != 'Benign' and ((lab[va] == g).sum() + (lab[te] == g).sum()) >= 50]
    rows = []
    for p in list(rng.permutation(cand))[:2]:
        ptr = tr[lab[tr] != p]
        pid = va[lab[va] != p]
        pood = np.r_[va[lab[va] == p], te[lab[te] == p]]
        le = LabelEncoder().fit(lab[ptr])
        ytr, yid = le.transform(lab[ptr]), le.transform(lab[pid])
        Z = [qt.transform(Xm[i]).astype(np.float64) for i in (ptr, pid, pood)]
        fs = feature_scores(Z[0], ytr, [Z[1], Z[2]], seed)
        m = make('XGBoost', seed).fit(Xm[ptr], ytr)
        ms, _ = model_scores('XGBoost', m, [Xm[pid], Xm[pood]], yid)
        for meth, (sid, so) in {**ms, **fs}.items():
            rows.append({'held': held, 'seed': seed, 'pseudo': p, 'method': meth,
                         'auroc_pseudo': roc_auc_score(np.r_[np.zeros(len(sid)), np.ones(len(so))], np.r_[sid, so])})
    return rows


def logo(col, helds, models, seeds, level, Xm=None, pseudo=False):
    Xm = X if Xm is None else Xm
    lab = df[col].values
    rows, prow = [], []
    for held in helds:
        ood_all = np.where(lab == held)[0]
        seen_idx = np.where(lab != held)[0]
        for seed in seeds:
            t0 = time.time()
            rng = np.random.RandomState(seed)
            tr, rest = train_test_split(seen_idx, test_size=0.4, stratify=lab[seen_idx], random_state=seed)
            va, te = train_test_split(rest, test_size=0.5, stratify=lab[rest], random_state=seed)
            va, te, ood_idx = sub(va, rng), sub(te, rng), sub(ood_all, rng)
            le = LabelEncoder().fit(lab[tr])
            ytr, yva, yte = le.transform(lab[tr]), le.transform(lab[va]), le.transform(lab[te])
            qt = QuantileTransformer(n_quantiles=500, output_distribution='normal',
                                     subsample=100000, random_state=seed).fit(Xm[tr])
            Ztr, Zva, Zte, Zo = [qt.transform(Xm[i]).astype(np.float64) for i in (tr, va, te, ood_idx)]
            fs = feature_scores(Ztr, ytr, [Zva, Zte, Zo], seed)
            sets = [Xm[va], Xm[te], Xm[ood_idx]]
            first = seed == seeds[0] and level == 'group'
            if first:   # kelas seen terdekat dari tiap sampel OOD (analisis mekanisme)
                s1 = np.random.RandomState(seed).choice(len(Ztr), min(15000, len(Ztr)), replace=False)
                nn1 = NearestNeighbors(n_neighbors=1, algorithm='brute', n_jobs=-1).fit(Ztr[s1])
                nn_lab = le.classes_[ytr[s1][nn1.kneighbors(Zo)[1][:, 0]]]
            for name in models:
                m = make(name, seed).fit(Xm[tr], ytr)
                ms, probs = model_scores(name, m, sets, yva)
                if name == 'RandomForest':
                    ms['Fusion_MItrees_kNN'] = fuse(ms['MI_trees'], fs['kNN'])
                if name == 'XGBEns':
                    ms['Fusion_MI_kNN'] = fuse(ms['MI'], fs['kNN'])
                    if first:
                        rej = ms['MI'][2] > np.quantile(ms['MI'][0], 0.95)
                        for c in np.unique(nn_lab):
                            mk = nn_lab == c
                            NN_ROWS.append({'held': held, 'nn_class': c, 'n': int(mk.sum()),
                                            'frac_ood': mk.mean(), 'mi_rej_rate': rej[mk].mean()})
                        for meth, (_, s_id, s_ood) in {**ms, **fs}.items():
                            DUMP[f'{held}|{meth}|id'] = np.asarray(s_id, dtype=np.float32)
                            DUMP[f'{held}|{meth}|ood'] = np.asarray(s_ood, dtype=np.float32)
                pred = probs[1].argmax(1)
                correct = (pred == yte).astype(float)
                if correct.mean() < 0.8:
                    print(f'!! PERINGATAN: classifier {name} gagal ({held}, seed {seed}): acc={correct.mean():.3f}')
                base = {'level': level, 'held': held, 'model': name, 'seed': seed,
                        'f1_seen': f1_score(yte, pred, average='macro'), 'acc_seen': correct.mean(),
                        'conf_unseen': probs[2].max(1).mean() * 100}
                for r in (0.01, 0.05, 0.10):
                    base[f'mixerr_none_r{int(r * 100)}'] = (1 - r) * (1 - correct.mean()) + r
                for meth, (sva, ste, so) in {**ms, **fs}.items():
                    rows.append({**base, 'method': meth, **metrics(sva, ste, so, correct)})
            if pseudo:
                prow += pseudo_rows(held, seed, tr, va, te, lab, qt, Xm)
            print(f'[{level}] {str(held):24s} seed={seed} {time.time() - t0:5.1f}s', flush=True)
    return (pd.DataFrame(rows), pd.DataFrame(prow)) if pseudo else pd.DataFrame(rows)


runs_g, pseudo_df = logo('group', GROUPS, MODELS, SEEDS, 'group', pseudo=True)
runs_g.to_csv(f'{OUT}/logo_group_runs.csv', index=False)
pseudo_df.to_csv(f'{OUT}/pseudo_ood_runs.csv', index=False)
pd.DataFrame(NN_ROWS).to_csv(f'{OUT}/nn_analysis.csv', index=False)
np.savez_compressed(f'{OUT}/scores_seed0.npz', **{k.replace('|', '__'): v for k, v in DUMP.items()})

# %%
if RUN_TYPE:
    runs_t = logo('Attack_type', TYPES, TYPE_MODELS, TYPE_SEEDS, 'type')
    runs_t.to_csv(f'{OUT}/logo_type_runs.csv', index=False)
else:
    runs_t = pd.DataFrame(columns=runs_g.columns)

# sensitivity: apakah kolom port mengubah kesimpulan? (XGBoost, grup)
if Xp is not None:
    runs_p = logo('group', GROUPS, ['XGBoost'], SEEDS, 'group_with_ports', Xm=Xp)
    runs_p.to_csv(f'{OUT}/logo_group_ports_runs.csv', index=False)

# %% [markdown]
# ## E3-E5: ringkasan, uji signifikansi, kurva risk-coverage

# %%
MET = ['auroc', 'aupr_ood', 'fpr95', 'aurc', 'coverage', 'sel_acc', 'ood_rej', 'mixerr_r1', 'mixerr_r5', 'mixerr_r10',
       'coverage_t90', 'sel_acc_t90', 'ood_rej_t90', 'coverage_t99', 'sel_acc_t99', 'ood_rej_t99']
MAIN = ['MSP', 'Entropy', 'Energy', 'MSP_T', 'Entropy_T', 'MI', 'Fusion_MI_kNN', 'Mahalanobis', 'kNN', 'IsolationForest']
METHODS = MAIN + ['MI_n2', 'MI_n8', 'Entropy_n2', 'Entropy_n8', 'MI_trees', 'Fusion_MItrees_kNN']


def summarize(runs, path):
    g = runs.groupby(['model', 'method'])[MET].agg(['mean', 'std'])
    g.columns = [f'{a}_{b}' for a, b in g.columns]
    g = g.round(4).reset_index()
    g.to_csv(path, index=False)
    return g


sum_g = summarize(runs_g, f'{OUT}/logo_group_summary.csv')
sum_t = summarize(runs_t, f'{OUT}/logo_type_summary.csv') if RUN_TYPE else None
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print('== LOGO grup (XGBoost) =='); print(sum_g[sum_g.model == 'XGBoost'].to_string(index=False))
if RUN_TYPE:
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


stats = pd.concat([wilcox(runs_g, 'group')] + ([wilcox(runs_t, 'type')] if RUN_TYPE else []))
stats.to_csv(f'{OUT}/stats_wilcoxon.csv', index=False)
print(stats[(stats.model == 'XGBoost')].to_string(index=False))

# gambar 1: heatmap AUROC per held-out (XGBoost)
piv = per_held.pivot(index='method', columns='held', values='auroc').reindex([m for m in METHODS if m in per_held.method.unique()])
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
    s = d.groupby('method')['auroc'].agg(['mean', 'std']).reindex(MAIN)
    ax.bar(np.arange(len(MAIN)) + k * w, s['mean'].fillna(0), w, yerr=s['std'].fillna(0), label=model, capsize=2)
ax.set_xticks(np.arange(len(MAIN)) + w, MAIN, rotation=30, ha='right'); ax.set_ylim(0.4, 1)
ax.set_ylabel('AUROC (mean ± std)'); ax.legend(fontsize=8); ax.set_title('LOGO grup: AUROC per metode OOD')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_auroc_methods_group.png', dpi=150); plt.close(fig)

# gambar 3: risk-coverage (XGBoost, grup)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
d = runs_g[runs_g.model == 'XGBoost']
for meth in [m for m in METHODS if m in d.method.unique()]:
    dm = d[d.method == meth]
    ax.plot(COVS, [dm[f'selacc_cov{int(q * 100)}'].mean() for q in COVS], marker='o', ms=3, label=meth)
ax.set_xlabel('Coverage (data seen yang diterima)'); ax.set_ylabel('Selective accuracy')
ax.legend(fontsize=7); ax.set_title('Risk-coverage, LOGO grup, XGBoost')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_risk_coverage_group.png', dpi=150); plt.close(fig)

if RUN_TYPE:
    # gambar 4: AUROC per jenis serangan (XGBoost)
    dt = runs_t[runs_t.model == 'XGBoost'].groupby(['held', 'method'])['auroc'].mean().unstack().reindex(columns=[m for m in METHODS if m in ['MSP', 'Entropy', 'Energy', 'MSP_T', 'Entropy_T', 'Mahalanobis', 'kNN', 'IsolationForest']])
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
# %% [markdown]
# ## Pseudo-OOD: bisakah metode terbaik dipilih tanpa data unseen asli?

# %%
ps = pseudo_df.groupby(['held', 'seed', 'method'])['auroc_pseudo'].mean().rename('pseudo').reset_index()
real = runs_g[runs_g.model == 'XGBoost'][['held', 'seed', 'method', 'auroc']]
mm = ps.merge(real, on=['held', 'seed', 'method'])
res = []
for (h, s), d in mm.groupby(['held', 'seed']):
    pick, best = d.loc[d.pseudo.idxmax()], d.loc[d.auroc.idxmax()]
    res.append({'held': h, 'seed': s, 'spearman': spearmanr(d.pseudo, d.auroc).correlation,
                'pick': pick.method, 'best': best.method, 'hit': pick.method == best.method,
                'regret_pseudo': best.auroc - pick.auroc,
                'regret_always_entropy': best.auroc - d[d.method == 'Entropy'].auroc.iloc[0],
                'regret_always_msp': best.auroc - d[d.method == 'MSP'].auroc.iloc[0]})
pr = pd.DataFrame(res)
pr.to_csv(f'{OUT}/pseudo_ood_selection.csv', index=False)
print(pr.round(4).to_string(index=False))
print('rata-rata Spearman(pseudo, real):', round(pr.spearman.mean(), 3), '| hit rate top-1:', round(pr.hit.mean(), 3))
print('regret rata-rata  pseudo:', round(pr.regret_pseudo.mean(), 4),
      '| selalu Entropy:', round(pr.regret_always_entropy.mean(), 4),
      '| selalu MSP:', round(pr.regret_always_msp.mean(), 4))

# %% [markdown]
# ## Ablation ukuran ensemble, titik operasi, dan analisis mekanisme

# %%
rf = runs_g[runs_g.model == 'RandomForest'].groupby('method')[['auroc', 'ood_rej', 'mixerr_r5']].mean().round(4)
rf.to_csv(f'{OUT}/rf_tree_mi.csv'); print(rf)
ab = (runs_g[(runs_g.model == 'XGBEns') & runs_g.method.str.match(r'^(MI|Entropy)_n\d')]
      .groupby('method')[['auroc', 'ood_rej', 'mixerr_r5']].mean().round(4))
ab.to_csv(f'{OUT}/ens_size_ablation.csv')
print(ab)
op = (runs_g[runs_g.model == 'XGBEns'].groupby('method')[
    ['coverage_t90', 'sel_acc_t90', 'ood_rej_t90', 'coverage', 'sel_acc', 'ood_rej',
     'coverage_t99', 'sel_acc_t99', 'ood_rej_t99']].mean().round(3))
op.to_csv(f'{OUT}/operating_points_xgbens.csv')
print(op)
nn = pd.DataFrame(NN_ROWS)
print(nn[nn.frac_ood >= 0.05].round(3).to_string(index=False))

# %%
print('SELESAI. File di', OUT)
