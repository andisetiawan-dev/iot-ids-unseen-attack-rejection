"""Membuat Gambar 2-6 untuk paper dari hasil eksperimen (Gambar 1 = diagram draw.io di folder figures/).

Jalankan dari akar repo:  python src/make_figures.py
Masukan : results/evaluation_inputs/*.csv  dan  results/raw/<dataset>/scores_seed0.npz
Keluaran: figures/fig2_auroc_per_family.png, fig3_score_distribution.png, fig4_ensemble_size.png,
          fig5_risk_coverage.png, fig6_mechanism.png
"""
import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get('REPO_ROOT', Path(__file__).resolve().parents[1]))
IN, RAW, OUT = ROOT / 'results' / 'evaluation_inputs', ROOT / 'results' / 'raw', ROOT / 'figures'
OUT.mkdir(exist_ok=True)
DS = ('edge', 'cic', 'ton')
NAMES = {'edge': 'Edge-IIoTset', 'cic': 'CICIoT2023', 'ton': 'TON_IoT'}
runs = {d: pd.read_csv(IN / f'{d}_logo_group_runs.csv') for d in DS}
xe = {d: r[r.model == 'XGBEns'] for d, r in runs.items()}       # model utama paper: ensemble XGBoost
COL = {'Entropy': '#9aa5b1', 'MI': '#2b6cb0', 'kNN': '#d69e2e', 'Fusion_MI_kNN': '#c05621'}
LAB = {'Entropy': 'Entropy', 'MI': 'MI (4 members)', 'kNN': 'kNN distance', 'Fusion_MI_kNN': 'Fusion MI+kNN'}
plt.rcParams.update({'font.family': 'serif', 'font.size': 11})

# ---- Gambar 2: AUROC tiap grup yang disembunyikan
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.7), gridspec_kw={'width_ratios': [5, 7, 5]})
for ax, d in zip(axs, DS):
    p = xe[d][xe[d].method.isin(COL)].pivot_table(index='held', columns='method', values='auroc')[list(COL)]
    p = p.rename(index={'Information_Gathering': 'Info. Gath.', 'DoS_DDoS': 'DoS/DDoS'})
    for k, m in enumerate(COL):
        ax.bar(np.arange(len(p)) + k * 0.2, p[m], 0.2, label=LAB[m], color=COL[m])
    ax.set_xticks(np.arange(len(p)) + 0.3); ax.set_xticklabels(p.index, rotation=25, ha='right')
    ax.axhline(0.5, color='k', lw=0.6, ls=':'); ax.set_ylim(0.4, 1.02); ax.set_title(NAMES[d]); ax.spines[['top', 'right']].set_visible(False)
axs[0].set_ylabel('AUROC (unseen vs. seen)')
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, fontsize=10, frameon=False)
plt.tight_layout(rect=(0, 0.07, 1, 1)); plt.savefig(OUT / 'fig2_auroc_per_family.png', dpi=220); plt.close()

# ---- Gambar 3: sebaran skor unseen sebagai persentil terhadap skor seen (seed 0)
CASES = [('edge', 'Malware'), ('cic', 'DDoS'), ('ton', 'Information_Gathering')]
fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
for ax, (d, held) in zip(axs, CASES):
    z = np.load(RAW / d / 'scores_seed0.npz')
    for meth, col, lab in (('Entropy', '#7a8794', 'Entropy'), ('MI', '#2b6cb0', 'MI (4 members)')):
        seen, ood = z[f'{held}__{meth}__id'], z[f'{held}__{meth}__ood']
        pct = np.searchsorted(np.sort(seen), ood) / len(seen)
        ax.hist(pct, bins=20, range=(0, 1), density=True, histtype='stepfilled', alpha=.45, color=col, label=lab, lw=0)
        ax.hist(pct, bins=20, range=(0, 1), density=True, histtype='step', color=col, lw=1.4)
    ax.axhline(1.0, color='k', lw=.9, ls='--')
    ax.set_title(f'{NAMES[d]}: {held.replace("_", " ")} held out', fontsize=10)
    ax.set_xlabel('Score percentile among seen test flows'); ax.spines[['top', 'right']].set_visible(False)
axs[0].set_ylabel('Density of unseen flows'); axs[0].legend(fontsize=10, frameon=False, loc='upper left')
plt.tight_layout(); plt.savefig(OUT / 'fig3_score_distribution.png', dpi=220); plt.close()

# ---- Gambar 4: AUROC MI dan Entropy terhadap jumlah anggota ensemble
fig, ax = plt.subplots(figsize=(7.4, 3.2))
for d, c in zip(DS, ('#2b6cb0', '#c05621', '#2f855a')):
    a = xe[d].groupby('method').auroc.mean()
    ax.plot([2, 4, 8], [a[f'MI_n{n}'] for n in (2, 4, 8)], 'o-', color=c, label=f'{NAMES[d]} - MI')
    ax.plot([2, 4, 8], [a[f'Entropy_n{n}'] for n in (2, 4, 8)], 's--', color=c, alpha=.55, label=f'{NAMES[d]} - Entropy')
ax.set_xticks([2, 4, 8]); ax.set_xlabel('Ensemble members'); ax.set_ylabel('AUROC')
ax.legend(fontsize=7.5, frameon=False, ncol=1, loc='center left', bbox_to_anchor=(1.02, 0.5)); ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / 'fig4_ensemble_size.png', dpi=220); plt.close()

# ---- Gambar 5: risk-coverage (akurasi selektif pada trafik seen)
COV = [50, 60, 70, 80, 90, 95, 100]
fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
for ax, d in zip(axs, DS):
    for m in COL:
        dm = xe[d][xe[d].method == m]
        ax.plot([c / 100 for c in COV], [dm[f'selacc_cov{c}'].mean() for c in COV], 'o-', ms=3.5, color=COL[m], label=LAB[m])
    ax.set_title(NAMES[d]); ax.set_xlabel('Coverage of seen test flows'); ax.spines[['top', 'right']].set_visible(False)
axs[0].set_ylabel('Selective accuracy'); h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, fontsize=10, frameon=False)
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(OUT / 'fig5_risk_coverage.png', dpi=220); plt.close()

# ---- Gambar 6: kelas seen terdekat vs porsi unseen yang ditolak MI
fig, axs = plt.subplots(1, 3, figsize=(11, 4.1))
for ax, d in zip(axs, DS):
    n = pd.read_csv(IN / f'{d}_nn_analysis.csv'); n = n[n.frac_ood >= 0.05]
    rej, sh = n.pivot(index='held', columns='nn_class', values='mi_rej_rate'), n.pivot(index='held', columns='nn_class', values='frac_ood')
    ax.imshow(rej.values, cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(rej.shape[1]), [c.replace('Information_Gathering', 'Info. Gath.') for c in rej.columns], rotation=35, ha='right', fontsize=9)
    ax.set_yticks(range(rej.shape[0]), [h.replace('Information_Gathering', 'Info. Gath.') for h in rej.index], fontsize=9)
    for i in range(rej.shape[0]):
        for j in range(rej.shape[1]):
            v = rej.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v * 100:.0f}%\n({sh.values[i, j] * 100:.0f}%)', ha='center', va='center', fontsize=7.8, color='white' if v > .55 else 'black')
    ax.set_title(NAMES[d]); ax.set_xlabel('Nearest seen class of the unseen flow', fontsize=9.5)
axs[0].set_ylabel('Held-out family', fontsize=9.5)
plt.tight_layout(); plt.savefig(OUT / 'fig6_mechanism.png', dpi=220); plt.close()
print('selesai:', sorted(p.name for p in OUT.glob('fig*.png')))
