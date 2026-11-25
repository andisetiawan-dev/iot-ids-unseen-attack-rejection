# %% [markdown]
# # Notebook G: evaluasi dan pembuatan tabel paper
#
# Notebook ini **tidak melatih model**. Ia hanya membaca file hasil eksperimen (CSV) dari notebook D, E, dan F,
# lalu menghitung ulang semua angka, tabel, uji statistik, dan gambar yang ada di paper.
# Di akhir, setiap angka yang tertulis di paper dicek otomatis terhadap hasil hitungan ini.
#
# Alur: baca data -> hitung tabel (satu bagian per tabel di paper) -> uji Wilcoxon -> gambar -> cek angka paper.
#
# Istilah singkat:
# - **held** = grup serangan yang disembunyikan dari training (dianggap *unseen*).
# - **method** = skor ketidakpastian (MSP, Entropy, MI, kNN, ...). Skor tinggi = diduga serangan baru.
# - **AUROC** = seberapa baik skor memisahkan unseen dari seen (0,5 = acak, 1 = sempurna).
# - **ood_rej** = persen unseen yang ditolak pada ambang yang mempertahankan 95% data seen.

# %%
import os, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# lokasi data: di Kaggle otomatis dicari, di laptop pakai env DATA
hits = glob.glob('/kaggle/input/**/edge_logo_group_runs.csv', recursive=True)
DATA = os.path.dirname(hits[0]) if hits else os.environ['DATA']
OUT = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.environ.get('OUT', 'out')
os.makedirs(OUT, exist_ok=True)
DS = ['edge', 'cic', 'ton']
NAMA = {'edge': 'Edge-IIoTset', 'cic': 'CICIoT2023', 'ton': 'TON_IoT'}
print('data:', DATA)


def baca(ds, nama='logo_group_runs'):
    return pd.read_csv(f'{DATA}/{ds}_{nama}.csv')


def simpan(df, nama):
    df.to_csv(f'{OUT}/{nama}.csv')
    return df


# Satu baris di logo_group_runs = satu (grup yang disembunyikan, seed, model, metode).
RUN = {ds: baca(ds) for ds in DS}
XE = {ds: r[r.model == 'XGBEns'] for ds, r in RUN.items()}      # ensemble XGBoost = model utama paper
print({ds: len(r) for ds, r in RUN.items()})


# %% [markdown]
# ## Alat bantu: rata-rata per metode dan uji Wilcoxon
# Uji Wilcoxon membandingkan dua metode pada pasangan yang sama (grup, seed): untuk 5 grup x 5 seed = 25 pasangan,
# untuk CIC 7 grup x 5 seed = 35 pasangan. `selisih` = rata-rata AUROC metode a dikurangi metode b.

# %%
def rata(r, model, kolom):
    """rata-rata kolom (mis. auroc) untuk tiap metode, pada model tertentu"""
    return r[r.model == model].groupby('method')[kolom].mean()


def wilcox(r, model, a, b):
    tabel = r[r.model == model].pivot_table(index=['held', 'seed'], columns='method', values='auroc')
    d = tabel[a] - tabel[b]
    return {'selisih': d.mean(), 'p': wilcoxon(d).pvalue, 'n_pasangan': len(d), 'n_positif': int((d > 0).sum())}


# %% [markdown]
# ## Tabel 1: closed-world (macro-F1)

# %%
cw = {ds: pd.read_csv(f'{DATA}/{ds}_cw_summary.csv', index_col=0)['f1_mean'] for ds in DS}
ablasi = {ds: pd.read_csv(f'{DATA}/{ds}_cw_ablation.csv').set_index('setting')['f1_macro'] for ds in DS}
T1 = simpan(pd.DataFrame(cw).rename(columns=NAMA), 'T1_closed_world_f1')
print(T1.round(3))
print('F1 dengan kolom port:', {ds: ablasi[ds].filter(like='DENGAN').round(3).tolist() for ds in DS})

# %% [markdown]
# ## Tabel 2: deteksi unseen dengan XGBEns (AUROC dan FPR@95TPR)
# Rata-rata atas semua grup dan seed. `MI_n8` = MI dengan 8 anggota (MI utama memakai 4 anggota).

# %%
METODE = ['MSP', 'Entropy', 'Mahalanobis', 'kNN', 'IsolationForest', 'MI', 'MI_n8', 'Fusion_MI_kNN']
T2 = pd.concat({NAMA[ds]: XE[ds].groupby('method')[['auroc', 'fpr95']].mean().loc[METODE] for ds in DS}, axis=1)
simpan(T2, 'T2_deteksi_xgbens'); print(T2.round(3))
conf = pd.Series({ds: XE[ds][XE[ds].method == 'MSP'].conf_unseen.mean() for ds in DS})
print('rata-rata confidence model pada serangan unseen (%):'); print(conf.round(1))

# %% [markdown]
# ## Uji signifikansi utama (XGBEns, per dataset)

# %%
PASANGAN = [('Entropy', 'MSP'), ('MI', 'MSP'), ('Fusion_MI_kNN', 'MI'), ('Fusion_MI_kNN', 'kNN'), ('MI_n8', 'MI')]
STAT = {(ds, a, b): wilcox(RUN[ds], 'XGBEns', a, b) for ds in DS for a, b in PASANGAN}
T_stat = pd.DataFrame({f'{ds}: {a} - {b}': v for (ds, a, b), v in STAT.items()}).T
simpan(T_stat, 'T_uji_wilcoxon_xgbens'); print(T_stat.round(5))

# %% [markdown]
# ## Tabel 3: pengaruh jenis classifier (skor berbasis logit)
# Energy tidak ada untuk random forest dan XGBEns (tidak punya logit tunggal).

# %%
MODEL = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'XGBEns']
T3 = pd.concat({NAMA[ds]: RUN[ds][RUN[ds].method.isin(['MSP', 'Entropy', 'Energy'])]
                .groupby(['method', 'model']).auroc.mean().unstack().reindex(columns=MODEL)
                .loc[['MSP', 'Entropy', 'Energy']] for ds in DS})
simpan(T3, 'T3_pengaruh_classifier'); print(T3.round(3))

# %% [markdown]
# ## Tabel 4: random forest dan MI antar pohon
# Untuk Edge dan CIC, skor MI pohon dihitung di notebook F (file `*_logo_group_runs_rf.csv`);
# untuk TON_IoT sudah ada di file utama.

# %%
RF = {'edge': baca('edge', 'logo_group_runs_rf'), 'cic': baca('cic', 'logo_group_runs_rf'),
      'ton': RUN['ton'][RUN['ton'].model == 'RandomForest']}
baris = {}
for ds in DS:
    a, rej = RF[ds].groupby('method').auroc.mean(), RF[ds].groupby('method').ood_rej.mean()
    baris[NAMA[ds]] = {'RF MSP': a['MSP'], 'RF Entropy': a['Entropy'], 'RF MI pohon': a['MI_trees'],
                       'XGBEns MI': XE[ds][XE[ds].method == 'MI'].auroc.mean(),
                       'RF Entropy rej': rej['Entropy'], 'RF MI rej': rej['MI_trees'],
                       'XGBEns MI rej': XE[ds][XE[ds].method == 'MI'].ood_rej.mean()}
T4 = simpan(pd.DataFrame(baris).T, 'T4_random_forest_mi_pohon'); print(T4.round(3))


def wilcox_rf(ds, a, b):
    tabel = RF[ds].pivot_table(index=['held', 'seed'], columns='method', values='auroc')
    d = tabel[a] - tabel[b]
    return {'selisih': d.mean(), 'p': wilcoxon(d).pvalue, 'n_pasangan': len(d), 'n_positif': int((d > 0).sum())}


PRF = [('MI_trees', 'MSP'), ('MI_trees', 'Entropy'), ('Fusion_MItrees_kNN', 'MI_trees')]
STAT_RF = {(ds, a, b): wilcox_rf(ds, a, b) for ds in DS for a, b in PRF}
T_stat_rf = pd.DataFrame({f'{ds}: {a} - {b}': v for (ds, a, b), v in STAT_RF.items()}).T
simpan(T_stat_rf, 'T4b_uji_wilcoxon_rf'); print(T_stat_rf.round(5))

# %% [markdown]
# ## Ukuran ensemble (Gambar 2) dan operating point (Tabel 5)

# %%
UKURAN = pd.concat({NAMA[ds]: XE[ds][XE[ds].method.str.match(r'^(MI|Entropy)_n\d')].groupby('method').auroc.mean()
                    for ds in DS}, axis=1)
simpan(UKURAN, 'ukuran_ensemble_auroc'); print(UKURAN.round(3))

KOL5 = ['sel_acc_t90', 'ood_rej_t90', 'sel_acc', 'ood_rej', 'sel_acc_t99', 'ood_rej_t99']
T5 = pd.concat({NAMA[ds]: XE[ds].groupby('method')[KOL5].mean().loc[['Entropy', 'MI', 'kNN', 'Fusion_MI_kNN']] for ds in DS})
simpan(T5, 'T5_operating_points'); print(T5.round(3))

# %% [markdown]
# ## Tabel 6: error rate pada aliran campuran
# r = porsi trafik unseen di aliran. "None" = menerima semua tanpa penolakan.

# %%
KOL6 = ['mixerr_r1', 'mixerr_r5', 'mixerr_r10']
dasar = lambda ds: XE[ds][XE[ds].method == 'MSP'][['mixerr_none_r1', 'mixerr_none_r5', 'mixerr_none_r10']].mean().set_axis(KOL6)
T6 = pd.concat({NAMA[ds]: pd.concat([dasar(ds).to_frame('None').T,
                                     XE[ds].groupby('method')[KOL6].mean().loc[['MSP', 'Entropy', 'MI', 'kNN', 'Fusion_MI_kNN']]])
                for ds in DS})
simpan(T6, 'T6_mixed_stream'); print(T6.round(3))

# %% [markdown]
# ## Hasil per grup serangan (Gambar 1)

# %%
UTAMA = ['Entropy', 'MI', 'kNN', 'Fusion_MI_kNN']
PERGRUP = {ds: XE[ds][XE[ds].method.isin(UTAMA + ['Mahalanobis'])].pivot_table(index='held', columns='method', values='auroc')
           for ds in DS}
for ds in DS:
    simpan(PERGRUP[ds], f'per_grup_{ds}')
print(PERGRUP['cic'].round(3))

# %% [markdown]
# ## Tabel 7: memilih metode dengan pseudo-unseen

# %%
PS = {ds: pd.read_csv(f'{DATA}/{ds}_pseudo_ood_selection.csv') for ds in DS}
T7 = simpan(pd.DataFrame({NAMA[ds]: {'hit_rate': p.hit.mean(), 'spearman': p.spearman.mean(),
                                    'regret_pilihan': p.regret_pseudo.mean(), 'regret_selalu_entropy': p.regret_always_entropy.mean()}
                          for ds, p in PS.items()}).T, 'T7_pseudo_ood')
print(T7.round(3))

# %% [markdown]
# ## Sensitivitas kolom port, dan tingkat jenis serangan (Edge)

# %%
PORT = {ds: pd.read_csv(f'{DATA}/{ds}_logo_group_ports_runs.csv') for ds in ['edge', 'ton']}
PM = ['MSP', 'Entropy', 'kNN', 'Mahalanobis']
T_port = pd.concat({NAMA[ds]: pd.DataFrame({'tanpa_port': rata(RUN[ds], 'XGBoost', 'auroc')[PM],
                                            'dengan_port': PORT[ds].groupby('method').auroc.mean()[PM]}) for ds in PORT})
simpan(T_port, 'T_sensitivitas_port'); print(T_port.round(3))

TIPE = baca('edge', 'logo_type_runs')
T_tipe = TIPE[TIPE.model == 'XGBoost'].groupby('method').auroc.mean().to_frame('auroc_xgboost')
simpan(T_tipe, 'T_tingkat_tipe_edge'); print(T_tipe.round(3))
tabel = TIPE[TIPE.model == 'XGBoost'].pivot_table(index=['held', 'seed'], columns='method', values='auroc')
d_tipe = tabel['kNN'] - tabel['MSP']
P_TIPE = {'selisih': d_tipe.mean(), 'p': wilcoxon(d_tipe).pvalue, 'n': len(d_tipe)}
print('kNN - MSP (tipe):', P_TIPE)

# %% [markdown]
# ## Analisis mekanisme: kelas seen terdekat dari tiap sampel unseen (seed 0)
# `frac_ood` = porsi sampel unseen yang paling dekat ke kelas itu, `mi_rej_rate` = porsi yang ditolak oleh MI.

# %%
NN = {ds: pd.read_csv(f'{DATA}/{ds}_nn_analysis.csv') for ds in DS}
for ds in DS:
    simpan(NN[ds][NN[ds].frac_ood >= 0.1].set_index(['held', 'nn_class']), f'mekanisme_{ds}')
print(NN['cic'][NN['cic'].frac_ood >= 0.1].round(3).to_string(index=False))

# %% [markdown]
# ## Gambar 1 dan Gambar 2

# %%
plt.rcParams.update({'font.family': 'serif', 'font.size': 9})
WARNA = {'Entropy': '#9aa5b1', 'MI': '#2b6cb0', 'kNN': '#d69e2e', 'Fusion_MI_kNN': '#c05621'}
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.5), gridspec_kw={'width_ratios': [5, 7, 5]})
for ax, ds in zip(axs, DS):
    p = PERGRUP[ds][UTAMA]
    for k, m in enumerate(UTAMA):
        ax.bar(np.arange(len(p)) + k * 0.2, p[m], 0.2, label=m, color=WARNA[m])
    ax.set_xticks(np.arange(len(p)) + 0.3); ax.set_xticklabels(p.index, rotation=25, ha='right')
    ax.axhline(0.5, color='k', lw=0.6, ls=':'); ax.set_ylim(0.4, 1.02); ax.set_title(NAMA[ds])
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, frameon=False)
plt.tight_layout(rect=(0, 0.07, 1, 1)); plt.savefig(f'{OUT}/gambar1_auroc_per_grup.png', dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(5.2, 3.3))
for ds, c in zip(DS, ('#2b6cb0', '#c05621', '#2f855a')):
    ax.plot([2, 4, 8], [UKURAN.loc[f'MI_n{n}', NAMA[ds]] for n in (2, 4, 8)], 'o-', color=c, label=f'{NAMA[ds]} MI')
    ax.plot([2, 4, 8], [UKURAN.loc[f'Entropy_n{n}', NAMA[ds]] for n in (2, 4, 8)], 's--', color=c, alpha=.55, label=f'{NAMA[ds]} Entropy')
ax.set_xticks([2, 4, 8]); ax.set_xlabel('Jumlah anggota ensemble'); ax.set_ylabel('AUROC'); ax.legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(f'{OUT}/gambar2_ukuran_ensemble.png', dpi=200); plt.close()

# %% [markdown]
# ## Cek angka paper
# Tiap baris di bawah adalah satu angka yang tertulis di paper (draft v2), dibandingkan dengan hasil hitungan di atas.
# `cek(nama, hitungan, paper, toleransi)`: lulus bila selisihnya di bawah toleransi (default 0,0006 karena paper
# menulis 3 angka desimal). `cek_p(nama, p, batas)`: lulus bila p-value di bawah batas (untuk klaim "p < 0.0001").

# %%
HASIL = []


def cek(nama, hitungan, paper, tol=0.0006):
    HASIL.append({'angka': nama, 'hitungan': float(hitungan), 'paper': paper, 'lulus': abs(hitungan - paper) <= tol})


def cek_p(nama, p, batas=1e-4):
    HASIL.append({'angka': nama, 'hitungan': float(p), 'paper': f'< {batas}', 'lulus': p < batas})


# Tabel 1
PAPER1 = {'XGBoost': (.953, .883, .946), 'LightGBM': (.952, .888, .946), 'CatBoost': (.950, .870, .927),
          'RandomForest': (.934, .830, .936), 'XGBEns': (.953, .885, .948)}
for m, v in PAPER1.items():
    for ds, x in zip(DS, v): cek(f'T1 {m} F1 {ds}', T1.loc[m, NAMA[ds]], x)
cek('F1 dengan port edge', ablasi['edge'].filter(like='DENGAN').iloc[0], .979)
cek('F1 dengan port ton', ablasi['ton'].filter(like='DENGAN').iloc[0], .954)
cek('F1 15 tipe edge', ablasi['edge'].filter(like='jenis').iloc[0], .907)

# Tabel 2 (auroc, fpr95) per dataset: edge, cic, ton
PAPER2 = {'MSP': [(.799, .866), (.811, .777), (.926, .445)], 'Entropy': [(.804, .849), (.811, .777), (.928, .439)],
          'Mahalanobis': [(.801, .628), (.742, .696), (.669, .875)], 'kNN': [(.852, .351), (.797, .708), (.901, .395)],
          'IsolationForest': [(.682, .713), (.626, .889), (.640, .947)], 'MI': [(.879, .453), (.833, .700), (.956, .221)],
          'MI_n8': [(.889, .426), (.841, .678), (.958, .219)], 'Fusion_MI_kNN': [(.897, .336), (.858, .547), (.942, .232)]}
for m, v in PAPER2.items():
    for ds, (a, f) in zip(DS, v):
        cek(f'T2 {m} AUROC {ds}', T2.loc[m, (NAMA[ds], 'auroc')], a)
        cek(f'T2 {m} FPR95 {ds}', T2.loc[m, (NAMA[ds], 'fpr95')], f)
for ds, x in zip(DS, (87.4, 85.3, 80.9)): cek(f'confidence unseen {ds}', conf[ds], x, 0.06)

# uji utama
for ds, x in zip(DS, (.005, .0001, .002)): cek(f'Entropy-MSP selisih {ds}', STAT[(ds, 'Entropy', 'MSP')]['selisih'], x, 0.0006)
for ds, x in zip(DS, (.002, .57, .008)): cek(f'Entropy-MSP p {ds}', STAT[(ds, 'Entropy', 'MSP')]['p'], x, 0.0006 if x < .1 else 0.006)
for ds, x in zip(DS, (.080, .022, .031)):
    cek(f'MI-MSP selisih {ds}', STAT[(ds, 'MI', 'MSP')]['selisih'], x)
    cek_p(f'MI-MSP p {ds}', STAT[(ds, 'MI', 'MSP')]['p'])
    cek_p(f'Fusion-kNN p {ds}', STAT[(ds, 'Fusion_MI_kNN', 'kNN')]['p'])
for ds, s, p in zip(DS, (.019, .025, -.014), (.096, .001, .15)):
    cek(f'Fusion-MI selisih {ds}', STAT[(ds, 'Fusion_MI_kNN', 'MI')]['selisih'], s)
    cek(f'Fusion-MI p {ds}', STAT[(ds, 'Fusion_MI_kNN', 'MI')]['p'], p, 0.006)

# Tabel 3
PAPER3 = {'edge': {'MSP': (.768, .787, .838, .879, .799), 'Entropy': (.770, .789, .848, .902, .804), 'Energy': (.723, .740, .901, None, None)},
          'cic': {'MSP': (.804, .811, .767, .803, .811), 'Entropy': (.804, .811, .768, .803, .811), 'Energy': (.786, .777, .787, None, None)},
          'ton': {'MSP': (.927, .928, .946, .960, .926), 'Entropy': (.928, .931, .944, .960, .928), 'Energy': (.898, .945, .934, None, None)}}
for ds, d in PAPER3.items():
    for sk, v in d.items():
        for m, x in zip(MODEL, v):
            if x is not None: cek(f'T3 {ds} {sk} {m}', T3.loc[(NAMA[ds], sk), m], x)

# Tabel 4 dan uji RF
PAPER4 = {'edge': (.879, .902, .926, .879, .570, .789, .547), 'cic': (.803, .804, .836, .833, .353, .513, .300),
          'ton': (.960, .960, .963, .956, .874, .837, .778)}
for ds, v in PAPER4.items():
    for kol, x in zip(T4.columns, v): cek(f'T4 {ds} {kol}', T4.loc[NAMA[ds], kol], x)
cek('RF MI-MSP selisih edge', STAT_RF[('edge', 'MI_trees', 'MSP')]['selisih'], .047); cek('RF MI-MSP p edge', STAT_RF[('edge', 'MI_trees', 'MSP')]['p'], .01, 0.0016)
cek('RF MI-Entropy selisih edge', STAT_RF[('edge', 'MI_trees', 'Entropy')]['selisih'], .024); cek('RF MI-Entropy p edge', STAT_RF[('edge', 'MI_trees', 'Entropy')]['p'], .08, 0.006)
cek('RF MI-Entropy selisih cic', STAT_RF[('cic', 'MI_trees', 'Entropy')]['selisih'], .032); cek('RF MI-Entropy p cic', STAT_RF[('cic', 'MI_trees', 'Entropy')]['p'], .087, 0.0006)
cek('RF MI-Entropy positif cic', STAT_RF[('cic', 'MI_trees', 'Entropy')]['n_positif'], 15, 0)
cek('RF MI-Entropy selisih ton', STAT_RF[('ton', 'MI_trees', 'Entropy')]['selisih'], .002); cek('RF MI-Entropy p ton', STAT_RF[('ton', 'MI_trees', 'Entropy')]['p'], .21, 0.006)
cek('RF Fusion-MI selisih edge', STAT_RF[('edge', 'Fusion_MItrees_kNN', 'MI_trees')]['selisih'], -.024); cek_p('RF Fusion-MI p edge', STAT_RF[('edge', 'Fusion_MItrees_kNN', 'MI_trees')]['p'])
cek('RF Fusion-MI selisih ton', STAT_RF[('ton', 'Fusion_MItrees_kNN', 'MI_trees')]['selisih'], -.016); cek('RF Fusion-MI p ton', STAT_RF[('ton', 'Fusion_MItrees_kNN', 'MI_trees')]['p'], .11, 0.006)
cek('RF Fusion-MI selisih cic', STAT_RF[('cic', 'Fusion_MItrees_kNN', 'MI_trees')]['selisih'], .007); cek('RF Fusion-MI p cic', STAT_RF[('cic', 'Fusion_MItrees_kNN', 'MI_trees')]['p'], .76, 0.006)

# ukuran ensemble
PAPER_U = {'edge': (.862, .879, .889, .802, .804, .806), 'cic': (.827, .833, .841, .810, .811, .812), 'ton': (.943, .956, .958, .927, .928, .927)}
for ds, v in PAPER_U.items():
    for nama, x in zip(['MI_n2', 'MI_n4', 'MI_n8', 'Entropy_n2', 'Entropy_n4', 'Entropy_n8'], v): cek(f'ukuran {ds} {nama}', UKURAN.loc[nama, NAMA[ds]], x)

# Tabel 5
PAPER5 = {'edge': {'Entropy': (.989, .284, .975, .153, .958, .064), 'MI': (.975, .648, .965, .547, .956, .356),
                   'kNN': (.951, .683, .953, .649, .954, .593), 'Fusion_MI_kNN': (.957, .763, .956, .665, .955, .480)},
          'cic': {'Entropy': (.992, .362, .979, .221, .960, .052), 'MI': (.989, .458, .975, .300, .959, .100),
                  'kNN': (.956, .484, .954, .292, .954, .155), 'Fusion_MI_kNN': (.975, .635, .967, .452, .958, .171)},
          'ton': {'Entropy': (.999, .679, .997, .560, .990, .324), 'MI': (.993, .896, .993, .778, .989, .435),
                  'kNN': (.989, .792, .988, .605, .986, .081), 'Fusion_MI_kNN': (.991, .903, .990, .761, .988, .268)}}
for ds, d in PAPER5.items():
    for m, v in d.items():
        for kol, x in zip(KOL5, v): cek(f'T5 {ds} {m} {kol}', T5.loc[(NAMA[ds], m), kol], x)

# Tabel 6
PAPER6 = {'edge': {'None': (.056, .094, .141), 'MSP': (.034, .069, .114), 'Entropy': (.034, .069, .113), 'MI': (.040, .059, .083), 'kNN': (.051, .065, .084), 'Fusion_MI_kNN': (.047, .061, .080)},
          'cic': {'None': (.055, .094, .141), 'MSP': (.028, .061, .102), 'Entropy': (.029, .061, .102), 'MI': (.032, .061, .098), 'kNN': (.053, .082, .119), 'Fusion_MI_kNN': (.038, .061, .090)},
          'ton': {'None': (.024, .063, .113), 'MSP': (.007, .026, .051), 'Entropy': (.007, .026, .050), 'MI': (.010, .019, .032), 'kNN': (.016, .033, .055), 'Fusion_MI_kNN': (.013, .023, .037)}}
for ds, d in PAPER6.items():
    for m, v in d.items():
        for kol, x in zip(KOL6, v): cek(f'T6 {ds} {m} {kol}', T6.loc[(NAMA[ds], m), kol], x)

# Tabel 7
PAPER7 = {'edge': (.36, .147, .081, .146), 'cic': (.114, .26, .062, .049), 'ton': (.24, .566, .038, .015)}
for ds, v in PAPER7.items():
    for kol, x in zip(T7.columns, v): cek(f'T7 {ds} {kol}', T7.loc[NAMA[ds], kol], x, 0.006)

# port dan tipe
PAPER_P = {'edge': {'MSP': (.768, .884), 'Entropy': (.770, .886), 'kNN': (.852, .881), 'Mahalanobis': (.801, .810)},
           'ton': {'MSP': (.927, .944), 'Entropy': (.928, .945), 'kNN': (.901, .942)}}
for ds, d in PAPER_P.items():
    for m, (a, b) in d.items():
        cek(f'port {ds} {m} tanpa', T_port.loc[(NAMA[ds], m), 'tanpa_port'], a); cek(f'port {ds} {m} dengan', T_port.loc[(NAMA[ds], m), 'dengan_port'], b)
for m, x in {'MSP': .759, 'Entropy': .760, 'Energy': .776, 'Mahalanobis': .734, 'kNN': .785, 'IsolationForest': .616}.items():
    cek(f'tipe edge {m}', T_tipe.loc[m, 'auroc_xgboost'], x)
cek('tipe kNN-MSP p', P_TIPE['p'], .85, 0.006); cek('tipe n pasangan', P_TIPE['n'], 42, 0)

# hasil per grup dan mekanisme yang dikutip di teks
for ds, g, m, x in [('edge', 'MITM', 'MI', .996), ('edge', 'Malware', 'Entropy', .645), ('edge', 'Malware', 'MI', .782), ('edge', 'Malware', 'kNN', .875),
                    ('edge', 'Malware', 'Fusion_MI_kNN', .886), ('cic', 'Mirai', 'kNN', .962), ('cic', 'Mirai', 'Fusion_MI_kNN', .967),
                    ('cic', 'DoS', 'Entropy', .538), ('cic', 'DoS', 'MI', .551), ('cic', 'DoS', 'kNN', .594), ('cic', 'DoS', 'Mahalanobis', .647),
                    ('ton', 'Malware', 'MI', .918), ('ton', 'MITM', 'MI', .990), ('ton', 'Information_Gathering', 'Entropy', .863)]:
    cek(f'per grup {ds} {g} {m}', PERGRUP[ds].loc[g, m], x)
for ds, h, c, fr, rj in [('cic', 'DoS', 'DDoS', .919, .001), ('cic', 'DDoS', 'DoS', .613, .025), ('cic', 'DDoS', 'Mirai', .216, .504),
                         ('edge', 'Malware', 'Benign', .627, .225), ('edge', 'Malware', 'Injection', .265, .607),
                         ('ton', 'Malware', 'Information_Gathering', .444, .222), ('ton', 'Malware', 'Injection', .397, .627),
                         ('ton', 'DoS_DDoS', 'Information_Gathering', .447, .997)]:
    b = NN[ds][(NN[ds].held == h) & (NN[ds].nn_class == c)].iloc[0]
    cek(f'mekanisme {ds} {h}->{c} frac', b.frac_ood, fr); cek(f'mekanisme {ds} {h}->{c} ditolak', b.mi_rej_rate, rj)

# %%
CEK = pd.DataFrame(HASIL)
CEK.to_csv(f'{OUT}/cek_angka_paper.csv', index=False)
print(f'ANGKA PAPER YANG LULUS: {CEK.lulus.sum()} dari {len(CEK)}')
print(CEK[~CEK.lulus].to_string(index=False) if (~CEK.lulus).any() else 'Semua angka paper cocok dengan hasil hitungan.')
print('SELESAI. File di', OUT)
