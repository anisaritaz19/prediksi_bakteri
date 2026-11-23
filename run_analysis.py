"""
run_analysis.py - Menjalankan seluruh analisis TA-07 dan menyimpan hasil nyata:
  figures/*.png, results/*.csv, results/hasil.json
Jalankan:  python run_analysis.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import simulasi as S
from diagram import gambar_stock_flow

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"; RES = ROOT / "results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

H_BASE = 1.0           # langkah waktu baseline Euler
hasil = {}

# ----------------------------------------------------------------- data & EDA
tr = S.muat_data("reconstruction", "train")
t, P = tr.time.values, tr.population.values
P0, t0 = float(P[0]), float(t[0])
hasil["eda"] = {
    "n": int(len(tr)), "t_min": float(t.min()), "t_max": float(t.max()),
    "dt": float(t[1] - t[0]), "missing": int(tr.isnull().sum().sum()),
    "duplikat_populasi": int(tr.population.duplicated().sum()),
    "P_min": float(P.min()), "P_max": float(P.max()), "P_mean": float(P.mean()),
    "P_std": float(P.std(ddof=1)), "P_median": float(np.median(P)),
    "monoton_naik": bool(np.all(np.diff(P) > 0)),
}
hasil["eda"]["deskriptif"] = tr.population.describe().to_dict()
hasil["identik"] = {
    "train_recon=train_extrap": bool(S.muat_data("reconstruction", "train").equals(S.muat_data("extrapolation", "train"))),
    "train_recon=train_compl": bool(S.muat_data("reconstruction", "train").equals(S.muat_data("completion", "train"))),
    "test_recon=train": bool(S.muat_data("reconstruction", "test").equals(tr)),
}
hasil["ukuran_split"] = {f"{s}_{k}": int(len(S.muat_data(s, k))) for s in S.SKENARIO_DATA for k in ("train", "test")}

# fig 1: data
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot(t, P, "o-", ms=3, color="#1d4ed8"); ax[0].set(title="Data populasi bakteri (train)", xlabel="Waktu (t)", ylabel="Populasi")
dP = np.gradient(P, t)
ax[1].plot(t, dP, color="#b91c1c"); ax[1].set(title="Laju pertumbuhan dP/dt dari data", xlabel="Waktu (t)", ylabel="dP/dt")
fig.tight_layout(); fig.savefig(FIG / "01_data_eda.png"); plt.close(fig)

# fig 2: umpan balik: laju per kapita vs P
r_reg, K_reg = S.tebakan_awal(t, P)
gr, ic = np.polyfit(P, dP / P, 1)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(P, dP / P, "o", ms=3.5, label="data (dP/dt)/P")
ax.plot(P, gr * P + ic, "r-", label=f"regresi linear: {ic:.5f} {gr:+.5f}·P")
ax.set(title="Bukti umpan balik negatif: laju per kapita menurun terhadap P", xlabel="Populasi P", ylabel="(dP/dt)/P")
ax.legend(); fig.tight_layout(); fig.savefig(FIG / "02_umpan_balik_regresi.png"); plt.close(fig)
hasil["regresi_laju_per_kapita"] = {"intersep_r": float(ic), "gradien": float(gr), "K_implisit": float(-ic / gr)}

# ----------------------------------------------------------------- kalibrasi
r_an, K_an = S.kalibrasi_analitik(t, P)
r_eu, K_eu = S.kalibrasi_numerik(t, P, "euler", H_BASE)
hasil["kalibrasi"] = {
    "P0": P0, "tebakan_regresi": {"r": r_reg, "K": K_reg},
    "analitik_curve_fit": {"r": r_an, "K": K_an},
    "euler_h1": {"r": r_eu, "K": K_eu, "h": H_BASE},
    "titik_belok_analitik": S.titik_belok_analitik(P0, r_an, K_an, t0),
}

# ----------------------------------------------------------------- stock-flow diagram
fig = gambar_stock_flow(r_eu, K_eu)
fig.savefig(FIG / "03_diagram_stock_flow.png", dpi=170); plt.close(fig)

# ----------------------------------------------------------------- perbandingan metode
t_fine = np.linspace(t0, t.max(), 600)
baris = []
sims = {}
for nama, m, h in [("Euler (h=1)", "euler", 1.0), ("RK4 (h=1)", "rk4", 1.0), ("solve_ivp (RK45)", "solve_ivp", 1.0),
                   ("Analitik", "analitik", 1.0)]:
    r_, K_ = (r_eu, K_eu) if m == "euler" else (r_an, K_an)
    pred = S.simulasi(m, P0, r_, K_, t, h=h, t0=t0)
    sims[nama] = (r_, K_, pred)
    baris.append({"Metode": nama, "r": r_, "K": K_, **S.metrik(P, pred)})
tab_metode = pd.DataFrame(baris)
tab_metode.to_csv(RES / "tabel_metode_vs_data.csv", index=False)
hasil["metode_vs_data"] = baris

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax[0].scatter(t, P, s=14, c="#1d4ed8", label="Data asli", zorder=3)
tg, Pg = S.euler(S.logistic_model, P0, t0, t.max(), H_BASE, args=(r_eu, K_eu))
ax[0].plot(tg, Pg, "r-", lw=2, label=f"Simulasi Euler (h={H_BASE:g}, r={r_eu:.5f}, K={K_eu:.3f})")
ax[0].set(title="Data asli vs simulasi Euler (model logistik)", xlabel="Waktu (t)", ylabel="Populasi"); ax[0].legend()
res = P - S.simulasi("euler", P0, r_eu, K_eu, t, h=H_BASE, t0=t0)
ax[1].plot(t, res, color="#7c3aed"); ax[1].axhline(0, color="k", lw=0.8)
ax[1].set(title="Residual (data − simulasi Euler)", xlabel="Waktu (t)", ylabel="Residual")
fig.tight_layout(); fig.savefig(FIG / "04_simulasi_euler_vs_data.png"); plt.close(fig)
hasil["residual_euler"] = {"min": float(res.min()), "max": float(res.max()), "mean": float(res.mean())}

# ----------------------------------------------------------------- pengaruh step size
hs = [100, 75, 50, 30, 15, 10, 5, 2, 1, 0.5, 0.1]
pa = S.simulasi("analitik", P0, r_an, K_an, t, t0=t0)
baris = []
for h in hs:
    pe = S.simulasi("euler", P0, r_an, K_an, t, h=h, t0=t0)
    baris.append({"h": h, "r*h": r_an * h, "RMSE vs analitik": S.metrik(pa, pe)["RMSE"],
                  "RMSE vs data": S.metrik(P, pe)["RMSE"], "Galat maks vs analitik": S.metrik(pa, pe)["Galat maks"],
                  "P maks simulasi": float(pe.max())})
tab_h = pd.DataFrame(baris); tab_h.to_csv(RES / "tabel_pengaruh_h.csv", index=False)
hasil["pengaruh_h"] = baris
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax[0].scatter(t, P, s=10, c="k", label="Data", zorder=5)
for h, c in [(75, "#dc2626"), (30, "#f59e0b"), (10, "#16a34a"), (1, "#2563eb")]:
    tg, Pg = S.euler(S.logistic_model, P0, t0, t.max(), h, args=(r_an, K_an))
    ax[0].plot(tg, Pg, "-o" if h >= 30 else "-", ms=3.5, color=c, label=f"Euler h={h}")
ax[0].set(title="Pengaruh ukuran langkah h (r, K tetap)", xlabel="Waktu (t)", ylabel="Populasi"); ax[0].legend()
ax[1].loglog(tab_h["h"], tab_h["RMSE vs analitik"], "o-", color="#b91c1c", label="RMSE Euler vs analitik")
hh = np.array(hs, float)
ax[1].loglog(hh, tab_h["RMSE vs analitik"].iloc[hs.index(1)] * hh / 1.0, "k--", lw=1, label="kemiringan 1 (orde O(h))")
ax[1].set(title="Galat Euler terhadap h (skala log-log)", xlabel="h", ylabel="RMSE"); ax[1].legend()
fig.tight_layout(); fig.savefig(FIG / "05_pengaruh_step_size.png"); plt.close(fig)
slope = np.polyfit(np.log(tab_h["h"][tab_h["h"] <= 10]), np.log(tab_h["RMSE vs analitik"][tab_h["h"] <= 10]), 1)[0]
hasil["orde_konvergensi_euler"] = float(slope)

# ----------------------------------------------------------------- What-if
base = dict(P0=P0, r=r_eu, K=K_eu)
T_END = 300.0
tt = np.arange(t0, T_END + 1e-9, H_BASE)
skenario = {
    "Baseline": base,
    "r turun 50%": {**base, "r": 0.5 * r_eu},
    "r naik 50%": {**base, "r": 1.5 * r_eu},
    "K turun 25%": {**base, "K": 0.75 * K_eu},
    "K naik 25%": {**base, "K": 1.25 * K_eu},
    "P0 turun 50%": {**base, "P0": 0.5 * P0},
}
baris, kurva = [], {}
for nama, p in skenario.items():
    tg, Pg = S.euler(S.logistic_model, p["P0"], t0, T_END, H_BASE, args=(p["r"], p["K"]))
    kurva[nama] = (tg, Pg)
    ring = S.ringkasan_skenario(tg, Pg, p["r"], p["K"])
    baris.append({"Skenario": nama, "P0": p["P0"], "r": p["r"], "K": p["K"], **ring,
                  "Selisih P akhir vs baseline": float(Pg[-1] - kurva["Baseline"][1][-1])})
tab_wi = pd.DataFrame(baris); tab_wi.to_csv(RES / "tabel_whatif.csv", index=False)
hasil["whatif"] = baris

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
grup = [("Pengaruh laju pertumbuhan r", ["Baseline", "r turun 50%", "r naik 50%"]),
        ("Pengaruh kapasitas dukung K", ["Baseline", "K turun 25%", "K naik 25%"]),
        ("Pengaruh populasi awal P0", ["Baseline", "P0 turun 50%"])]
warna = {"Baseline": "k", "r turun 50%": "#dc2626", "r naik 50%": "#16a34a", "K turun 25%": "#dc2626",
         "K naik 25%": "#16a34a", "P0 turun 50%": "#dc2626"}
for a, (jd, ks) in zip(ax, grup):
    for k in ks:
        a.plot(*kurva[k], color=warna[k], lw=2.4 if k == "Baseline" else 1.8, ls="-" if k == "Baseline" else "--", label=k)
    a.set(title=jd, xlabel="Waktu (t)"); a.legend(loc="lower right")
ax[0].set_ylabel("Populasi")
fig.tight_layout(); fig.savefig(FIG / "06_whatif_analysis.png"); plt.close(fig)

# fig 7: dua skenario ekstra - kejadian mendadak (kematian 50% pada t=150) memakai Euler langkah demi langkah
def euler_kejadian(P0_, r_, K_, h, t_kejadian, faktor):
    tg = np.arange(t0, T_END + 1e-9, h); Pg = np.empty_like(tg); Pg[0] = P0_
    sudah = False
    for i in range(len(tg) - 1):
        if (not sudah) and tg[i] >= t_kejadian:
            Pg[i] *= faktor; sudah = True
        Pg[i + 1] = Pg[i] + h * S.logistic_model(tg[i], Pg[i], r_, K_)
    return tg, Pg
tg1, Pg1 = euler_kejadian(P0, r_eu, K_eu, H_BASE, 100.0, 0.5)
tgb, Pgb = kurva["Baseline"]
idx_ev = int(np.searchsorted(tg1, 100.0)); idx_ev = min(idx_ev, len(tg1) - 1)
fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.plot(tgb, Pgb, "k", lw=2.2, label="Baseline"); ax.plot(tg1, Pg1, "r--", lw=1.9, label="Kematian 50% populasi pada t = 100")
ax.axvline(100, color="gray", ls=":"); ax.set(title="Skenario kejadian mendadak (What-If)", xlabel="Waktu (t)", ylabel="Populasi"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "07_whatif_kejadian.png"); plt.close(fig)
ring_ev = {"P akhir": float(Pg1[-1]), "Selisih P akhir vs baseline": float(Pg1[-1] - Pgb[-1]),
           "Waktu P=0,9K": S.waktu_mencapai(tg1, Pg1, 0.9 * K_eu), "Waktu P=0,9K baseline": S.waktu_mencapai(tgb, Pgb, 0.9 * K_eu),
           "P tepat setelah kejadian": float(Pg1[idx_ev]), "P tepat sebelum (baseline)": float(Pgb[idx_ev])}
hasil["whatif_kejadian"] = ring_ev

# ----------------------------------------------------------------- Validasi 3 skenario data
baris = []
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for a, (skn, ket) in zip(ax, S.SKENARIO_DATA.items()):
    trn = S.muat_data(skn, "train"); tst = S.muat_data(skn, "test")
    tr_t, tr_P = trn.time.values, trn.population.values
    re_, Ke_ = S.kalibrasi_numerik(tr_t, tr_P, "euler", H_BASE)
    for bag, d in [("train", trn), ("test", tst)]:
        pred = S.simulasi("euler", float(tr_P[0]), re_, Ke_, d.time.values, h=H_BASE, t0=float(tr_t[0]))
        baris.append({"Skenario": skn, "Bagian": bag, "n": len(d), "t maks": float(d.time.max()),
                      "r (Euler, dilatih di train)": re_, "K": Ke_, **S.metrik(d.population.values, pred)})
    if skn == "extrapolation":
        luar = tst[tst.time > tr_t.max()]
        pred = S.simulasi("euler", float(tr_P[0]), re_, Ke_, tst.time.values, h=H_BASE, t0=float(tr_t[0]))[tst.time.values > tr_t.max()]
        mm = S.metrik(luar.population.values, pred)
        mm["R2"] = float("nan")   # R2 tidak bermakna: variansi populasi pada t > 300 sangat kecil
        baris.append({"Skenario": skn, "Bagian": "test (t > 300 saja)", "n": len(luar), "t maks": float(luar.time.max()),
                      "r (Euler, dilatih di train)": re_, "K": Ke_, **mm})
    a.scatter(trn.time, trn.population, s=9, c="#9ca3af", label="train")
    a.scatter(tst.time, tst.population, s=7, c="#1d4ed8", label="test", alpha=0.7)
    tg, Pg = S.euler(S.logistic_model, float(tr_P[0]), float(tr_t[0]), float(max(tst.time.max(), tr_t.max())), H_BASE, args=(re_, Ke_))
    a.plot(tg, Pg, "r-", lw=1.8, label="Euler"); a.set(title=skn, xlabel="Waktu (t)", ylabel="Populasi"); a.legend()
fig.tight_layout(); fig.savefig(FIG / "08_validasi_tiga_skenario.png"); plt.close(fig)
tab_val = pd.DataFrame(baris); tab_val.to_csv(RES / "tabel_validasi_skenario.csv", index=False)
hasil["validasi"] = baris

# ----------------------------------------------------------------- simpan
tab_sim = pd.DataFrame({"time": t, "populasi_data": P,
                        "euler_h1": S.simulasi("euler", P0, r_eu, K_eu, t, h=H_BASE, t0=t0),
                        "rk4_h1": S.simulasi("rk4", P0, r_an, K_an, t, h=H_BASE, t0=t0),
                        "solve_ivp": S.simulasi("solve_ivp", P0, r_an, K_an, t, t0=t0),
                        "analitik": S.simulasi("analitik", P0, r_an, K_an, t, t0=t0)})
tab_sim.to_csv(RES / "hasil_simulasi_vs_data.csv", index=False)
with open(RES / "hasil.json", "w", encoding="utf-8") as f:
    json.dump(hasil, f, indent=2, ensure_ascii=False, default=float)
print("Selesai. Ringkasan:")
print(json.dumps({k: hasil[k] for k in ["kalibrasi", "orde_konvergensi_euler", "residual_euler"]}, indent=2))
print(tab_metode.to_string()); print(tab_h.to_string()); print(tab_wi.to_string()); print(tab_val.to_string())
print(json.dumps(hasil["whatif_kejadian"], indent=2)); print(hasil["identik"], hasil["ukuran_split"])
