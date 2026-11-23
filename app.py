"""
app.py - Simulator Streamlit: Prediksi Pertumbuhan Populasi Bakteri
Metode Euler + Model Logistik  (TA-07 Pemodelan dan Simulasi)

Jalankan:  streamlit run app.py
"""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import simulasi as S
from diagram import gambar_stock_flow

st.set_page_config(page_title="Simulasi Populasi Bakteri - Euler & Logistik", page_icon="🦠", layout="wide")
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

METODE = {"Euler": "euler", "Runge-Kutta 4 (RK4)": "rk4", "SciPy solve_ivp (RK45)": "solve_ivp"}


# ----------------------------------------------------------------------------- util
@st.cache_data(show_spinner=False)
def _muat(skenario: str, bagian: str) -> pd.DataFrame:
    return S.muat_data(skenario, bagian)


@st.cache_data(show_spinner=False)
def _kalibrasi(t: tuple, P: tuple, metode: str, h: float):
    t_, P_ = np.array(t), np.array(P)
    if metode == "solve_ivp":
        return S.kalibrasi_analitik(t_, P_)
    return S.kalibrasi_numerik(t_, P_, metode, h)


def _fig(ax_n=1, w=9.5, h=4.3):
    return plt.subplots(1, ax_n, figsize=(w, h))


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Pengaturan")
sumber = st.sidebar.radio("Sumber data", ["Dataset bawaan (dari tugas)", "Unggah CSV sendiri"])

df_train, df_test, nama_data = None, None, ""
if sumber.startswith("Dataset"):
    skn = st.sidebar.selectbox("Skenario dataset", list(S.SKENARIO_DATA), format_func=lambda k: S.SKENARIO_DATA[k])
    df_train, df_test = _muat(skn, "train"), _muat(skn, "test")
    nama_data = skn
else:
    up = st.sidebar.file_uploader("CSV dengan kolom time & population", type=["csv"])
    if up is not None:
        try:
            df_train = S.normalisasi(pd.read_csv(io.BytesIO(up.getvalue())))
            nama_data = up.name
        except Exception as e:  # noqa: BLE001
            st.sidebar.error(f"CSV tidak valid: {e}")
    if df_train is None:
        st.info("Unggah CSV (kolom `time` dan `population`) atau pilih dataset bawaan di sidebar.")
        st.stop()

metode_label = st.sidebar.selectbox("Metode numerik", list(METODE), index=0)
metode = METODE[metode_label]
h = st.sidebar.select_slider("Ukuran langkah h", options=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0], value=1.0,
                             disabled=(metode == "solve_ivp"),
                             help="Tidak dipakai oleh solve_ivp (langkah adaptif).")

t_tr, P_tr = df_train["time"].to_numpy(), df_train["population"].to_numpy()
P0, t0 = float(P_tr[0]), float(t_tr[0])
try:
    r_cal, K_cal = _kalibrasi(tuple(t_tr), tuple(P_tr), metode, float(h))
except Exception as e:  # noqa: BLE001
    st.error(f"Kalibrasi gagal pada data ini: {e}")
    st.stop()

mode = st.sidebar.radio("Parameter", ["Hasil kalibrasi (otomatis)", "Manual"])
sig = f"{nama_data}-{metode}-{h}"
if mode == "Manual":
    r = st.sidebar.slider("r (laju pertumbuhan)", 0.001, max(0.2, 3 * r_cal), float(r_cal), 0.0005, format="%.4f", key=f"r{sig}")
    K = st.sidebar.slider("K (kapasitas dukung)", float(P_tr.max() * 0.5), float(P_tr.max() * 3), float(K_cal), 0.05, key=f"K{sig}")
else:
    r, K = r_cal, K_cal
    st.sidebar.caption(f"r = {r:.5f}  |  K = {K:.4f}  |  P0 = {P0:.5f}")

st.sidebar.markdown("---")
st.sidebar.caption("TA-07 Pemodelan dan Simulasi · Dynamic System Simulation")

# ----------------------------------------------------------------------------- header
st.title("🦠 Simulasi Prediksi Pertumbuhan Populasi Bakteri")
st.caption("Metode Euler dan Model Logistik  ·  dP/dt = r · P · (1 − P/K)")

tabs = st.tabs(["📊 Data & EDA", "📈 Simulasi vs Data", "🔬 Pengaruh langkah h", "🧪 What-If", "🧩 Stock-Flow", "✅ Validasi", "ℹ️ Tentang"])

# ----------------------------------------------------------------------------- tab 1
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah observasi", len(df_train))
    c2.metric("Rentang waktu", f"{t_tr.min():g} – {t_tr.max():g}")
    c3.metric("Populasi min", f"{P_tr.min():.4f}")
    c4.metric("Populasi maks", f"{P_tr.max():.4f}")
    a, b = st.columns([1, 1])
    with a:
        st.dataframe(df_train.head(15), height=330)
        st.write("**Statistik deskriptif populasi**")
        st.dataframe(df_train["population"].describe().to_frame().T)
        st.caption(f"Nilai hilang: {int(df_train.isnull().sum().sum())} · Duplikat populasi: "
                   f"{int(df_train['population'].duplicated().sum())} · Monoton naik: {bool(np.all(np.diff(P_tr) > 0))}")
    with b:
        fig, ax = _fig(1, 6, 4.2)
        ax.plot(t_tr, P_tr, "o-", ms=3, color="#1d4ed8")
        ax.set(title="Data populasi bakteri", xlabel="Waktu (t)", ylabel="Populasi")
        st.pyplot(fig); plt.close(fig)

# ----------------------------------------------------------------------------- tab 2
with tabs[1]:
    horizon = float(t_tr.max())
    if df_test is not None:
        horizon = float(max(horizon, df_test["time"].max()))
    t_grid_end = st.slider("Batas waktu simulasi", float(t_tr.max() * 0.5), float(horizon * 1.5), horizon, 1.0)
    if metode in ("euler", "rk4"):
        fn = S.euler if metode == "euler" else S.rk4
        tg, Pg = fn(S.logistic_model, P0, t0, t_grid_end, h, args=(r, K))
    else:
        tg = np.linspace(t0, t_grid_end, 600)
        Pg = S.simulasi("solve_ivp", P0, r, K, tg, t0=t0)
    pred_tr = S.simulasi(metode, P0, r, K, t_tr, h=h, t0=t0)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].scatter(t_tr, P_tr, s=14, c="#1d4ed8", label="Data asli (train)", zorder=3)
    if df_test is not None and horizon > t_tr.max():
        luar = df_test[df_test["time"] > t_tr.max()]
        ax[0].scatter(luar["time"], luar["population"], s=14, c="#16a34a", label="Data uji (di luar rentang latih)", zorder=3)
    ax[0].plot(tg, Pg, "r-", lw=2, label=f"Simulasi {metode_label}")
    ax[0].axhline(K, color="gray", ls=":", label=f"K = {K:.3f}")
    ax[0].set(title="Data asli vs simulasi", xlabel="Waktu (t)", ylabel="Populasi"); ax[0].legend()
    ax[1].plot(t_tr, P_tr - pred_tr, color="#7c3aed"); ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(title="Residual (data − simulasi)", xlabel="Waktu (t)", ylabel="Residual")
    st.pyplot(fig); plt.close(fig)
    m = S.metrik(P_tr, pred_tr)
    cols = st.columns(5)
    for c, (k, v) in zip(cols, m.items()):
        c.metric(k, f"{v:.6f}" if k != "MAPE (%)" else f"{v:.4f}")
    ring = S.ringkasan_skenario(tg, Pg, r, K)
    st.write("**Ringkasan perilaku sistem (dari hasil simulasi)**")
    st.dataframe(pd.DataFrame([ring]).round(4), hide_index=True)
    st.download_button("⬇️ Unduh hasil simulasi (CSV)", pd.DataFrame({"time": tg, "populasi_simulasi": Pg}).to_csv(index=False),
                       "hasil_simulasi.csv", "text/csv")

# ----------------------------------------------------------------------------- tab 3
with tabs[2]:
    st.write("Bandingkan Euler dengan berbagai ukuran langkah **h** (r dan K tetap, memakai nilai di sidebar). "
             "Pembanding galat adalah solusi analitik logistik.")
    pilihan = st.multiselect("Nilai h", [0.1, 0.5, 1, 2, 5, 10, 15, 30, 50, 75, 100], default=[100, 30, 10, 1])
    if pilihan:
        pa = S.solusi_analitik(t_tr, P0, r, K, t0)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        ax[0].scatter(t_tr, P_tr, s=9, c="k", label="Data", zorder=5)
        ax[0].plot(t_tr, pa, "k--", lw=1, label="Analitik")
        rows = []
        for hh in sorted(pilihan, reverse=True):
            tg, Pg = S.euler(S.logistic_model, P0, t0, float(t_tr.max()), float(hh), args=(r, K))
            ax[0].plot(tg, Pg, "-o" if hh >= 30 else "-", ms=3, label=f"h = {hh}")
            pe = np.interp(t_tr, tg, Pg)
            rows.append({"h": hh, "r·h": r * hh, "RMSE vs analitik": S.metrik(pa, pe)["RMSE"],
                         "RMSE vs data": S.metrik(P_tr, pe)["RMSE"], "P maks": float(pe.max())})
        ax[0].set(title="Euler dengan berbagai h", xlabel="Waktu (t)", ylabel="Populasi"); ax[0].legend(fontsize=8)
        tab = pd.DataFrame(rows).sort_values("h")
        ax[1].loglog(tab["h"], tab["RMSE vs analitik"], "o-", color="#b91c1c")
        ax[1].set(title="Galat vs h (log-log)", xlabel="h", ylabel="RMSE vs analitik")
        st.pyplot(fig); plt.close(fig)
        st.dataframe(tab, hide_index=True)
        if (tab["r·h"] >= 2).any():
            st.warning("Ada h dengan r·h ≥ 2: titik kesetimbangan K tidak lagi stabil pada skema Euler "
                       "(faktor galat |1 − r·h| ≥ 1), sehingga hasil dapat berosilasi.")

# ----------------------------------------------------------------------------- tab 4
with tabs[3]:
    st.write("Ubah parameter terhadap **baseline** (parameter di sidebar) lalu bandingkan dalam satu grafik.")
    c1, c2, c3 = st.columns(3)
    fr = c1.slider("Faktor r (×)", 0.1, 3.0, 0.5, 0.05)
    fK = c2.slider("Faktor K (×)", 0.3, 2.0, 1.0, 0.05)
    fP = c3.slider("Faktor P0 (×)", 0.1, 3.0, 1.0, 0.05)
    c4, c5, c6 = st.columns(3)
    pakai_kejadian = c4.checkbox("Tambahkan kejadian mendadak", value=False)
    t_kej = c5.slider("Waktu kejadian", float(t0), float(t_tr.max()), float(t_tr.max() / 3), 1.0, disabled=not pakai_kejadian)
    f_kej = c6.slider("Faktor populasi setelah kejadian", 0.05, 2.0, 0.5, 0.05, disabled=not pakai_kejadian)
    T = float(t_tr.max())
    hw = h if metode in ("euler", "rk4") else 1.0

    def jalankan(P0_, r_, K_, kejadian=False):
        tg = np.arange(t0, T + 1e-9, hw); Pg = np.empty_like(tg); Pg[0] = P0_; sudah = False
        for i in range(len(tg) - 1):
            if kejadian and (not sudah) and tg[i] >= t_kej:
                Pg[i] *= f_kej; sudah = True
            Pg[i + 1] = Pg[i] + hw * S.logistic_model(tg[i], Pg[i], r_, K_)
        return tg, Pg

    sk = {"Baseline": jalankan(P0, r, K), "Skenario What-If": jalankan(P0 * fP, r * fr, K * fK, pakai_kejadian)}
    fig, ax = _fig(1, 9.5, 4.5)
    ax.plot(*sk["Baseline"], "k", lw=2.4, label="Baseline")
    ax.plot(*sk["Skenario What-If"], "r--", lw=2, label=f"What-If (r×{fr:g}, K×{fK:g}, P0×{fP:g}" + (", ada kejadian)" if pakai_kejadian else ")"))
    ax.set(title="Baseline vs skenario What-If (Euler)", xlabel="Waktu (t)", ylabel="Populasi"); ax.legend()
    st.pyplot(fig); plt.close(fig)
    rows = []
    for nama, (tg, Pg) in sk.items():
        pk = {"Baseline": (P0, r, K), "Skenario What-If": (P0 * fP, r * fr, K * fK)}[nama]
        rows.append({"Skenario": nama, "P0": pk[0], "r": pk[1], "K": pk[2], **S.ringkasan_skenario(tg, Pg, pk[1], pk[2])})
    st.dataframe(pd.DataFrame(rows).round(4), hide_index=True)
    st.caption("“Waktu P = 0,9 K” bernilai NaN bila populasi belum mencapai 90% K dalam rentang simulasi.")

# ----------------------------------------------------------------------------- tab 5
with tabs[4]:
    fig = gambar_stock_flow(r, K)
    st.pyplot(fig)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=170); plt.close(fig)
    st.download_button("⬇️ Unduh diagram (PNG)", buf.getvalue(), "diagram_stock_flow.png", "image/png")
    st.markdown(
        "- **Stock**: P(t), populasi bakteri.\n"
        "- **Inflow** `r·P` — pertumbuhan; membentuk *reinforcing loop* (makin besar P, makin besar pertumbuhan).\n"
        "- **Outflow** `r·P²/K` — pembatasan akibat kapasitas dukung; membentuk *balancing loop* yang menguat saat P mendekati K.\n"
        "- Aliran bersih = Inflow − Outflow = `r·P·(1 − P/K)`.")

# ----------------------------------------------------------------------------- tab 6
with tabs[5]:
    if df_test is None:
        st.info("Validasi data uji tersedia untuk dataset bawaan. Untuk CSV sendiri, lihat tab *Simulasi vs Data*.")
    else:
        st.write(f"Parameter dikalibrasi hanya dari **data latih** ({metode_label}), lalu diuji pada data uji skenario **{nama_data}**.")
        rows = []
        for bag, d in [("train", df_train), ("test", df_test)]:
            pr = S.simulasi(metode, P0, r, K, d["time"].to_numpy(), h=h, t0=t0)
            rows.append({"Bagian": bag, "n": len(d), "t maks": d["time"].max(), **S.metrik(d["population"].to_numpy(), pr)})
        st.dataframe(pd.DataFrame(rows).round(6), hide_index=True)
        st.caption("Catatan: untuk skenario *reconstruction* dan *completion*, data uji berada pada rentang waktu yang sama dengan data latih; "
                   "hanya *extrapolation* yang menguji waktu di luar rentang latih (t > 300).")

# ----------------------------------------------------------------------------- tab 7
with tabs[6]:
    st.markdown(
        """
**Model.** Pertumbuhan logistik `dP/dt = r·P·(1 − P/K)` dengan P populasi, r laju pertumbuhan, K kapasitas dukung.

**Metode Euler.** `P(n+1) = P(n) + h · r · P(n) · (1 − P(n)/K)`. Galat global berorde O(h).

**Kalibrasi.** r dan K dicari dengan *least squares* (SciPy) antara data dan hasil simulasi metode yang dipilih (P0 = observasi pertama).

**Dataset.** Pasangan *time–population* dengan tiga skenario (reconstruction, extrapolation, completion), masing-masing train/test.

**Batasan.** Model logistik adalah model deterministik satu variabel; ia tidak memuat fase lag, kematian, atau fluktuasi acak. 
Data pada tugas ini sangat halus sehingga kecocokan sangat tinggi — tidak otomatis berlaku untuk data laboratorium yang berisik.
        """
    )
