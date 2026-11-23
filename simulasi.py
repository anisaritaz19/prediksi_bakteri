"""
simulasi.py
===========
Modul inti simulasi pertumbuhan populasi bakteri (TA-07 Pemodelan dan Simulasi).

Model   : Pertumbuhan logistik  dP/dt = r * P * (1 - P/K)
Metode  : Euler, Runge-Kutta 4 (RK4), SciPy solve_ivp, dan solusi analitik (pembanding)
Stock   : P (populasi bakteri)
Flow    : inflow  = r*P            (pertumbuhan, feedback positif / reinforcing)
          outflow = r*P*P/K        (pembatasan kapasitas, feedback negatif / balancing)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, least_squares

DATA_DIR = Path(__file__).resolve().parent / "data"

SKENARIO_DATA = {
    "reconstruction": "Reconstruction (rekonstruksi data latih)",
    "extrapolation": "Extrapolation (prediksi ke t = 400)",
    "completion": "Completion (grid waktu lebih rapat)",
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def muat_data(skenario: str = "reconstruction", bagian: str = "train") -> pd.DataFrame:
    """Muat CSV (kolom: time, population). Kolom indeks tanpa nama dibuang."""
    path = DATA_DIR / f"{skenario}_{bagian}.csv"
    df = pd.read_csv(path)
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed") and c != ""]]
    return normalisasi(df)


def normalisasi(df: pd.DataFrame) -> pd.DataFrame:
    """Pastikan kolom time & population numerik, terurut, tanpa baris kosong."""
    kolom = {c.lower().strip(): c for c in df.columns}
    if "time" not in kolom or "population" not in kolom:
        raise ValueError("CSV harus memiliki kolom 'time' dan 'population'.")
    out = pd.DataFrame(
        {
            "time": pd.to_numeric(df[kolom["time"]], errors="coerce"),
            "population": pd.to_numeric(df[kolom["population"]], errors="coerce"),
        }
    ).dropna()
    out = out.sort_values("time").reset_index(drop=True)
    if len(out) < 5:
        raise ValueError("Data terlalu sedikit (minimal 5 baris valid).")
    return out


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def logistic_model(t, P, r, K):
    """Persamaan diferensial logistik dP/dt = r P (1 - P/K)."""
    return r * P * (1.0 - P / K)


def aliran(P, r, K):
    """Dekomposisi Stock-Flow: (inflow, outflow, net)."""
    inflow = r * P
    outflow = r * P * P / K
    return inflow, outflow, inflow - outflow


def solusi_analitik(t, P0, r, K, t0=0.0):
    """Solusi eksak logistik: P(t) = K / (1 + ((K-P0)/P0) exp(-r (t-t0)))."""
    t = np.asarray(t, dtype=float)
    return K / (1.0 + ((K - P0) / P0) * np.exp(-r * (t - t0)))


# --------------------------------------------------------------------------- #
# Integrator numerik
# --------------------------------------------------------------------------- #
def _grid(t0: float, t_end: float, h: float) -> np.ndarray:
    n = int(math.ceil((t_end - t0) / h - 1e-12))
    g = t0 + h * np.arange(n + 1)
    g[-1] = min(g[-1], t_end) if g[-1] > t_end else g[-1]
    if g[-1] < t_end - 1e-12:
        g = np.append(g, t_end)
    return g


def euler(f, P0, t0, t_end, h, args=()):
    """Metode Euler:  P_{n+1} = P_n + h * f(t_n, P_n). Mengembalikan (t, P)."""
    t = _grid(t0, t_end, h)
    P = np.empty_like(t)
    P[0] = P0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        P[i + 1] = P[i] + dt * f(t[i], P[i], *args)
    return t, P


def rk4(f, P0, t0, t_end, h, args=()):
    """Runge-Kutta orde 4 (klasik). Mengembalikan (t, P)."""
    t = _grid(t0, t_end, h)
    P = np.empty_like(t)
    P[0] = P0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        k1 = f(t[i], P[i], *args)
        k2 = f(t[i] + dt / 2, P[i] + dt * k1 / 2, *args)
        k3 = f(t[i] + dt / 2, P[i] + dt * k2 / 2, *args)
        k4 = f(t[i] + dt, P[i] + dt * k3, *args)
        P[i + 1] = P[i] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t, P


def simulasi(metode, P0, r, K, t_eval, h=1.0, t0=None):
    """
    Simulasikan model logistik lalu kembalikan nilai pada waktu `t_eval`.

    Euler/RK4 dihitung pada grid seragam berjarak h dari t0 sampai max(t_eval),
    lalu diinterpolasi linear ke t_eval (tepat bila t_eval jatuh pada grid).
    """
    t_eval = np.asarray(t_eval, dtype=float)
    t0 = float(t_eval[0]) if t0 is None else float(t0)
    t_end = float(t_eval[-1])
    metode = metode.lower()
    if metode == "euler":
        tg, Pg = euler(logistic_model, P0, t0, t_end, h, args=(r, K))
        return np.interp(t_eval, tg, Pg)
    if metode == "rk4":
        tg, Pg = rk4(logistic_model, P0, t0, t_end, h, args=(r, K))
        return np.interp(t_eval, tg, Pg)
    if metode in ("solve_ivp", "scipy"):
        sol = solve_ivp(
            logistic_model, (t0, t_end), [P0], args=(r, K),
            t_eval=t_eval, method="RK45", rtol=1e-9, atol=1e-12,
        )
        return sol.y[0]
    if metode in ("analitik", "analytic"):
        return solusi_analitik(t_eval, P0, r, K, t0)
    raise ValueError(f"Metode tidak dikenal: {metode}")


# --------------------------------------------------------------------------- #
# Kalibrasi parameter
# --------------------------------------------------------------------------- #
def tebakan_awal(t, P):
    """
    Tebakan awal r dan K dari regresi linear laju per kapita:
        (dP/dt)/P = r - (r/K) P   ->  intersep = r,  K = -intersep/gradien
    """
    dP = np.gradient(P, t)
    gradien, intersep = np.polyfit(P, dP / P, 1)
    r = float(intersep)
    K = float(-intersep / gradien)
    return r, K


def kalibrasi_analitik(t, P):
    """Fit solusi analitik dengan curve_fit (P0 = data pertama). Mengembalikan r, K."""
    P0, t0 = float(P[0]), float(t[0])
    r0, K0 = tebakan_awal(t, P)
    f = lambda tt, r, K: solusi_analitik(tt, P0, r, K, t0)
    popt, _ = curve_fit(f, t, P, p0=[max(r0, 1e-4), max(K0, P.max() * 1.01)],
                        bounds=([1e-6, P.max() * 0.5], [10.0, P.max() * 20]))
    return float(popt[0]), float(popt[1])


def kalibrasi_numerik(t, P, metode="euler", h=1.0):
    """
    Kalibrasi r dan K dengan meminimalkan galat kuadrat antara data dan hasil
    simulasi numerik (metode & h yang dipakai sama dengan simulasi akhir).
    """
    P0, t0 = float(P[0]), float(t[0])
    r0, K0 = tebakan_awal(t, P)

    def resid(x):
        return simulasi(metode, P0, x[0], x[1], t, h=h, t0=t0) - P

    hasil = least_squares(
        resid, x0=[max(r0, 1e-4), max(K0, P.max() * 1.01)],
        bounds=([1e-6, P.max() * 0.5], [10.0, P.max() * 20]),
        x_scale=[0.01, 1.0],
    )
    return float(hasil.x[0]), float(hasil.x[1])


# --------------------------------------------------------------------------- #
# Metrik
# --------------------------------------------------------------------------- #
def metrik(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    e = y_pred - y_true
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "MAE": float(np.mean(np.abs(e))),
        "MAPE (%)": float(np.mean(np.abs(e / y_true)) * 100),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "Galat maks": float(np.max(np.abs(e))),
    }


# --------------------------------------------------------------------------- #
# Besaran turunan (analisis perilaku sistem)
# --------------------------------------------------------------------------- #
def waktu_mencapai(t, P, target):
    """Waktu pertama P melampaui `target` (interpolasi linear); NaN bila tidak tercapai."""
    t = np.asarray(t, float); P = np.asarray(P, float)
    idx = np.where(P >= target)[0]
    if len(idx) == 0:
        return float("nan")
    i = idx[0]
    if i == 0:
        return float(t[0])
    return float(t[i - 1] + (target - P[i - 1]) * (t[i] - t[i - 1]) / (P[i] - P[i - 1]))


def ringkasan_skenario(t, P, r, K):
    """Ringkasan perilaku: nilai akhir, waktu 50%/90% K, laju maksimum (dari simulasi)."""
    dP = np.gradient(P, t)
    i = int(np.argmax(dP))
    return {
        "P akhir": float(P[-1]),
        "Waktu P = 0,5 K": waktu_mencapai(t, P, 0.5 * K),
        "Waktu P = 0,9 K": waktu_mencapai(t, P, 0.9 * K),
        "Laju maks (dP/dt)": float(dP[i]),
        "Waktu laju maks": float(t[i]),
    }


def titik_belok_analitik(P0, r, K, t0=0.0):
    """Waktu saat P = K/2 (titik belok kurva S): t* = t0 + ln((K-P0)/P0)/r."""
    if P0 >= K / 2:
        return float("nan")
    return float(t0 + math.log((K - P0) / P0) / r)
