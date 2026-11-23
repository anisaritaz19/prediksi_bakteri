# TA-07 · Simulasi Prediksi Pertumbuhan Populasi Bakteri (Metode Euler & Model Logistik)

Mata kuliah **Pemodelan dan Simulasi** — Modul 7 *Dynamic System Simulation* (Mohammad Bayu Anggara, S.Kom., M.Kom.).

Model: `dP/dt = r · P · (1 − P/K)` — **Stock** P (populasi), **Inflow** `r·P` (umpan balik penguat), **Outflow** `r·P²/K` (umpan balik penyeimbang).
Metode: Euler (utama), RK4 dan `scipy.integrate.solve_ivp` (pembanding), solusi analitik (acuan galat).

## Isi repositori
| Berkas / folder | Keterangan |
|---|---|
| `TA07_Simulasi_Bakteri_Euler_Logistik.ipynb` | Notebook lengkap (sudah dieksekusi): model, EDA, simulasi, What-If, validasi |
| `app.py` | Aplikasi Streamlit (7 tab) |
| `simulasi.py` | Modul inti: model, Euler, RK4, solve_ivp, kalibrasi, metrik |
| `diagram.py` | Diagram Stock-Flow |
| `run_analysis.py` | Mereproduksi semua grafik & tabel di `figures/` dan `results/` |
| `data/` | 6 berkas CSV dari tugas (`<skenario>_<train|test>.csv`) |
| `figures/` | Diagram Stock-Flow dan semua grafik hasil |
| `results/` | Tabel hasil (CSV) dan `hasil.json` |
| `laporan/` | Laporan DOCX + PDF (mengikuti template contoh laporan) |

## Menjalankan
```bash
pip install -r requirements.txt
streamlit run app.py                 # aplikasi simulator
python run_analysis.py               # mereproduksi grafik & tabel

pip install -r requirements-notebook.txt
jupyter notebook TA07_Simulasi_Bakteri_Euler_Logistik.ipynb
```

## Deployment
- **Streamlit Community Cloud:** push folder ini ke GitHub → *New app* → pilih repo, file utama `app.py`.
- **Render / platform lain:** *Start command* `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` (lihat `Procfile`).

## Ringkasan hasil (dari eksekusi kode)
- Data: 200 titik, t = 0–300, populasi 2,5186 → 11,9815; tanpa nilai hilang; kurva sigmoid.
- Kalibrasi (solusi analitik): r ≈ 0,02600, K ≈ 12,000. Kalibrasi pada simulasi Euler (h = 1): r = 0,026044, K = 11,9919.
- Euler (h = 1) vs data: RMSE 0,00613, R² = 0,999996. RK4: RMSE 4,0×10⁻⁵. solve_ivp: RMSE 5,0×10⁻⁶.
- Galat Euler ∝ h (kemiringan log-log ≈ 1,00); skema melampaui K bila r·h > 2 (diuji pada h = 100).
- What-If: r menentukan kecepatan, K menentukan level jenuh, P₀ menggeser waktu.

## Catatan penting
- Dataset pada tugas ini **berbeda** dengan `bacterial_growth.csv` pada contoh tutorial modul.
- Data *train* ketiga skenario identik; hanya data *test* yang berbeda (lihat laporan bagian 3.1).
- Isi dahulu **NIM**, tautan video, repositori, dan aplikasi yang diberi sorotan kuning pada laporan DOCX, lalu ekspor ulang ke PDF.
