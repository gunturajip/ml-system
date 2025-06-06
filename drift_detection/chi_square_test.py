import numpy as np
from scipy.stats import chisquare
 
# Data lama: distribusi kategori fitur
data_lama = np.array([50, 30, 20])  # Frekuensi kategori (misalnya: A, B, C)
 
# Data baru: distribusi kategori fitur
data_baru = np.array([30, 40, 30])  # Frekuensi kategori baru (misalnya: A, B, C)
 
# Normalisasi data baru agar total frekuensinya sama dengan data lama
data_baru_normalized = data_baru * (data_lama.sum() / data_baru.sum())
 
# Melakukan uji Chi-Square
statistic, p_value = chisquare(data_baru_normalized, f_exp=data_lama)
 
# Menampilkan hasil
print(f"Chi-Square Statistic: {statistic}")
print(f"P-Value: {p_value}")
 
# Deteksi Drift
if p_value < 0.05:  # Ambang batas 0.05
    print("Drift terdeteksi! Distribusi kategori telah berubah secara signifikan.")
else:
    print("Tidak ada drift. Distribusi kategori tetap stabil.")