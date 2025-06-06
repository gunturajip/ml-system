from scipy.stats import ks_2samp
 
# Contoh dua distribusi data
old_data = [1, 2, 3, 4, 5, 6]
new_data = [4, 5, 6, 7, 8, 9]
 
# Uji KS
statistic, p_value = ks_2samp(old_data, new_data)
if p_value < 0.05:  # Tingkat signifikansi 0.05
    print("Drift terdeteksi!")
else:
    print("Tidak ada drift.")