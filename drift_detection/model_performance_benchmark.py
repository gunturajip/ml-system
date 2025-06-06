from sklearn.metrics import accuracy_score
 
# Metrik akurasi pada batch, asumsi sudah diprediksi sebelumnya
accuracy_old = 0.9  # Akurasi pada windows sebelumnya
accuracy_new = 0.7  # Akurasi pada windows baru
 
if accuracy_old - accuracy_new > 0.1:  # Penurunan signifikan
    print("Drift terdeteksi!")