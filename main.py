import os
import cv2
import numpy as np
import face_recognition
from sklearn.svm import SVC
import joblib

# Lokasi dataset dan model
path_dataset = r"dataset/dataset/"
path_model = "svm_face_model.pkl"

# Cek apakah path dataset tersedia
if not os.path.exists(path_dataset):
    print(f"Error: Path dataset '{path_dataset}' tidak ditemukan. Mohon periksa kembali!")
    exit()

def muat_encodings_wajah(path_dataset):
    encodings_wajah = []
    nama_wajah = []

    for nama_person in os.listdir(path_dataset):
        path_person = os.path.join(path_dataset, nama_person)

        if os.path.isdir(path_person):
            for nama_gambar in os.listdir(path_person):
                path_gambar = os.path.join(path_person, nama_gambar)
                try:
                    gambar = face_recognition.load_image_file(path_gambar)
                    encoding = face_recognition.face_encodings(gambar)

                    if encoding:
                        encodings_wajah.append(encoding[0])
                        nama_wajah.append(nama_person)
                except Exception as error:
                    print(f"Error pada file {path_gambar}: {error}")

    return encodings_wajah, nama_wajah

def latih_model_svm(encodings, nama):
    print("Melatih model SVM...")
    model_svm = SVC(kernel="linear", probability=True)
    model_svm.fit(encodings, nama)
    joblib.dump(model_svm, path_model)
    print("Model berhasil dilatih dan disimpan!")

def kenali_wajah_svm(frame, model_svm):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lokasi_wajah = face_recognition.face_locations(frame_rgb)
    encoding_wajah = face_recognition.face_encodings(frame_rgb, lokasi_wajah)

    for encoding, lokasi in zip(encoding_wajah, lokasi_wajah):
        nama = "Unknown"

        probabilitas = model_svm.predict_proba([encoding])[0]
        index_terbaik = np.argmax(probabilitas)
        if probabilitas[index_terbaik] > 0.6:
            nama = model_svm.classes_[index_terbaik]

        atas, kanan, bawah, kiri = lokasi
        cv2.rectangle(frame, (kiri, atas), (kanan, bawah), (0, 255, 0), 2)
        cv2.putText(frame, nama, (kiri, atas - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

if not os.path.exists(path_model):
    print("Model belum tersedia. Memuat dataset dan melatih model baru...")
    encoding_dikenal, nama_dikenal = muat_encodings_wajah(path_dataset)
    if encoding_dikenal and nama_dikenal:
        latih_model_svm(encoding_dikenal, nama_dikenal)
    else:
        print("Dataset kosong atau tidak valid!")
        exit()
else:
    print("Memuat model SVM yang tersedia...")
    model_svm = joblib.load(path_model)

kamera = cv2.VideoCapture(0)
kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Tekan 'q' untuk keluar...")

while True:
    berhasil, frame = kamera.read()
    if not berhasil:
        print("Gagal membuka kamera!")
        break

    kenali_wajah_svm(frame, model_svm)
    cv2.imshow("Pengenalan Wajah dengan SVM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()
