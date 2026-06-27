# akilli-yol-projesi

# 🚗 VERA - 5G Destekli Akıllı Yol Güvenliği Sistemi



> **TEKNOFEST 5G Konumlandırma Yarışması** kapsamında geliştirilen gerçek zamanlı sürücü davranışı analiz sistemi.



VERA, yol güvenliğini artırmak amacıyla geliştirilen yapay zekâ destekli bir sürücü analiz sistemidir. Sistem, iki aşamalı YOLO11 mimarisi kullanarak araçları gerçek zamanlı tespit eder, araç belirlenen mesafeye ulaştığında Turkcell Open Gateway Quality on Demand (QoD) servisini tetikleyerek daha güçlü yapay zekâ modeline geçiş yapar ve sürücü ihlallerini analiz eder.



---



# 📌 Proje Özellikleri



- 🚗 Gerçek zamanlı araç tespiti

- 📱 Telefon kullanımı tespiti

- 🚬 Sigara kullanımı tespiti

- 👀 Sağ / Sol bakış analizi

- 📦 Araç içerisindeki nesnelerin tespiti

- 🔢 OCR ile plaka okuma

- ⚡ Hibrit hız tahmini

- 📶 Turkcell Open Gateway QoD entegrasyonu

- 🎥 Gerçek zamanlı video işleme

- 🖥️ Kullanıcı dostu arayüz



---



# 🏗️ Sistem Mimarisi



```text

Video

↓

YOLO11 Hafif Model

↓

Araç Yaklaşma Analizi

↓

Turkcell Open Gateway QoD

↓

YOLO11 Güçlü Model

↓

Sürücü Analizi

│

├── Telefon

├── Sigara

├── Bakış Yönü

├── Nesne Tespiti

├── OCR

└── Hız Tahmini

↓

Arayüz

```



---



# 🛠 Kullanılan Teknolojiler



- Python 3.12

- OpenCV

- Ultralytics YOLO11

- RapidOCR

- ONNX Runtime

- NumPy

- Lucas-Kanade Optical Flow

- FastAPI

- Turkcell Open Gateway QoD API



---



# 📂 Proje Yapısı



```text

TeknofestModelEgit/



│

├── app.py

├── combined.py

├── train_guclu_yeni.py

├── test_model.py

├── visualize_results.py

├── visualize_results2.py

├── requirements.txt

│

├── dataset/

├── uploaded_videos/

├── processed_videos/

├── runs/

│

├── yolo11n.pt

├── yolo11s.pt

└── yolo26n.pt

```



---



# 📥 Kurulum



Projeyi klonlayın.



```bash

git clone https://github.com/semanurcetintas/akilli-yol-projesi.git

```



Proje klasörüne girin.



```bash

cd akilli-yol-projesi

```



Gerekli kütüphaneleri yükleyin.



```bash

pip install -r requirements.txt

```



---



# 📁 Google Drive İçeriği



GitHub depo boyutunu küçük tutabilmek amacıyla aşağıdaki büyük dosyalar Google Drive üzerinde paylaşılmıştır.



Drive içerisinde;



- 📂 Dataset

- 📂 Eğitim çıktıları (runs)

- 📂 Test videoları

- 📂 İşlenmiş videolar

- 🤖 Eğitilmiş YOLO modelleri





bulunmaktadır.



## Google Drive Bağlantısı



👉 https://drive.google.com/drive/folders/1NPpx1xikffctu8jWwfI7WX610gkfsKgQ?usp=sharing



---



# ▶️ Projeyi Çalıştırma



Gerekli dosyaları Google Drive üzerinden indiriniz.



Aşağıdaki klasörleri proje dizinine yerleştiriniz.



```text

dataset/

runs/

uploaded_videos/

processed_videos/



yolo11n.pt

yolo11s.pt

yolo26n.pt

```



Ardından;







```bash

python combined.py

```



komutui ile uygulamayı çalıştırabilirsiniz. 



---



# 📊 Model Çıktıları



Eğitim sırasında elde edilen;



- Precision

- Recall

- mAP50

- mAP50-95

- F1 Score

- Confusion Matrix

- Precision-Recall Curve

- Loss Grafikleri



Google Drive içerisindeki **runs** klasöründe yer almaktadır.



---
