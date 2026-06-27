import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid")


csv_path = os.path.join(
    "runs", "detect", "runs", "detect", "guclu_model_yeni", "results.csv"
)
output_dir = os.path.join(
    "runs", "detect", "runs", "detect", "guclu_model_yeni", "gorsellestirme_sonuclari"
)
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(csv_path):
    print(f"Hata: {csv_path} dosyası bulunamadı! Lütfen dosya konumunu veya scripti çalıştırdığınız dizini kontrol edin.")
    exit()


df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()


df['metrics/F1-Score'] = 2 * (df['metrics/precision(B)'] * df['metrics/recall(B)']) / (df['metrics/precision(B)'] + df['metrics/recall(B)'] + 1e-8)
df['Tahmini_FPS'] = (1 / (df['time'] / 1000)).round(2)


report_df = df[[
    'epoch', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/F1-Score',
    'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'train/box_loss', 'val/box_loss', 'Tahmini_FPS'
]].copy()
report_df.columns = [
    'Epoch', 'Hassasiyet (Precision)', 'Duyarlılık (Recall)', 'F1-Score',
    'Dogruluk (mAP50)', 'Dogruluk (mAP50-95)', 'Train Box Loss', 'Val Box Loss', 'Hesaplanan FPS'
]
report_df.to_csv(os.path.join(output_dir, "guclu_model_rapor_tablosu.csv"), index=False)


fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Güçlü Model Yeni - Eğitim ve Performans Sonuçları', fontsize=16, fontweight='bold', y=0.98)


axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train', color='#1f77b4', linewidth=2)
axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val', color='#ff7f0e', linestyle='--', linewidth=2)
axes[0, 0].set_title('Kutu Kaybı (Box Loss)')
axes[0, 0].legend()


axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train', color='#1f77b4', linewidth=2)
axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val', color='#ff7f0e', linestyle='--', linewidth=2)
axes[0, 1].set_title('Sınıf Kaybı (Class Loss)')
axes[0, 1].legend()


if 'train/dfl_loss' in df.columns:
    axes[0, 2].plot(df['epoch'], df['train/dfl_loss'], label='Train', color='#1f77b4', linewidth=2)
    axes[0, 2].plot(df['epoch'], df['val/dfl_loss'], label='Val', color='#ff7f0e', linestyle='--', linewidth=2)
axes[0, 2].set_title('Dağılım Kaybı (DFL Loss)')
axes[0, 2].legend()


axes[0, 3].plot(df['epoch'], df['time'], color='#7f7f7f', linewidth=2)
axes[0, 3].set_title('Epok Süresi (Saniye)')


axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='#2ca02c', linewidth=2)
axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='#9467bd', linewidth=2)
axes[1, 0].set_title('Precision & Recall')
axes[1, 0].legend()


axes[1, 1].plot(df['epoch'], df['metrics/F1-Score'], color='#d62728', linewidth=2.5)
axes[1, 1].set_title('F1-Score Değişimi')


axes[1, 2].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', color='#8c564b', linewidth=2.5)
axes[1, 2].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', color='#e377c2', linewidth=2)
axes[1, 2].set_title('Doğruluk Oranları (mAP)')
axes[1, 2].legend()


axes[1, 3].plot(df['epoch'], df['Tahmini_FPS'], color='#17becf', linewidth=2)
axes[1, 3].set_title('Saniyedeki Kare Sayısı (FPS)')


for row in axes:
    for ax in row:
        ax.set_xlabel('Epoch')


plt.tight_layout()
graph_output_path = os.path.join(output_dir, "colab_tarzi_results_guclu_model.png")
plt.savefig(graph_output_path, dpi=300)
plt.close()

print(f"[+] Yeni dosya yoluna göre tablolar ve grafikler başarıyla oluşturuldu!")
print(f"[+] Görsel Çıktı Konumu: {graph_output_path}")