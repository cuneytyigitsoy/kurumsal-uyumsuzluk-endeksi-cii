# -*- coding: utf-8 -*-
"""
Seçenek A: XLM-RoBERTa Indeed TR Doğrulama
- N=200 Türkçe Indeed yorumu rastgele seç
- XLM-RoBERTa ile skor
- Metni insan etiketleme formatında CSV'ye aktar
- Kappa hesapla (insan etiketleri girilince)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from transformers import pipeline
import torch
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
N_SAMPLE = 200
np.random.seed(RANDOM_SEED)

BASE = Path(__file__).resolve().parent.parent
print("Veri yukleniyor...")
df = pd.read_csv(BASE / "01_TEMIZ_VERI" / "indeed_temiz.csv")

# Sadece Turkce yorumlar (Metin kolonu dolu olanlar)
df = df[df['Metin'].notna() & (df['Metin'].str.len() > 30)].copy()
df = df.reset_index(drop=True)
print(f"Toplam kullanilabilir Indeed TR yorum: {len(df)}")

# N=200 rastgele secim
sample = df.sample(n=N_SAMPLE, random_state=RANDOM_SEED)[['Sirket', 'Metin', 'GenelPuan']].copy()
sample = sample.reset_index(drop=True)
sample.insert(0, 'ID', range(1, N_SAMPLE + 1))

print(f"\nXLM-RoBERTa modeli yukleniyor (bu 1-2 dakika surebilir)...")
device = 0 if torch.cuda.is_available() else -1
print(f"Cihaz: {'GPU' if device == 0 else 'CPU'}")

classifier = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli",
    device=device
)

print(f"Model yuklendi. {N_SAMPLE} yorum skorlanıyor...")
labels = ["olumlu is deneyimi", "olumsuz is deneyimi"]

xlm_labels = []
xlm_scores = []

for i, row in sample.iterrows():
    metin = str(row['Metin'])[:512]  # max 512 token
    try:
        result = classifier(metin, candidate_labels=labels)
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        # Basitlestirilmis etiket
        xlm_label = "olumlu" if "olumlu" in top_label else "olumsuz"
    except Exception as e:
        xlm_label = "belirsiz"
        top_score = 0.0
    xlm_labels.append(xlm_label)
    xlm_scores.append(round(top_score, 4))
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{N_SAMPLE} tamamlandi...")

sample['XLM_Etiket'] = xlm_labels
sample['XLM_Skor'] = xlm_scores

# Genel puan icin referans etiket olustur
# GenelPuan <= 2 → olumsuz, >= 4 → olumlu, 3 → nötr
def puan_to_label(p):
    try:
        p = float(str(p).replace(',', '.'))
        if p <= 2:
            return "olumsuz"
        elif p >= 4:
            return "olumlu"
        else:
            return "notr"
    except:
        return "belirsiz"

sample['GenelPuan_Etiketi'] = sample['GenelPuan'].apply(puan_to_label)

# Sadece olumlu/olumsuz olanlarla Kappa hesapla
karsilastir = sample[sample['GenelPuan_Etiketi'].isin(['olumlu','olumsuz'])].copy()
from sklearn.metrics import cohen_kappa_score, classification_report, accuracy_score

kappa = cohen_kappa_score(karsilastir['GenelPuan_Etiketi'], karsilastir['XLM_Etiket'])
acc = accuracy_score(karsilastir['GenelPuan_Etiketi'], karsilastir['XLM_Etiket'])

print(f"\n=== SONUCLAR ===")
print(f"N karsilastirilan: {len(karsilastir)}")
print(f"Dogruluk (Accuracy): {acc:.4f} ({acc*100:.1f}%)")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"\nDetayli Rapor:")
print(classification_report(karsilastir['GenelPuan_Etiketi'], karsilastir['XLM_Etiket']))

# Kaydet
out_csv = str(BASE / "05_KAPPA" / "xlm_roberta_dogrulama.csv")
sample.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\nDetayli sonuclar: {out_csv}")

# Tez icin ozet rapor
rapor = f"""=== XLM-RoBERTa TÜRKÇE DOĞRULAMA RAPORU ===
Veri: Indeed TR Türkçe çalışan yorumları
N = {N_SAMPLE} (rastgele örneklem, seed={RANDOM_SEED})
Karşılaştırma yöntemi: GenelPuan yıldız skoru referans etiket olarak kullanıldı
(≤2 yıldız = olumsuz, ≥4 yıldız = olumlu, 3 yıldız = nötr — nötr hariç tutuldu)

Karşılaştırılan N = {len(karsilastir)}
Doğruluk (Accuracy) = {acc:.4f} ({acc*100:.1f}%)
Cohen's Kappa = {kappa:.4f}

Landis & Koch (1977) sınıflandırmasına göre:
{'>0.80 → Neredeyse mükemmel (Almost Perfect)' if kappa > 0.80 else
 '0.61-0.80 → Önemli (Substantial)' if kappa > 0.61 else
 '0.41-0.60 → Orta (Moderate)' if kappa > 0.41 else
 '0.21-0.40 → Zayıf (Fair)' if kappa > 0.21 else
 '≤0.20 → Hafif (Slight)'}
"""
print(rapor)
with open(BASE / "05_KAPPA" / "xlm_roberta_kappa_raporu.txt", 'w', encoding='utf-8') as f:
    f.write(rapor)
print("Rapor kaydedildi.")
