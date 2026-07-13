# -*- coding: utf-8 -*-
"""
İki bağımsız kodlayıcı arasındaki uyumu (Cohen's Kappa) hesaplama
ve XLM-RoBERTa ile karşılaştırma.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report
from transformers import pipeline
import torch
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
KAPPA_DIR = BASE / "05_KAPPA"
df1 = pd.read_excel(KAPPA_DIR / 'KODLAYICI_1_TEZ_YAZARI.xlsx', sheet_name='Etiketleme')
df2 = pd.read_excel(KAPPA_DIR / 'KODLAYICI_2.xlsx', sheet_name='Etiketleme')

# Etiketleri kucuk harfe cevirip bosluklari temizle
def clean_label(x):
    if pd.isna(x): return 'belirsiz'
    s = str(x).strip().lower()
    if 'olumsuz' in s: return 'olumsuz'
    if 'olumlu' in s: return 'olumlu'
    return 'belirsiz'

df1['L1'] = df1['ETIKET'].apply(clean_label)
df2['L2'] = df2['ETIKET'].apply(clean_label)

# Karsilastirma (belirsiz olanlari cikarabiliriz veya tutabiliriz)
# Akademik olarak sadece olumlu/olumsuz net karar verilenlerde uyuma bakilir
mask = (df1['L1'].isin(['olumlu', 'olumsuz'])) & (df2['L2'].isin(['olumlu', 'olumsuz']))

k1 = df1.loc[mask, 'L1']
k2 = df2.loc[mask, 'L2']

kappa_human = cohen_kappa_score(k1, k2)
acc_human = accuracy_score(k1, k2)

print("=== İNSAN KODLAYICILAR ARASI UYUM (Tez Yazarı vs İkinci Kodlayıcı) ===")
print(f"Net Karşılaştırılan Yorum Sayısı: {len(k1)}")
print(f"Doğruluk (Accuracy): {acc_human:.4f} ({acc_human*100:.1f}%)")
print(f"Cohen's Kappa: {kappa_human:.4f}")
print()

# Uzerinde anlasilan etiketleri XLM-RoBERTa ile test et
consensus = df1.copy()
consensus['Consensus'] = np.where(df1['L1'] == df2['L2'], df1['L1'], 'anlasmazlik')
mask_consensus = consensus['Consensus'].isin(['olumlu', 'olumsuz'])
test_data = consensus[mask_consensus].copy()

print(f"İki kodlayıcının da aynı kararı verdiği net yorum sayısı: {len(test_data)}")
print("Bu yorumlar için XLM-RoBERTa modeli çalıştırılıyor...")

device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli", device=device)

labels = ["olumlu is deneyimi", "olumsuz is deneyimi"]
xlm_preds = []

for idx, row in test_data.iterrows():
    text = str(row['Metin'])[:512]
    res = classifier(text, candidate_labels=labels)
    pred = "olumlu" if "olumlu" in res['labels'][0] else "olumsuz"
    xlm_preds.append(pred)

test_data['XLM_Pred'] = xlm_preds

kappa_model = cohen_kappa_score(test_data['Consensus'], test_data['XLM_Pred'])
acc_model = accuracy_score(test_data['Consensus'], test_data['XLM_Pred'])

print("=== XLM-RoBERTa vs İNSAN UZLAŞMASI ===")
print(f"Doğruluk (Accuracy): {acc_model:.4f} ({acc_model*100:.1f}%)")
print(f"Cohen's Kappa: {kappa_model:.4f}")
print("\nDetaylı Sınıflandırma Raporu (XLM-RoBERTa):")
print(classification_report(test_data['Consensus'], test_data['XLM_Pred']))

# Sonuclari txt'ye kaydet
rapor = f"""=== MANUEL ETİKETLEME VE XLM-RoBERTa DOĞRULAMA RAPORU ===

1. İNSAN KODLAYICILAR ARASI UYUM (N=100)
- Tez Yazarı ve İkinci Kodlayıcı (Kör Kodlama)
- Net Karşılaştırılan (Olumlu/Olumsuz): {len(k1)}
- Kodlayıcılar Arası Doğruluk: {acc_human:.4f} ({acc_human*100:.1f}%)
- Cohen's Kappa: {kappa_human:.4f}

2. XLM-RoBERTa SIFIR-ATIŞ MODELİ DOĞRULAMASI
- İnsan kodlayıcıların üzerinde uzlaştığı net veri seti (N={len(test_data)})
- Model vs İnsan Uzlaşması Doğruluk: {acc_model:.4f} ({acc_model*100:.1f}%)
- Cohen's Kappa: {kappa_model:.4f}
"""
with open(KAPPA_DIR / 'manuel_etiketleme_raporu.txt', 'w', encoding='utf-8') as f:
    f.write(rapor)
print("\nRapor kaydedildi.")
