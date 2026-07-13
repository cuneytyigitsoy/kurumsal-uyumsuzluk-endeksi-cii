# -*- coding: utf-8 -*-
"""
İki Bağımsız Kodlayıcı — Kör Etiketleme (Blind Coding)
N=100 yorum, iki ayrı dosya (Kodlayıcı 1: Tez Yazarı, Kodlayıcı 2: İkinci Kodlayıcı)
Birbirlerinin cevabını görmeden etiketleyecekler.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from pathlib import Path

RANDOM_SEED = 99
N = 100  # 100 yorum
np.random.seed(RANDOM_SEED)

BASE = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE / "01_TEMIZ_VERI" / "indeed_temiz.csv")
df = df[df['Metin'].notna() & (df['Metin'].str.len() > 30)].copy()
df = df.reset_index(drop=True)

# 100 rastgele yorum sec — ikisi de AYNI 100 yorumu değerlendiriyor (kör)
sample = df.sample(n=N, random_state=RANDOM_SEED)[['Sirket', 'Baslik', 'Metin']].copy()
sample = sample.reset_index(drop=True)
sample.insert(0, 'ID', range(1, N + 1))

talimat_rows = [
    'Bu dosyayı açtığınız için teşekkürler.',
    '',
    'Her satırdaki ÇALIŞAN YORUMUNU okuyun.',
    'ETIKET sütununa şunlardan birini yazın:',
    '',
    '   olumlu  → Çalışan deneyiminden genel olarak memnun görünüyor',
    '   olumsuz → Çalışan deneyiminden genel olarak şikayetçi görünüyor',
    '   belirsiz → Karar veremediyseniz veya yorum çok kısa/anlamsız',
    '',
    'ÖNEMLİ: Diğer kodlayıcının dosyasını açmayın!',
    'Bağımsız değerlendirme akademik güvenilirlik için zorunludur.',
    '',
    'Tahmini süre: 45-60 dakika.',
]
talimat_df = pd.DataFrame({'TALİMAT': talimat_rows})

# --- DOSYA 1: Tez Yazarı (Siz) ---
sample1 = sample.copy()
sample1['ETIKET'] = ''  # Siz dolduracaksınız
sample1['NOT'] = ''     # İsteğe bağlı

out1 = str(BASE / "05_KAPPA" / "KODLAYICI_1_TEZ_YAZARI.xlsx")
with pd.ExcelWriter(out1, engine='openpyxl') as writer:
    sample1.to_excel(writer, index=False, sheet_name='Etiketleme')
    talimat_df.to_excel(writer, index=False, sheet_name='TALİMAT')
print(f'Dosya 1 (Tez Yazarı) hazır: {out1}')

# --- DOSYA 2: İkinci Kodlayıcı ---
sample2 = sample.copy()
sample2['ETIKET'] = ''  # İkinci kodlayıcı dolduracak
sample2['NOT'] = ''     # İsteğe bağlı

out2 = str(BASE / "05_KAPPA" / "KODLAYICI_2.xlsx")
with pd.ExcelWriter(out2, engine='openpyxl') as writer:
    sample2.to_excel(writer, index=False, sheet_name='Etiketleme')
    talimat_df.to_excel(writer, index=False, sheet_name='TALİMAT')
print(f'Dosya 2 (İkinci Kodlayıcı) hazır: {out2}')

print(f'\nToplam yorum: {N}')
print('Her iki kodlayıcı da AYNI 100 yorumu görüyor ama birbirinden bağımsız.')
print('Bitince her iki dosyayı bana geri verin — Kappa hesaplayacağım.')
print(f'\nÖnizleme (ilk 5 yorum):')
for _, row in sample.head(5).iterrows():
    print(f'  [{row["ID"]}] {row["Sirket"]} — {str(row["Metin"])[:80]}...')
