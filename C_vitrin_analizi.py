# -*- coding: utf-8 -*-
"""
SCRIPT C: VİTRİN ANALİZİ (Kayıpsız Kayan Pencere)
Çalıştığı yer: GOOGLE COLAB (GPU açık)
Girdi klasörü : 01_TEMIZ_VERI/ (Script A çıktısı)
Çıktı klasörü : 04_VITRIN_SONUC/

KURULUM: !pip install -q transformers torch sentencepiece accelerate
GİRDİ: vitrin_temiz.csv
ÇIKTI: vitrin_skorlu.csv
(Colab'da flat oturuma yükleniyorsa girdi/çıktı script ile aynı klasörde
aranır; yerel proje düzeninde 08_SCRIPTLER/ altında çalıştırılırsa üst
klasördeki 01_TEMIZ_VERI / 04_VITRIN_SONUC otomatik bulunur.)
"""
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from pathlib import Path

_local_input = Path(__file__).resolve().parent.parent / "01_TEMIZ_VERI"
GIRDI = str(_local_input) if _local_input.exists() else "."
CIKTI = str(Path(__file__).resolve().parent.parent / "04_VITRIN_SONUC") if _local_input.exists() else "04_VITRIN_SONUC"
Path(CIKTI).mkdir(parents=True, exist_ok=True)

DEVICE = 0 if torch.cuda.is_available() else -1
print("Cihaz:", "GPU" if DEVICE == 0 else "CPU")

zshot = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli",
                  device=DEVICE, truncation=True, max_length=512)
LABELS = ["çalışan refahı", "müşteri memnuniyeti", "kariyer gelişimi"]

# Tuzak #3 çözümü: hiçbir kelime atılmaz, 400/200 örtüşmeli kayan pencere
WINDOW_SIZE = 400
STRIDE = 200

def sliding_windows(text, window_size=WINDOW_SIZE, stride=STRIDE):
    words = text.split()
    if len(words) <= window_size:
        return [text]
    windows = []
    for start in range(0, len(words), stride):
        chunk = words[start:start + window_size]
        if not chunk:
            break
        windows.append(" ".join(chunk))
        if start + window_size >= len(words):
            break
    return windows

vitrin = pd.read_csv(f"{GIRDI}/vitrin_temiz.csv")
vitrin["Metin"] = vitrin["Metin"].fillna("").astype(str)

rows = []
for idx, row in vitrin.iterrows():
    sirket, text = row["Sirket"], row["Metin"]
    windows = sliding_windows(text)
    print(f"{sirket}: {len(windows)} pencere ({len(text.split())} kelime)")
    pencere_skor = {lbl: [] for lbl in LABELS}
    for w in windows:
        if not w.strip():
            continue
        res = zshot(w, candidate_labels=LABELS, multi_label=True)
        for lbl, sc in zip(res["labels"], res["scores"]):
            pencere_skor[lbl].append(sc)
    ortalama = {lbl: (np.mean(v) if v else np.nan) for lbl, v in pencere_skor.items()}
    ortalama["Sirket"] = sirket
    ortalama["Pencere_Sayisi"] = len(windows)
    rows.append(ortalama)

vitrin_skorlu = pd.DataFrame(rows)
vitrin_skorlu.to_csv(f"{CIKTI}/vitrin_skorlu.csv", index=False, encoding="utf-8-sig")
print(f"\nKaydedildi: {CIKTI}/vitrin_skorlu.csv")
print(vitrin_skorlu.round(3).to_string(index=False))
print("\n--- SCRIPT C TAMAMLANDI ---")
