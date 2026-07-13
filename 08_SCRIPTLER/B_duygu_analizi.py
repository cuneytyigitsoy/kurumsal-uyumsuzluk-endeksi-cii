# -*- coding: utf-8 -*-
"""
SCRIPT B: DUYGU ANALİZİ + DISSONANCE (Yıldız-Metin Uyumsuzluğu)
Çalıştığı yer: GOOGLE COLAB (GPU açık)
Girdi klasörü : 01_TEMIZ_VERI/ (Script A çıktısı)
Çıktı klasörü : 02_DUYGU_SONUC/

KURULUM (Colab'da ilk hücre):
!pip install -q transformers torch sentencepiece accelerate

GİRDİLER: indeed_temiz.csv, glassdoor_temiz.csv, sikayetvar_temiz.csv
(Colab'da çalıştırıyorsanız bu 3 dosyayı ve script'i aynı klasöre/oturuma
yükleyin; yerel çalıştırıyorsanız script'i 08_SCRIPTLER/ içinde bırakın,
girdi/çıktı yolları otomatik hesaplanır.)

ÇIKTILAR (02_DUYGU_SONUC klasörüne kaydedilir):
- sikayetvar_duygu.csv
- calisan_birlesik_duygu.csv   (Indeed+Glassdoor, hacim ağırlıklı birleşik)
"""
import pandas as pd
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# Colab'da script tek başına flat bir oturuma yükleniyorsa (girdi CSV'leri
# script ile aynı klasörde), GIRDI "." olarak kalır. Yerel/proje klasörü
# düzeninde çalıştırılıyorsa (08_SCRIPTLER/ altında), üst klasördeki
# 01_TEMIZ_VERI otomatik bulunur.
_local_input = Path(__file__).resolve().parent.parent / "01_TEMIZ_VERI"
GIRDI = str(_local_input) if _local_input.exists() else "."
CIKTI = str(Path(__file__).resolve().parent.parent / "02_DUYGU_SONUC") if _local_input.exists() else "02_DUYGU_SONUC"
Path(CIKTI).mkdir(parents=True, exist_ok=True)

DEVICE = 0 if torch.cuda.is_available() else -1
print("Cihaz:", "GPU" if DEVICE == 0 else "CPU (YAVAŞ olur, Colab'da GPU açın)")
BATCH_SIZE = 16
LAMBDA_DISSONANCE = 0.3  # Tuzak#5 ağırlığı; 0.2-0.4 arası duyarlılık analizi önerilir

# ============================================================
# YARDIMCI: DISSONANCE SKORU (Tuzak #5)
# Dissonance = max(0, Normalize(Yıldız) - NLP_Negatiflik_Skoru_Tersi)
# Yıldız YÜKSEK ama metin NEGATİF ise -> yüksek dissonance (gizli mağduriyet sinyali)
# Yıldız EKSİK (NaN) ise -> dissonance=0, formül çökmez, sadece NLP skoru kullanılır
# ============================================================
def hesapla_dissonance(genel_puan_1_5, nlp_negatiflik_0_1):
    """genel_puan_1_5: 1-5 arası veya NaN. nlp_negatiflik_0_1: 0-1 arası (0=olumlu,1=olumsuz)"""
    if pd.isna(genel_puan_1_5):
        return 0.0
    puan_norm = (genel_puan_1_5 - 1) / 4.0   # 1-5 -> 0-1 (1=kötü, 5=iyi olarak normalize)
    puan_norm = min(max(puan_norm, 0), 1)
    fark = puan_norm - (1 - nlp_negatiflik_0_1)  # ikisi de "iyi=1" yönünde olmalı
    # Aslında istediğimiz: yıldız iyi (puan_norm yüksek) AMA metin kötü (negatiflik yüksek)
    dissonance = max(0.0, puan_norm - (1 - nlp_negatiflik_0_1))
    return dissonance

# ============================================================
# 1) ŞİKAYETVAR -> BERTurk (Şikayetvar'da yıldız puanı yok, dissonance uygulanmaz)
# ============================================================
print("\n=== ŞİKAYETVAR: BERTurk Duygu Analizi ===")
# NOT: Şikayetvar ham verisinde yıldız puanı (GenelPuan) sütunu YOKTUR.
# Bu nedenle Dissonance formülü (Tuzak#5) burada UYGULANMAZ -- bu kasıtlıdır,
# eksiklik değildir. Tez metninde açıkça belirtin: "Şikayetvar platformu
# puanlama sistemi içermediği için dissonance analizi yalnızca
# Indeed/Glassdoor için uygulanmıştır."
berturk_name = "savasy/bert-base-turkish-sentiment-cased"
tok = AutoTokenizer.from_pretrained(berturk_name)
mdl = AutoModelForSequenceClassification.from_pretrained(berturk_name)
sent_pipe = pipeline("sentiment-analysis", model=mdl, tokenizer=tok, device=DEVICE,
                      truncation=True, max_length=512)

sikayetvar = pd.read_csv(f"{GIRDI}/sikayetvar_temiz.csv")
sikayetvar["Metin"] = sikayetvar["Metin"].fillna("").astype(str)
texts = sikayetvar["Metin"].tolist()

etiketler, skorlar = [], []
for i in range(0, len(texts), BATCH_SIZE):
    batch = [t if t.strip() else "." for t in texts[i:i + BATCH_SIZE]]
    for r in sent_pipe(batch):
        etiketler.append(r["label"]); skorlar.append(r["score"])
    if (i // BATCH_SIZE) % 50 == 0:
        print(f"  İşlenen: {i+len(batch)}/{len(texts)}")

sikayetvar["Duygu_Etiket"] = etiketler
sikayetvar["Duygu_Skor"] = skorlar
sikayetvar["Negatiflik_Skoru"] = np.where(
    sikayetvar["Duygu_Etiket"].str.lower().str.contains("neg"),
    sikayetvar["Duygu_Skor"], 1 - sikayetvar["Duygu_Skor"])

dagilim = sikayetvar["Duygu_Etiket"].value_counts(normalize=True) * 100
print(f"\nDuygu dağılımı:\n{dagilim.round(1)}")
if dagilim.max() > 90:
    print(">>> UYARI: %90+ negatif -- Şikayetvar'ın platform doğası gereği BEKLENEN "
          "bir durumdur (Tuzak#1). D script'inde Z-score ile göreli değerlendirilecek.")

sikayetvar.to_csv(f"{CIKTI}/sikayetvar_duygu.csv", index=False, encoding="utf-8-sig")
print(f"Kaydedildi: {CIKTI}/sikayetvar_duygu.csv")

# ============================================================
# 2) INDEED + GLASSDOOR -> XLM-R Zero-Shot + Dissonance + Hacim Ağırlıklı Birleştirme
# ============================================================
print("\n=== ÇALIŞAN VERİSİ: XLM-R Zero-Shot + Dissonance ===")
zshot = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli",
                  device=DEVICE, truncation=True, max_length=512)
LABELS = ["olumsuz iş deneyimi", "olumlu iş deneyimi"]

def analiz_et(df, isim):
    df = df.copy()
    df["Metin"] = df["Metin"].fillna("").astype(str)
    texts = df["Metin"].tolist()
    olumsuz_l, olumlu_l = [], []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [t if t.strip() else "." for t in texts[i:i + BATCH_SIZE]]
        for t in batch:
            res = zshot(t, candidate_labels=LABELS, multi_label=True)
            sc = dict(zip(res["labels"], res["scores"]))
            olumlu_l.append(sc.get("olumlu iş deneyimi", np.nan))
            olumsuz_l.append(sc.get("olumsuz iş deneyimi", np.nan))
        if (i // BATCH_SIZE) % 20 == 0:
            print(f"  [{isim}] İşlenen: {i+len(batch)}/{len(texts)}")
    df["Olumlu_Skor"] = olumlu_l
    df["Olumsuz_Skor"] = olumsuz_l
    # Dissonance (Tuzak #5): GenelPuan varsa kullan, yoksa otomatik 0
    if "GenelPuan" in df.columns:
        df["Dissonance"] = df.apply(
            lambda r: hesapla_dissonance(r["GenelPuan"], r["Olumsuz_Skor"]), axis=1)
    else:
        df["Dissonance"] = 0.0
    df["Olumsuz_Skor_Nihai"] = (df["Olumsuz_Skor"] + LAMBDA_DISSONANCE * df["Dissonance"]).clip(0, 1)
    return df

indeed = pd.read_csv(f"{GIRDI}/indeed_temiz.csv")
glassdoor = pd.read_csv(f"{GIRDI}/glassdoor_temiz.csv")

indeed = analiz_et(indeed, "Indeed")
glassdoor = analiz_et(glassdoor, "Glassdoor")

indeed["Kaynak"] = "INDEED"
glassdoor["Kaynak"] = "GLASSDOOR"

ortak_kolonlar = ["Sirket", "Metin", "GenelPuan", "Olumsuz_Skor", "Olumlu_Skor",
                   "Dissonance", "Olumsuz_Skor_Nihai", "Kaynak"]
calisan_birlesik = pd.concat([
    indeed[[c for c in ortak_kolonlar if c in indeed.columns]],
    glassdoor[[c for c in ortak_kolonlar if c in glassdoor.columns]],
], ignore_index=True)

# ============================================================
# 3) HACİM AĞIRLIKLI ŞİRKET DÜZEYİ BİRLEŞTİRME (Tuzak #2 çözümü)
# Calisan_Skoru(şirket) = (n_indeed*Skor_indeed + n_glassdoor*Skor_glassdoor) / (n_indeed+n_glassdoor)
# Bu, basit %50-%50 ortalamanın aksine, az yorumlu platformun çok yorumlu
# platformu domine etmesini engeller (ters-varyans ağırlıklandırma mantığı).
# ============================================================
sirket_ozet = calisan_birlesik.groupby(["Sirket", "Kaynak"])["Olumsuz_Skor_Nihai"].agg(
    ["mean", "count"]).reset_index()
sirket_ozet.columns = ["Sirket", "Kaynak", "Ort_Negatiflik", "N"]

hacim_agirlikli = []
for sirket, grup in sirket_ozet.groupby("Sirket"):
    toplam_n = grup["N"].sum()
    agirlikli_ort = (grup["Ort_Negatiflik"] * grup["N"]).sum() / toplam_n
    satir = {"Sirket": sirket, "Calisan_Negatiflik_HacimAgirlikli": agirlikli_ort, "Toplam_N": toplam_n}
    for _, r in grup.iterrows():
        satir[f"N_{r['Kaynak']}"] = r["N"]
        satir[f"Ort_{r['Kaynak']}"] = r["Ort_Negatiflik"]
    hacim_agirlikli.append(satir)

calisan_sirket_duzeyi = pd.DataFrame(hacim_agirlikli)
calisan_sirket_duzeyi.to_csv(f"{CIKTI}/calisan_birlesik_duygu.csv", index=False, encoding="utf-8-sig")
print(f"\nKaydedildi: {CIKTI}/calisan_birlesik_duygu.csv")
print(calisan_sirket_duzeyi[["Sirket", "Toplam_N", "Calisan_Negatiflik_HacimAgirlikli"]]
      .sort_values("Calisan_Negatiflik_HacimAgirlikli").to_string(index=False))

print("\n--- SCRIPT B TAMAMLANDI ---")
