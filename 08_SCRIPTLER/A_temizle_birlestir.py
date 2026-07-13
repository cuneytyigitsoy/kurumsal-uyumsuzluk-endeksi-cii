# -*- coding: utf-8 -*-
"""
SCRIPT A: HAM VERİYİ TEMİZLE + BİRLEŞTİR
Çalıştığı yer: Normal Python (Colab GEREKMEZ, GPU gerekmez)
Girdi klasörü : <proje_kök>/00_HAM_VERI
Çıktı klasörü : <proje_kök>/01_TEMIZ_VERI
(Yollar script konumuna göre otomatik hesaplanır, elle düzenlemeye gerek yok.)

GİRDİLER (00_HAM_VERI içine koyun):
- indeed_ham_veri_birlesik.csv
- glassdoor_ham_veri_birlesik.csv
- sikayetvar_ham_veri_birlesik.csv
- vitrin_ham_veri_birlesik.csv

ÇIKTILAR (01_TEMIZ_VERI klasörüne kaydedilir):
- indeed_temiz.csv
- glassdoor_temiz.csv
- sikayetvar_temiz.csv
- vitrin_temiz.csv
- sektor_haritasi.csv
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
GIRDI = BASE / "00_HAM_VERI"
CIKTI = BASE / "01_TEMIZ_VERI"
CIKTI.mkdir(parents=True, exist_ok=True)

def clean_text(t):
    """KAYIPSIZ temizlik: sadece yapısal gürültü (fazla boşluk, satır sonu).
    Argo, noktalama, stopword SİLİNMEZ -- duygu yoğunluğu korunur."""
    if pd.isna(t):
        return t
    t = str(t)
    t = t.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def parse_puan(x):
    """'1,0' / 1.0 / NaN -> float veya NaN (asla hata vermez)."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).replace(",", "."))
    except ValueError:
        return np.nan

# ============================================================
# 1) OKU
# ============================================================
indeed = pd.read_csv(GIRDI / "indeed_ham_veri_birlesik.csv")
glassdoor = pd.read_csv(GIRDI / "glassdoor_ham_veri_birlesik.csv")
sikayetvar = pd.read_csv(GIRDI / "sikayetvar_ham_veri_birlesik.csv")
vitrin = pd.read_csv(GIRDI / "vitrin_ham_veri_birlesik.csv")

print(f"Okunan: Indeed={len(indeed)}, Glassdoor={len(glassdoor)}, "
      f"Sikayetvar={len(sikayetvar)}, Vitrin={len(vitrin)}")

# ============================================================
# 2) ŞİRKET İSMİ STANDARDİZASYONU (Firma_Adi -> Sirket, hepsi BÜYÜK_ALT_ÇİZGİ)
# ============================================================
def standart_isim(s):
    if pd.isna(s):
        return s
    s = str(s).strip().upper()
    s = s.replace("İ", "I").replace(" ", "_")
    return s

for df in [indeed, glassdoor, sikayetvar, vitrin]:
    df["Sirket"] = df["Firma_Adi"].apply(standart_isim)

kanon_sirketler = sorted(set(indeed["Sirket"]))
print(f"\nKanonik şirket sayısı: {len(kanon_sirketler)}")
for ad, df, isim in [("Indeed", indeed, "Indeed"), ("Glassdoor", glassdoor, "Glassdoor"),
                       ("Sikayetvar", sikayetvar, "Sikayetvar"), ("Vitrin", vitrin, "Vitrin")]:
    eksik = set(kanon_sirketler) - set(df["Sirket"])
    fazla = set(df["Sirket"]) - set(kanon_sirketler)
    if eksik or fazla:
        print(f"  !!! {isim}: eksik={eksik}, fazla={fazla}")
    else:
        print(f"  OK {isim}: 24 şirket tam eşleşti")

# ============================================================
# 3) GenelPuan SAYISALLAŞTIR (virgül/nokta karışıklığını çöz)
# ============================================================
for df in [indeed, glassdoor]:
    if "GenelPuan" in df.columns:
        df["GenelPuan"] = df["GenelPuan"].apply(parse_puan)

# ============================================================
# 4) METİN TEMİZLİĞİ (kayıpsız)
# ============================================================
for col in ["Metin", "Baslik", "Arti", "Eksi"]:
    if col in indeed.columns:
        indeed[col] = indeed[col].apply(clean_text)
    if col in glassdoor.columns:
        glassdoor[col] = glassdoor[col].apply(clean_text)
for col in ["Metin", "Baslik"]:
    sikayetvar[col] = sikayetvar[col].apply(clean_text)
vitrin["Metin"] = vitrin["Metin"].apply(clean_text)

# ============================================================
# 5) MÜKERRER KAYIT TEMİZLİĞİ
# ============================================================
def dedup(df, isim, subset):
    once = len(df)
    df2 = df.drop_duplicates(subset=subset)
    print(f"  {isim}: {once} -> {len(df2)} ({once - len(df2)} mükerrer silindi)")
    return df2

print("\nMükerrer kayıt temizliği:")
indeed = dedup(indeed, "Indeed", ["Sirket", "Metin", "Tarih", "Pozisyon"])
glassdoor = dedup(glassdoor, "Glassdoor", ["Sirket", "Metin", "Tarih", "Pozisyon"])
sikayetvar = dedup(sikayetvar, "Sikayetvar", ["Sirket", "Metin", "Baslik"])
vitrin = dedup(vitrin, "Vitrin", ["Sirket", "Metin"])

# ============================================================
# 6) KAYDET
# ============================================================
indeed.to_csv(CIKTI / "indeed_temiz.csv", index=False, encoding="utf-8-sig")
glassdoor.to_csv(CIKTI / "glassdoor_temiz.csv", index=False, encoding="utf-8-sig")
sikayetvar.to_csv(CIKTI / "sikayetvar_temiz.csv", index=False, encoding="utf-8-sig")
vitrin.to_csv(CIKTI / "vitrin_temiz.csv", index=False, encoding="utf-8-sig")

# ============================================================
# 7) SEKTÖR HARİTASI (elle doğrulanmış, 24 şirket -> 7 sektör)
# ============================================================
# ÖNEMLİ - GİZLİLİK: Gerçek şirket isimleri bu script içinde TUTULMAZ
# (etik onay + tez anonimleştirmesiyle uyumlu olması için). Eşleştirme,
# yalnızca sizin bilgisayarınızda kalması gereken, Git'e hiç eklenmeyen
# 00_HAM_VERI/sektor_haritasi_KAYNAK.csv dosyasından okunur.
#
# O dosyanın formatı (2 sütun, başlık satırıyla):
#   Sirket,Sektor
#   ŞİRKET_ADI_1,Perakende
#   ŞİRKET_ADI_2,Finans
#   ... (24 satır, standart_isim() ile aynı BÜYÜK_ALT_ÇİZGİ formatında)
harita_dosyasi = GIRDI / "sektor_haritasi_KAYNAK.csv"
if not harita_dosyasi.exists():
    raise FileNotFoundError(
        f"'{harita_dosyasi}' bulunamadı. Gerçek şirket->sektör eşleştirmesini "
        "içeren bu dosyayı kendi bilgisayarınızda 00_HAM_VERI/ klasörüne "
        "koyun (bu dosya asla Git'e/GitHub'a eklenmemelidir, .gitignore "
        "içinde zaten hariç tutulmuştur)."
    )
sektor_df = pd.read_csv(harita_dosyasi)
# Not: Türkçe "İ" harfi standart_isim() içinde "I"ya çevrilir (örn. "İ" içeren
# şirket adlarında farklı yazımlar aynı koda düşsün diye) -- eşleşme sorunu
# çıkarsa CSV'deki Sirket sütununu bu dönüşüme göre kontrol edin.
eksik = set(kanon_sirketler) - set(sektor_df["Sirket"])
if eksik:
    print(f"\n!!! SEKTÖR HARİTASINDA EKSİK ŞİRKET(LER): {eksik} -- elle ekleyin!")
sektor_df.to_csv(CIKTI / "sektor_haritasi.csv", index=False, encoding="utf-8-sig")

print(f"\n--- SCRIPT A TAMAMLANDI ---")
print(f"Çıktılar '{CIKTI}' klasöründe: indeed_temiz.csv, glassdoor_temiz.csv, "
      f"sikayetvar_temiz.csv, vitrin_temiz.csv, sektor_haritasi.csv")
