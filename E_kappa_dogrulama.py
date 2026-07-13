# -*- coding: utf-8 -*-
"""
SCRIPT E: İNSAN DOĞRULAYICILAR (COHEN'S KAPPA)
Çalıştığı yer: Normal Python (Colab GEREKMEZ, GPU gerekmez)
Girdi klasörü : 01_TEMIZ_VERI/ (Script A çıktısı)
Çıktı klasörü : 05_KAPPA/

Mutfak verisinden (Şikayetvar yorumları) rastgele 200 yorum çekilir.
Bir 'İşletmeci' ve bağımsız bir 'Sosyolog' bu 200 yorumu BAĞIMSIZ olarak
'olumlu' / 'olumsuz' şeklinde etiketler (manuel, Excel/Google Sheets üzerinden).

AŞAMA A: rastgele 200 yorumu örnekleyip etiketleme şablonu üretir
AŞAMA B: iki etiketleyici doldurduktan sonra Cohen's Kappa hesaplar

ÇALIŞTIRMA:
  python E_kappa_dogrulama.py            <- Aşama A (şablon üretir)
  python E_kappa_dogrulama.py hesapla    <- Aşama B (Kappa hesaplar)

GİRDİ (Aşama A): 01_TEMIZ_VERI/sikayetvar_temiz.csv
ÇIKTI (Aşama A): 05_KAPPA/kappa_etiketleme_sablonu.csv
                 (Isletmeci_Etiket / Sosyolog_Etiket sütunları BOŞ, elle doldurulur)

GİRDİ (Aşama B): 05_KAPPA/kappa_etiketleme_sablonu_DOLU.csv (elle doldurulmuş)
ÇIKTI (Aşama B): 05_KAPPA/kappa_sonuc.txt

(Yollar script konumuna göre otomatik hesaplanır — kendi bilgisayarınızda
farklı bir klasör düzeni kullanıyorsanız yalnızca aşağıdaki BASE satırını
düzenlemeniz yeterli.)
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
GIRDI = BASE / "01_TEMIZ_VERI"
CIKTI = BASE / "05_KAPPA"
CIKTI.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
ORNEKLEM_BOYUTU = 200


def asama_a_sablon_uret():
    sikayetvar = pd.read_csv(f"{GIRDI}/sikayetvar_temiz.csv")
    sikayetvar = sikayetvar.dropna(subset=["Metin"])
    sikayetvar = sikayetvar[sikayetvar["Metin"].str.strip() != ""]

    ornek = sikayetvar.sample(n=min(ORNEKLEM_BOYUTU, len(sikayetvar)), random_state=RANDOM_SEED).copy()
    ornek = ornek.reset_index(drop=True)
    ornek["Yorum_ID"] = ornek.index + 1

    ornek["Isletmeci_Etiket"] = ""   # 'olumlu' / 'olumsuz' olarak elle doldurulacak
    ornek["Sosyolog_Etiket"] = ""    # 'olumlu' / 'olumsuz' olarak elle doldurulacak

    cols = ["Yorum_ID", "Sirket", "Baslik", "Metin", "Isletmeci_Etiket", "Sosyolog_Etiket"]
    ornek = ornek[cols]
    ornek.to_excel(f"{CIKTI}/kappa_etiketleme_sablonu.xlsx", index=False)
    print(f"{len(ornek)} yorumluk örneklem oluşturuldu -> {CIKTI}/kappa_etiketleme_sablonu.xlsx")
    print("Lütfen bu Excel dosyasını iki bağımsız etiketleyiciye (İşletmeci, Sosyolog) verin.")
    print("Her ikisi de BİRBİRİNDEN BAĞIMSIZ şekilde 'olumlu' veya 'olumsuz' yazmalı.")
    print(f"Doldurulan dosyayı '{CIKTI}/kappa_etiketleme_sablonu_DOLU.xlsx' adıyla kaydedip Aşama B'yi çalıştırın:")
    print("  python E_kappa_dogrulama.py hesapla")


def asama_b_kappa_hesapla():
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    # İki ayrı dosyayı oku
    df1 = pd.read_excel(f"{CIKTI}/kappa_etiketleme_sablonu-1.xlsx")
    df2 = pd.read_excel(f"{CIKTI}/kappa_etiketleme_sablonu-2.xlsx")
    
    # İkisini birleştirmek için ana iskelet olarak 1. dosyayı kullanalım
    df = df1.copy()
    
    # Kullanıcılar hangi sütunu doldurmuş olursa olsun, o satırdaki dolu cevabı alan bir fonksiyon
    def get_label(row):
        i = str(row.get("Isletmeci_Etiket", "")).strip()
        s = str(row.get("Sosyolog_Etiket", "")).strip()
        if i and i.lower() != "nan" and i.lower() != "none": return i
        if s and s.lower() != "nan" and s.lower() != "none": return s
        return ""

    # 1. Dosyadaki cevapları İşletmeci'ye, 2. Dosyadaki cevapları Sosyolog'a aktar
    df["Isletmeci_Etiket"] = df1.apply(get_label, axis=1)
    df["Sosyolog_Etiket"] = df2.apply(get_label, axis=1)
    
    # Birleştirilmiş temiz veriyi hem Excel hem de arşiv için CSV formatında kaydedelim
    df.to_excel(f"{CIKTI}/kappa_etiketleme_sablonu_DOLU.xlsx", index=False)
    df.to_csv(f"{CIKTI}/kappa_etiketleme_sablonu_DOLU.csv", index=False, encoding="utf-8-sig")
    print(f"BİLGİ: İki dosya başarıyla birleştirilip '{CIKTI}/kappa_etiketleme_sablonu_DOLU.csv' adıyla kaydedildi.")

    eksik = df[(df["Isletmeci_Etiket"].isna()) | (df["Sosyolog_Etiket"].isna()) |
               (df["Isletmeci_Etiket"].astype(str).str.strip() == "") |
               (df["Sosyolog_Etiket"].astype(str).str.strip() == "")]
    if len(eksik) > 0:
        print(f"UYARI: {len(eksik)} satırda eksik etiket var, bunlar hesaplamadan çıkarılacak.")
        df = df.dropna(subset=["Isletmeci_Etiket", "Sosyolog_Etiket"])
        df = df[(df["Isletmeci_Etiket"].astype(str).str.strip() != "") &
                (df["Sosyolog_Etiket"].astype(str).str.strip() != "")]

    isletmeci = df["Isletmeci_Etiket"].astype(str).str.strip().str.lower()
    sosyolog = df["Sosyolog_Etiket"].astype(str).str.strip().str.lower()

    kappa = cohen_kappa_score(isletmeci, sosyolog)
    print(f"\nDeğerlendirilen yorum sayısı: {len(df)}")
    print(f"Cohen's Kappa: {kappa:.4f}")

    if kappa < 0:
        yorum = "Uyumdan daha kötü (sistematik anlaşmazlık)"
    elif kappa < 0.20:
        yorum = "Hafif uyum (slight)"
    elif kappa < 0.40:
        yorum = "Zayıf uyum (fair)"
    elif kappa < 0.60:
        yorum = "Orta uyum (moderate)"
    elif kappa < 0.80:
        yorum = "Yüksek uyum (substantial)"
    else:
        yorum = "Neredeyse mükemmel uyum (almost perfect)"
    print(f"Yorum: {yorum} (Landis & Koch, 1977 skalası)")

    etiketler = sorted(set(isletmeci) | set(sosyolog))
    cm = confusion_matrix(isletmeci, sosyolog, labels=etiketler)
    cm_df = pd.DataFrame(cm, index=etiketler, columns=etiketler)
    print("\nKarışıklık matrisi (İşletmeci satır, Sosyolog sütun):")
    print(cm_df)

    with open(f"{CIKTI}/kappa_sonuc.txt", "w", encoding="utf-8") as f:
        f.write(f"Değerlendirilen yorum sayısı: {len(df)}\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n")
        f.write(f"Yorum: {yorum} (Landis & Koch, 1977 skalası)\n\n")
        f.write("Karışıklık matrisi:\n")
        f.write(cm_df.to_string())

    print(f"\nKaydedildi: {CIKTI}/kappa_sonuc.txt")
    print("Tez metnine bu Kappa değerini ve yorumunu, NLP modelinin insan")
    print("etiketleyicilerle ne kadar uyumlu olduğunun kanıtı olarak ekleyin.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "hesapla":
        asama_b_kappa_hesapla()
    else:
        asama_a_sablon_uret()
        print("\n--- SCRIPT E (AŞAMA A) TAMAMLANDI ---")
