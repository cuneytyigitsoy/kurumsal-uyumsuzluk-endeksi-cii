# -*- coding: utf-8 -*-
"""
SCRIPT D: NİHAİ CII + HİPOTEZ TESTLERİ (5 Tuzak Çözümü Entegre)
Çalıştığı yer: Normal Python (Colab GEREKMEZ, GPU gerekmez)
Girdiler (4 klasörden toplanır):
- 02_DUYGU_SONUC/sikayetvar_duygu.csv
- 02_DUYGU_SONUC/calisan_birlesik_duygu.csv
- 04_VITRIN_SONUC/vitrin_skorlu.csv
- 01_TEMIZ_VERI/sektor_haritasi.csv
Çıktı klasörleri: 03_CII_SONUC/ (CII tabloları + bootstrap CI)
                  06_SONUC_ISTATISTIK/ (hipotez testleri + tam rapor)

Bu script şu dosyaları üretir:
- 03_CII_SONUC/tablo_sirket_cii.csv
- 03_CII_SONUC/tablo_sektorel_cii.csv
- 03_CII_SONUC/bootstrap_ci_sonuclari.csv
- 06_SONUC_ISTATISTIK/spearman_fdr_sonuclari.csv
- 06_SONUC_ISTATISTIK/wilcoxon_sonuclari.csv
- 06_SONUC_ISTATISTIK/kruskal_wallis_sonucu.txt
- 06_SONUC_ISTATISTIK/guc_analizi_fisherz.txt
- 06_SONUC_ISTATISTIK/ISTATISTIK_TAM_RAPOR.txt   <- TEZ METNİNE SADECE BU DOSYADAN SAYI KOPYALAYIN
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
GIRDI_DUYGU = BASE / "02_DUYGU_SONUC"
GIRDI_VITRIN = BASE / "04_VITRIN_SONUC"
GIRDI_TEMIZ = BASE / "01_TEMIZ_VERI"
CIKTI_CII = BASE / "03_CII_SONUC"
CIKTI_ISTAT = BASE / "06_SONUC_ISTATISTIK"
CIKTI_CII.mkdir(parents=True, exist_ok=True)
CIKTI_ISTAT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_BOOTSTRAP = 10000
np.random.seed(RANDOM_SEED)

rapor = []
def yaz(s=""):
    print(s); rapor.append(str(s))

# ============================================================
# 1) VERİYİ YÜKLE
# ============================================================
sikayetvar = pd.read_csv(f"{GIRDI_DUYGU}/sikayetvar_duygu.csv")
calisan = pd.read_csv(f"{GIRDI_DUYGU}/calisan_birlesik_duygu.csv")
vitrin = pd.read_csv(f"{GIRDI_VITRIN}/vitrin_skorlu.csv")
sektor = pd.read_csv(f"{GIRDI_TEMIZ}/sektor_haritasi.csv")

# ============================================================
# 2) TUZAK #2 ÇÖZÜMÜ: çalışan skoru zaten Script B'de hacim ağırlıklı
#    birleştirildi -- burada doğrudan kullanılır.
# ============================================================
calisan_ozet = calisan[["Sirket", "Calisan_Negatiflik_HacimAgirlikli"]].copy()
calisan_ozet["Gercek_Calisan_Memnuniyeti"] = 1 - calisan_ozet["Calisan_Negatiflik_HacimAgirlikli"]

# ============================================================
# 3) TUZAK #1 ÇÖZÜMÜ: Şikayetvar -> mutlak DEĞİL, Z-SCORE (göreli) negatiflik
# ============================================================
sikayet_ozet = sikayetvar.groupby("Sirket")["Negatiflik_Skoru"].mean().reset_index()
sikayet_ozet = sikayet_ozet.rename(columns={"Negatiflik_Skoru": "Musteri_Negatiflik_Ham"})

mu, sigma = sikayet_ozet["Musteri_Negatiflik_Ham"].mean(), sikayet_ozet["Musteri_Negatiflik_Ham"].std()
sikayet_ozet["Musteri_Negatiflik_Z"] = (sikayet_ozet["Musteri_Negatiflik_Ham"] - mu) / sigma
# Z-score'u 0-1 aralığına taşı (yorumlanabilirlik için, ama sıralamayı bozmaz -> Spearman'a etkisi yok)
z_min, z_max = sikayet_ozet["Musteri_Negatiflik_Z"].min(), sikayet_ozet["Musteri_Negatiflik_Z"].max()
sikayet_ozet["Musteri_Negatiflik_Goreli01"] = (sikayet_ozet["Musteri_Negatiflik_Z"] - z_min) / (z_max - z_min)
sikayet_ozet["Gercek_Musteri_Memnuniyeti_Goreli"] = 1 - sikayet_ozet["Musteri_Negatiflik_Goreli01"]

yaz(f"Şikayetvar ham negatiflik: ort={mu:.4f}, ss={sigma:.4f}")
yaz(">>> Tuzak#1 çözümü uygulandı: mutlak skor yerine Z-SCORE (göreli) negatiflik kullanılıyor.")
yaz("    'En az kötü' şirket Gercek_Musteri_Memnuniyeti_Goreli ekseninde en yüksek puanı alır.\n")

# ============================================================
# 4) VİTRİN VERİSİ
# ============================================================
vitrin_ozet = vitrin[["Sirket", "çalışan refahı", "müşteri memnuniyeti", "kariyer gelişimi"]].copy()
vitrin_ozet = vitrin_ozet.rename(columns={
    "çalışan refahı": "Vitrin_Calisan_Refahi",
    "müşteri memnuniyeti": "Vitrin_Musteri_Memnuniyeti",
    "kariyer gelişimi": "Vitrin_Kariyer_Gelisimi",
})

# ============================================================
# 5) BİRLEŞTİR
# ============================================================
df = vitrin_ozet.merge(calisan_ozet[["Sirket", "Gercek_Calisan_Memnuniyeti"]], on="Sirket", how="inner")
df = df.merge(sikayet_ozet[["Sirket", "Gercek_Musteri_Memnuniyeti_Goreli"]], on="Sirket", how="inner")
df = df.merge(sektor, on="Sirket", how="left")

yaz(f"Birleştirilmiş veri seti: N={len(df)} şirket (beklenen 24)")
if len(df) != 24:
    yaz("!!! UYARI: N=24 değil, şirket isim eşleşmesini kontrol edin!")
if df.isna().any().any():
    yaz(f"!!! UYARI: Eksik (NaN) değer var:\n{df[df.isna().any(axis=1)][['Sirket']].to_string()}")

# ============================================================
# 6) CII HESAPLA (clipping ile, zero-lower-bound)
# ============================================================
df["CII_Calisan_Ham"] = df["Vitrin_Calisan_Refahi"] - df["Gercek_Calisan_Memnuniyeti"]
df["CII_Musteri_Ham"] = df["Vitrin_Musteri_Memnuniyeti"] - df["Gercek_Musteri_Memnuniyeti_Goreli"]
df["CII_Calisan"] = df["CII_Calisan_Ham"].clip(lower=0)
df["CII_Musteri"] = df["CII_Musteri_Ham"].clip(lower=0)
df["Toplam_CII"] = (df["CII_Calisan"] + df["CII_Musteri"]) / 2

n_clip_c = (df["CII_Calisan_Ham"] < 0).sum()
n_clip_m = (df["CII_Musteri_Ham"] < 0).sum()
yaz(f"\nClipping uygulanan şirket: Çalışan={n_clip_c}/24, Müşteri={n_clip_m}/24")

tablo_sirket = df[["Sirket", "Sektor", "CII_Calisan", "CII_Musteri", "Toplam_CII"]].sort_values(
    "Toplam_CII", ascending=False).reset_index(drop=True)
tablo_sirket.insert(0, "Sira", np.arange(1, len(tablo_sirket) + 1))
tablo_sirket.to_csv(f"{CIKTI_CII}/tablo_sirket_cii.csv", index=False, encoding="utf-8-sig")
yaz("\n=== ŞİRKET BAZLI CII (azalan Toplam_CII) ===")
yaz(tablo_sirket.round(3).to_string(index=False))

tablo_sektor = df.groupby("Sektor").agg(
    N=("Sirket", "count"), CII_Calisan_Ort=("CII_Calisan", "mean"),
    CII_Musteri_Ort=("CII_Musteri", "mean"), Toplam_CII_Ort=("Toplam_CII", "mean"),
).reset_index().sort_values("Toplam_CII_Ort", ascending=False)
tablo_sektor.to_csv(f"{CIKTI_CII}/tablo_sektorel_cii.csv", index=False, encoding="utf-8-sig")
yaz("\n=== SEKTÖREL CII ORTALAMALARI ===")
yaz(tablo_sektor.round(3).to_string(index=False))

# ============================================================
# 7) SPEARMAN H1-H5 + BENJAMINI-HOCHBERG FDR
# ============================================================
def sr(x, y, etiket):
    rho, p = stats.spearmanr(x, y)
    return {"Hipotez": etiket, "rho": rho, "p": p}

h1 = sr(df["Vitrin_Calisan_Refahi"], df["Gercek_Calisan_Memnuniyeti"], "H1_Calisan_Refahi")
h2 = sr(df["Vitrin_Musteri_Memnuniyeti"], df["Gercek_Musteri_Memnuniyeti_Goreli"], "H2_Musteri_Memnuniyeti")
h3 = sr(df["Vitrin_Kariyer_Gelisimi"], df["Gercek_Calisan_Memnuniyeti"], "H3_Kariyer_Gelisimi")
h4 = sr(df["Gercek_Calisan_Memnuniyeti"], df["Gercek_Musteri_Memnuniyeti_Goreli"], "H4_Gercek_Calisan_vs_Musteri")
h5 = sr(df["CII_Calisan"], df["CII_Musteri"], "H5_CII_Calisan_vs_Musteri")

tum = pd.DataFrame([h1, h2, h3, h4, h5]).sort_values("p").reset_index(drop=True)
m = len(tum)
tum["sira"] = np.arange(1, m + 1)
tum["BH_esik"] = tum["sira"] / m * 0.05
p_sirali = tum["p"].values
p_fdr = np.minimum.accumulate((p_sirali * m / np.arange(1, m + 1))[::-1])[::-1]
tum["p_FDR"] = np.minimum(p_fdr, 1.0)
tum["Anlamli_mi_FDR"] = tum["p_FDR"] < 0.05
tum.to_csv(f"{CIKTI_ISTAT}/spearman_fdr_sonuclari.csv", index=False, encoding="utf-8-sig")
yaz("\n=== SPEARMAN KORELASYON + BENJAMINI-HOCHBERG FDR (H1-H5) ===")
yaz(tum.round(4).to_string(index=False))

# ============================================================
# 8) KRUSKAL-WALLIS (sektörel fark)
# ============================================================
gruplar = [g["Toplam_CII"].values for _, g in df.groupby("Sektor")]
h_stat, kw_p = stats.kruskal(*gruplar)
kw_metin = (f"Kruskal-Wallis H = {h_stat:.4f}, p = {kw_p:.4f}\n"
            f"-> {'Anlamlı sektörel fark VAR' if kw_p < 0.05 else 'Anlamlı sektörel fark YOK'}")
yaz(f"\n=== KRUSKAL-WALLIS (SEKTÖREL) ===\n{kw_metin}")
with open(f"{CIKTI_ISTAT}/kruskal_wallis_sonucu.txt", "w", encoding="utf-8") as f:
    f.write(kw_metin)

# ============================================================
# 9) WILCOXON SIGNED-RANK
# ============================================================
w_c, p_wc = stats.wilcoxon(df["Vitrin_Calisan_Refahi"], df["Gercek_Calisan_Memnuniyeti"], alternative="greater")
w_m, p_wm = stats.wilcoxon(df["Vitrin_Musteri_Memnuniyeti"], df["Gercek_Musteri_Memnuniyeti_Goreli"], alternative="greater")
n_max = len(df) * (len(df) + 1) / 2
wdf = pd.DataFrame([
    {"Boyut": "Çalışan", "W": w_c, "p": p_wc,
     "Vitrin>Gercek": (df["Vitrin_Calisan_Refahi"] > df["Gercek_Calisan_Memnuniyeti"]).sum(),
     "N": len(df), "W_max": n_max},
    {"Boyut": "Müşteri", "W": w_m, "p": p_wm,
     "Vitrin>Gercek": (df["Vitrin_Musteri_Memnuniyeti"] > df["Gercek_Musteri_Memnuniyeti_Goreli"]).sum(),
     "N": len(df), "W_max": n_max},
])
wdf.to_csv(f"{CIKTI_ISTAT}/wilcoxon_sonuclari.csv", index=False, encoding="utf-8-sig")
yaz(f"\n=== WILCOXON SIGNED-RANK ===\n{wdf.round(4).to_string(index=False)}")
yaz("(W teorik maksimuma yakınsa: TAVAN ETKİSİ, veri hatası değildir.)")

# ============================================================
# 10) PAIRED BOOTSTRAP %95 CI
# ============================================================
def paired_bootstrap(x, y, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = len(x); x = np.asarray(x); y = np.asarray(y)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boots[i] = stats.spearmanr(x[idx], y[idx])[0]
    return np.nanpercentile(boots, 2.5), np.nanpercentile(boots, 97.5)

ciftler = {
    "H1": (df["Vitrin_Calisan_Refahi"], df["Gercek_Calisan_Memnuniyeti"]),
    "H2": (df["Vitrin_Musteri_Memnuniyeti"], df["Gercek_Musteri_Memnuniyeti_Goreli"]),
    "H3": (df["Vitrin_Kariyer_Gelisimi"], df["Gercek_Calisan_Memnuniyeti"]),
    "H4": (df["Gercek_Calisan_Memnuniyeti"], df["Gercek_Musteri_Memnuniyeti_Goreli"]),
    "H5": (df["CII_Calisan"], df["CII_Musteri"]),
}
boot_sonuc = []
for etiket, (x, y) in ciftler.items():
    alt, ust = paired_bootstrap(x, y)
    rho0 = stats.spearmanr(x, y)[0]
    boot_sonuc.append({"Hipotez": etiket, "rho": rho0, "CI_alt": alt, "CI_ust": ust,
                         "Sifiri_kapsiyor": alt <= 0 <= ust})
boot_df = pd.DataFrame(boot_sonuc)
boot_df.to_csv(f"{CIKTI_CII}/bootstrap_ci_sonuclari.csv", index=False, encoding="utf-8-sig")
yaz(f"\n=== PAIRED BOOTSTRAP %95 CI (n_boot={N_BOOTSTRAP}) ===\n{boot_df.round(4).to_string(index=False)}")

# ============================================================
# 11) FISHER-Z GÜÇ ANALİZİ (H1-H5 hepsi)
# ============================================================
def fz_guc(rho, n, alpha=0.05):
    zr = np.arctanh(rho); se = 1/np.sqrt(n-3); za = stats.norm.ppf(1-alpha/2)
    return 1 - stats.norm.cdf(za - zr/se) + stats.norm.cdf(-za - zr/se)

def fz_min_rho(n, alpha=0.05, power=0.80):
    za = stats.norm.ppf(1-alpha/2); zb = stats.norm.ppf(power); se = 1/np.sqrt(n-3)
    return np.tanh((za+zb)*se)

n = len(df)
min_rho = fz_min_rho(n)
guc_satirlari = [f"N={n}, minimum saptanabilir rho (alpha=.05, güç=.80) = {min_rho:.4f}"]
for etiket, (x, y) in ciftler.items():
    rho0 = stats.spearmanr(x, y)[0]
    g = fz_guc(rho0, n)
    s = f"{etiket}: rho={rho0:.4f} -> güç={g*100:.1f}%"
    guc_satirlari.append(s)
yaz("\n=== FISHER-Z GÜÇ ANALİZİ ===\n" + "\n".join(guc_satirlari))
with open(f"{CIKTI_ISTAT}/guc_analizi_fisherz.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(guc_satirlari))

# ============================================================
# 12) TAM RAPORU KAYDET
# ============================================================
with open(f"{CIKTI_ISTAT}/ISTATISTIK_TAM_RAPOR.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(rapor))

print(f"\n\n--- SCRIPT D TAMAMLANDI ---")
print(f"TEZ METNİNE YAZILACAK HER SAYI: '{CIKTI_ISTAT}/ISTATISTIK_TAM_RAPOR.txt'")
