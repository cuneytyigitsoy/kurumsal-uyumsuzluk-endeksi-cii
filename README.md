# Kurumsal Uyumsuzluk Endeksi (CII) — Analiz Pipeline'ı

Bu repo, "**Yapay Zekâ ve Transformer Tabanlı Kurumsal Uyumsuzluk Endeksi (CII): Dijital Synoptikon Çağında Söylem-Eylem Kopukluğu**" başlıklı yüksek lisans tezinde geliştirilen **Kurumsal Uyumsuzluk Endeksi (CII v1.0)**'nin hesaplanmasında kullanılan analiz kodlarını içerir. Amaç, tezin metodoloji bölümünde (Bölüm 3 ve Ek 8.1) anlatılan beş adımlı algoritmik protokolün **tekrarlanabilir (reproducible)** olmasını sağlamaktır.

> **Not:** Bu repo yalnızca kaynak kodu içerir. Ham/bireysel yorum verileri (Indeed, Glassdoor, Şikayetvar) gizlilik ve etik onay kapsamı gereği paylaşılmamaktadır — bkz. [Veri Erişimi](#veri-erişimi) bölümü.

---

## Pipeline Genel Bakış

| Adım | Script | Ne yapar | Çalıştığı yer |
|---|---|---|---|
| 1 | `A_temizle_birlestir.py` | Ham veriyi temizler, şirket adlarını standardize eder, mükerrer kayıtları siler | Yerel Python |
| 2 | `B_duygu_analizi.py` | XLM-RoBERTa / BERTurk ile duygu skorlama + yıldız-metin uyumsuzluğu (dissonance, λ=0,3) | Google Colab (GPU) |
| 3 | `C_vitrin_analizi.py` | Vitrin (kurumsal iletişim) metinlerini 400/200 kayan pencere ile zero-shot sınıflandırır | Google Colab (GPU) |
| 4 | `D_istatistik_final.py` | Nihai CII hesaplanır; Spearman+BH-FDR, Kruskal-Wallis, Wilcoxon, bootstrap CI, Fisher-Z güç analizi | Yerel Python |
| 5 | `E_kappa_dogrulama.py` | İnsan doğrulayıcılar için etiketleme şablonu üretir ve Cohen's Kappa hesaplar | Yerel Python |

**Ek doğrulama script'leri** (Bölüm 3.6/3.9 güvenilirlik analizleri için):
- `prepare_manual_labels.py` — iki bağımsız kodlayıcı için kör (blind) etiketleme dosyaları üretir
- `xlm_roberta_validate.py` — XLM-RoBERTa çıktısını N=200 insan etiketiyle karşılaştırır
- `calculate_human_kappa.py` — iki kodlayıcı arası Cohen's Kappa + model-insan karşılaştırması

Script'ler **sırasıyla** (A → B → C → D → E) çalıştırılmalıdır; D, B ve C'nin çıktılarına bağımlıdır.

---

## Kurulum

```bash
git clone <repo-url>
cd <repo-adı>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

B ve C script'leri (`transformers` tabanlı model çıkarımı) GPU olmadan da çalışır, ancak **çok yavaş** olur. Google Colab'da ücretsiz GPU ile çalıştırılması önerilir:

```python
!pip install -q transformers torch sentencepiece accelerate
```

---

## Klasör Yapısı

```
.
├── 00_HAM_VERI/          # Ham veri (paylaşılmaz — bkz. Veri Erişimi)
├── 01_TEMIZ_VERI/        # Script A çıktısı
├── 02_DUYGU_SONUC/       # Script B çıktısı
├── 03_CII_SONUC/         # Script D çıktısı — CII tabloları + bootstrap CI
├── 04_VITRIN_SONUC/      # Script C çıktısı
├── 05_KAPPA/             # Script E çıktısı — Kappa doğrulama
├── 06_SONUC_ISTATISTIK/  # Script D çıktısı — hipotez testleri + tam rapor
└── 08_SCRIPTLER/         # Tüm Python kaynak kodu (bu repo)
```

Tüm script'ler kendi konumlarına göre göreli yol (`Path(__file__).resolve().parent.parent`) kullanır; script'i nereye klonlarsanız klonlayın, girdi/çıktı klasörlerini otomatik bulur.

---

## Çalıştırma

```bash
cd 08_SCRIPTLER
python A_temizle_birlestir.py
python B_duygu_analizi.py          # veya Colab'da çalıştırın
python C_vitrin_analizi.py         # veya Colab'da çalıştırın
python D_istatistik_final.py
python E_kappa_dogrulama.py            # Aşama A: etiketleme şablonu üretir
# ... iki kodlayıcı bağımsız olarak etiketler ...
python E_kappa_dogrulama.py hesapla    # Aşama B: Kappa hesaplar
```

Tez metnine aktarılacak nihai sayılar: `06_SONUC_ISTATISTIK/ISTATISTIK_TAM_RAPOR.txt`

---

## Veri Erişimi

Bu çalışmada kullanılan ham veri (Indeed, Glassdoor, Şikayetvar'dan çekilen bireysel çalışan/müşteri yorumları) **kişisel veri ve platform kullanım koşulları** gereği bu repoda paylaşılmamıştır.

**Şirket kimliği gizliliği:** Tez, etik kurul onayı gereği 24 şirketin gerçek unvanlarını kod adlarıyla (PRK-1, GDA-2, TEL-1 vb.) anonimleştirmiştir. Bu nedenle `A_temizle_birlestir.py`, gerçek şirket→sektör eşleştirmesini script içinde **tutmaz** — bunun yerine yalnızca sizin bilgisayarınızda kalması gereken `00_HAM_VERI/sektor_haritasi_KAYNAK.csv` adlı, `.gitignore` ile Git'ten hariç tutulmuş bir dosyadan okur. Bu dosyayı GitHub'a asla yüklemeyin.

Toplulaştırılmış/özet sonuç tabloları (`03_CII_SONUC/`, `06_SONUC_ISTATISTIK/`) — şirket kodlarıyla, gerçek isim içermeyen — isteğe bağlı olarak paylaşılabilir. Veri erişimi için makul taleplerde yazar ile iletişime geçilebilir.

---

## Metodoloji Referansı

Bu kodun uyguladığı beş adımlı protokolün tam açıklaması için tezin **Ek 8.1**'ine, endeksin yönetsel yorumu için **Bölüm 6.1**'e bakınız.

---

## Zenodo / DOI (Kalıcı Arşiv)

Bu repo Zenodo ile bağlandığında, her GitHub "release"i otomatik olarak arşivlenir ve kalıcı bir DOI alır. Bağlantı kurulduktan sonra DOI rozetini buraya ekleyin:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

*(GitHub–Zenodo bağlantısı kurulup ilk release yapıldıktan sonra yukarıdaki `XXXXXXX` kısmını Zenodo'nun size verdiği gerçek numarayla değiştirin.)*

---

## Atıf (Citation)

Bu kodu kullanırsanız lütfen tezi kaynak gösterin:

> Yiğitsoy, C. (2026). *Yapay Zekâ ve Transformer Tabanlı Kurumsal Uyumsuzluk Endeksi (CII): Dijital Synoptikon Çağında Söylem-Eylem Kopukluğu*. Yüksek Lisans Tezi, Marmara Üniversitesi Sosyal Bilimler Enstitüsü.

---

## Lisans

Belirtilmedikçe bu kod [MIT Lisansı](https://opensource.org/licenses/MIT) altında paylaşılmaktadır — dilerseniz `LICENSE` dosyası ekleyip değiştirebilirsiniz.
