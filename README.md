# CareerLens

CareerLens, CV'leri iş ilanlarıyla gereksinim bazlı karşılaştıran ve yüklenen kariyer belgeleri üzerinden kanıt destekli yanıtlar veren yerel bir kariyer asistanıdır.

Projeyi geliştirirken temel hedefim yalnızca bir benzerlik skoru üretmek değil, iş ilanındaki her gereksinimin CV'de gerçekten desteklenip desteklenmediğini açıklanabilir biçimde göstermekti.

## Özellikler

- CV, iş ilanı ve isteğe bağlı kariyer rehberi yükleme
- Belgeleri yerel olarak işleme ve indeksleme
- Microsoft Foundry Local ile embedding üretimi
- SQLite tabanlı yerel veri saklama
- CV ve iş ilanı için belge türüne göre sorgu yönlendirme
- Zorunlu ve tercih edilen gereksinimleri ayrı değerlendirme
- `Eşleşiyor`, `Kısmen eşleşiyor` ve `CV'de açık kanıt yok` sonuçları
- İngilizce seviye şartlarını CEFR seviyeleri üzerinden kontrol etme
- Her eşleşme için CV'den kısa kanıt gösterme
- Yüklenen belgeler üzerinde hızlı soru-cevap
- İsteğe bağlı yerel dil modeliyle ayrıntılı yanıt

## Nasıl çalışır?

```text
Belge yükleme
    ↓
Metin çıkarma ve temizleme
    ↓
Parçalara ayırma
    ↓
Foundry Local ile embedding
    ↓
SQLite indeks
    ↓
Sorgu yönlendirme
    ↓
Retrieval / ilgili kanıtları bulma
    ↓
Gereksinim doğrulama veya soru-cevap
    ↓
CareerLens sonucu
```

### Belge işleme

Yüklenen PDF, DOCX veya metin dosyalarından içerik çıkarılır ve daha küçük metin bölümlerine ayrılır. Bu bölümler Foundry Local üzerinden yerel embedding modeline gönderilir. Oluşan vektörler ve belge bilgileri SQLite veritabanında saklanır.

### CV ↔ İş İlanı karşılaştırması

İş ilanındaki zorunlu ve tercih edilen nitelikler ayrıştırılır. Ardından her gereksinim CV içindeki açık kanıtlarla ayrı ayrı kontrol edilir.

Sistem bir gereksinimi değerlendirirken yalnızca CV'de bulunan ifadeleri kullanır. CV'de açık kanıt bulunmaması, adayın ilgili beceriye sahip olmadığı anlamına gelmez; yalnızca yüklenen belgede doğrulanamadığını belirtir.

Bazı şartlar yalnızca metin benzerliğiyle güvenilir biçimde değerlendirilemediği için ek kontroller uygulanır. Örneğin:

- öğrenci / mezun ayrımı
- B1 / B2 / C1 İngilizce seviyeleri
- `SQL + Excel` gibi birden fazla koşul içeren gereksinimler
- araç ve teknoloji kullanımının gerçekten deneyim cümlesinde geçip geçmediği

### CareerLens'e Sor

Soru önce CV, iş ilanı veya kariyer rehberi kapsamına yönlendirilir. Ardından yalnız ilgili belgeler içinde arama yapılır.

Varsayılan hızlı yanıt modu, bulunan kanıtları doğrudan kullanarak kısa cevap oluşturur. Bu yaklaşım özellikle CPU üzerinde çalışan yerel modellerde bekleme süresini azaltmak için kullanılır.

İstenirse arama ayarlarından yerel modelle ayrıntılandırma seçeneği açılabilir. Bu durumda bulunan bağlam Foundry Local üzerinden yerel sohbet modeline gönderilir.

## Kullanılan teknolojiler

- Python
- Streamlit
- Microsoft Foundry Local
- SQLite
- NumPy
- pypdf
- python-docx

## Kurulum

### Gereksinimler

- Windows
- Python 3.11 veya üzeri
- Microsoft Foundry Local

### İlk kurulum

Proje klasöründe:

```bat
setup.bat
```

Bu komut sanal ortamı oluşturur ve Python bağımlılıklarını yükler.

Kurulumdan sonra istersen çalışma ortamını kontrol edebilirsin:

```bat
run_test.bat
```

Uygulamayı başlatmak için:

```bat
run_app.bat
```

Ardından Streamlit tarafından verilen yerel adresi tarayıcıda açabilirsin.

## Proje yapısı

```text
CareerLens/
├── app.py
├── assets/
│   └── kariyerlens_logo_transparent.png
├── data/
├── src/
│   ├── database.py
│   ├── document_utils.py
│   ├── fast_qa.py
│   ├── foundry_runtime.py
│   ├── prompts.py
│   ├── query_router.py
│   ├── rag_engine.py
│   ├── requirement_guard.py
│   ├── text_utils.py
│   └── ui_evidence.py
├── tests/
├── requirements.txt
├── setup.bat
├── run_app.bat
└── run_test.bat
```

## Gizlilik

CareerLens yerel çalışma mantığıyla tasarlanmıştır. Belgeler, indeksler ve model çıkarımları cihaz üzerinde tutulur. `data/knowledge.db` gibi çalışma sırasında oluşan yerel veriler Git deposuna eklenmez.

## Not

CareerLens'in ürettiği eşleşme sonucu bir ATS puanı veya işe alınma olasılığı değildir. Sonuç yalnızca yüklenen CV ve iş ilanında açıkça bulunan kanıtlara dayanır.
