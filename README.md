# Türkçe e-Devlet Sesli RAG Asistanı

Bu proje, Türk vatandaşlarının e-Devlet hizmetlerine daha kolay erişim sağlaması için geliştirilmiş **Agentic** ve **Retrieval-Augmented Generation (RAG)** tabanlı bir sesli asistanıdır. Modern doğal dil işleme teknikleri kullanarak, kullanıcıların sorularını anlayıp en uygun kamu hizmetlerini öneren kapsamlı bir AI sistemidir.



## Kurulum

### 1. Projeyi İndirin
```bash
git clone <repository-url>
cd Archive
```

### 2. Python Sanal Ortamı Oluşturun
```bash
python3 -m venv .venv && source .venv/bin/activate
```

### 3. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

## Çalıştırma

### Web Arayüzü (Önerilen)

#### Linux/Mac/WSL
```bash
bash run_web.sh
```


## Kullanım

### Web Arayüzü
1. Tarayıcınızda `http://127.0.0.1:8000` adresine gidin
2. İlk açılışta modellerin indirilmesi uzun sürebilir, sağ üstte oturum id'sini görene kadar bekleyin, terminalden logları takip edin.
3. Chat arayüzünde sorularınızı yazın veya mikrofon butonuna basıp konuşun, konuşmanız bitince tekrar butona basın
4. Sistem size uygun e-Devlet hizmetlerini önerecek
5. Hizmet seçimi yapın ve işlemlerinizi tamamlayın

### Örnek Sorgular
```
- "Öğrenci belgesi sorgulama"
- "e-Devlet'ten askerlik sorgulama"
- "Vergi borcu sorgulama"
- "SGK hizmetleri"
```



## Lisans

Bu proje Apache 2.0 Lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## Teşekkürler


**Not**: Bu sistem sadece bilgi amaçlıdır. Resmi e-Devlet işlemleri için lütfen [e-devlet.gov.tr](https://e-devlet.gov.tr) adresini kullanın.
