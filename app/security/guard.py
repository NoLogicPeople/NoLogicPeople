import json
import os
from typing import Tuple, List, Optional

class Guard:
    """
    LLM girdilerini PII, prompt injection ve diğer istenmeyen kalıplara
    karşı korumak için yapılandırılabilir bir güvenlik katmanı.
    """
    def __init__(self, config_path: str = "config.json") -> None:
        """
        Guard'ı belirtilen yapılandırma dosyasından başlatır.
        """
        # Config dosyası öncelik sırası:
        # 1) GUARD_CONFIG_PATH ortam değişkeni
        # 2) app/security/config.json (bu dosya ile aynı klasörde)
        base_dir = os.path.dirname(__file__)
        default_path = config_path
        if not os.path.isabs(default_path):
            default_path = os.path.join(base_dir, default_path)

        env_path = os.getenv("GUARD_CONFIG_PATH")
        if env_path and not os.path.isabs(env_path):
            env_path = os.path.join(base_dir, env_path)

        candidate_paths = [p for p in [env_path, default_path] if p]

        config = {}
        loaded = False
        for candidate in candidate_paths:
            try:
                with open(candidate, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    loaded = True
                    break
            except FileNotFoundError:
                continue
            except json.JSONDecodeError as e:
                print(f"⚠️ Guard yapılandırma dosyası geçerli bir JSON değil ('{candidate}'): {e}. Varsayılan değerler kullanılacak.")
                break

        if not loaded:
            # Dosya bulunamazsa güvenli varsayılanlarla devam et
            print(f"⚠️ Guard yapılandırma dosyası bulunamadı. Şu yollar denendi: {candidate_paths}. Varsayılan değerler kullanılacak.")

        # Temel yapılandırmadan değerleri al
        self._max_length: int = config.get("max_prompt_length", 5000)
        self._pii_keywords: List[str] = config.get("pii_keywords", [])
        self._prompt_injection_keywords: List[str] = config.get("prompt_injection_keywords", [])
        
        # NLP güvenlik ayarları
        nlp_config = config.get("nlp_security", {})
        self._enable_ner = nlp_config.get("enable_ner", False)
        self._enable_semantic = nlp_config.get("enable_semantic_similarity", False)
        self._enable_toxicity = nlp_config.get("enable_toxicity_detection", False)
        self._ner_threshold = nlp_config.get("ner_confidence_threshold", 0.85)
        self._similarity_threshold = nlp_config.get("similarity_threshold", 0.75)
        self._toxicity_threshold = nlp_config.get("toxicity_threshold", 0.8)
        self._attack_prompts = nlp_config.get("known_attack_prompts", [])
        
        # NLP modelleri (lazy loading)
        self.ner_pipeline = None
        self.similarity_model = None
        self.attack_embeddings = None
        self.toxicity_pipeline = None
        
        # NLP modellerini yükle (eğer aktifse)
        self._init_nlp_models()

    def _init_nlp_models(self) -> None:
        """NLP modellerini lazy loading ile başlatır."""
        if self._enable_ner:
            try:
                print("🔒 Türkçe NER modeli yükleniyor...")
                from transformers import pipeline
                # Türkçe için fine-tune edilmiş BERT NER modeli
                self.ner_pipeline = pipeline("ner", model="savasy/bert-base-turkish-ner-cased", 
                                           aggregation_strategy="simple", device=-1)
                print("✅ Türkçe NER modeli hazır")
            except Exception as e:
                print(f"⚠️ Türkçe NER modeli yüklenemedi: {e}")
                # Fallback: Multilingual model
                try:
                    print("🔄 Çok dilli NER modeline geçiliyor...")
                    self.ner_pipeline = pipeline("ner", model="Davlan/xlm-roberta-base-ner-hrl", 
                                               aggregation_strategy="simple", device=-1)
                    print("✅ Çok dilli NER modeli hazır")
                except Exception as e2:
                    print(f"⚠️ NER modeli tamamen yüklenemedi: {e2}")
                    self._enable_ner = False
        
        if self._enable_semantic and self._attack_prompts:
            try:
                print("🔒 Semantik benzerlik modeli yükleniyor...")
                from sentence_transformers import SentenceTransformer
                import torch
                
                self.similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                
                # Attack prompts'ları güvenli şekilde encode et
                embeddings = self.similarity_model.encode(self._attack_prompts, convert_to_tensor=True)
                
                # Tensor boyutlarını kontrol et
                if isinstance(embeddings, torch.Tensor):
                    if len(embeddings.shape) == 1:
                        embeddings = embeddings.unsqueeze(0)
                    self.attack_embeddings = embeddings
                else:
                    # NumPy array ise tensor'a çevir
                    import numpy as np
                    if isinstance(embeddings, np.ndarray):
                        self.attack_embeddings = torch.tensor(embeddings)
                    else:
                        self.attack_embeddings = embeddings
                
                print("✅ Semantik model hazır")
            except Exception as e:
                print(f"⚠️ Semantik model yüklenemedi: {e}")
                self._enable_semantic = False
        
        if self._enable_toxicity:
            try:
                print("🔒 Türkçe toksisite modeli yükleniyor...")
                from transformers import pipeline
                # Türkçe için özel eğitilmiş toksisite modeli
                self.toxicity_pipeline = pipeline("text-classification", 
                                                 model="savasy/bert-base-turkish-sentiment-cased", device=-1)
                print("✅ Türkçe toksisite modeli hazır")
            except Exception as e:
                print(f"⚠️ Türkçe toksisite modeli yüklenemedi: {e}")
                # Fallback: Çok dilli toksisite modeli
                try:
                    print("🔄 Çok dilli toksisite modeline geçiliyor...")
                    self.toxicity_pipeline = pipeline("text-classification", 
                                                     model="unitary/multilingual-toxic-xlm-roberta", device=-1)
                    print("✅ Çok dilli toksisite modeli hazır")
                except Exception as e2:
                    print(f"⚠️ Toksisite modeli tamamen yüklenemedi: {e2}")
                    self._enable_toxicity = False

    def _check_length(self, text: str) -> Optional[str]:
        """Girdi metninin uzunluğunu kontrol eder."""
        if not text:
            return "Boş istek."
        if len(text) > self._max_length:
            return f"İstek çok uzun (Maks: {self._max_length} karakter)."
        return None

    def _check_keywords(self, text: str) -> Optional[str]:
        """Yapılandırılmış anahtar kelime listelerine göre metni kontrol eder."""
        # PII anahtar kelime kontrolü kaldırıldı - sadece NER kullanılacak
        if any(keyword in text for keyword in self._prompt_injection_keywords):
            return "Kural atlama girişimi algılandı."
        return None

    def _check_ner_entities(self, text: str) -> Optional[str]:
        """NER ile hassas varlıkları tespit eder."""
        if not self._enable_ner or not self.ner_pipeline:
            return None
            
        try:
            entities = self.ner_pipeline(text)
            # Sadece kişi adlarını kontrol et - TCKN, yer ve kurum adları e-Devlet için normaldir
            sensitive_entities = {'PER', 'PERSON'}
            
            for entity in entities:
                entity_type = entity.get('entity_group', '').upper()
                confidence = entity.get('score', 0)
                entity_word = entity.get('word', '').strip()
                
                # Kişi adı tespiti ve TCKN benzeri sayıları filtrele
                if entity_type in sensitive_entities and confidence > self._ner_threshold:
                    # TCKN benzeri 11 haneli sayıları ignore et
                    if entity_word.isdigit() and len(entity_word) == 11:
                        continue
                    # Kısa sayısal değerleri ignore et (kimlik no, dosya no vs.)
                    if entity_word.isdigit() and len(entity_word) <= 15:
                        continue
                    # Gerçek kişi adı tespit edildi
                    return f"Kişi adı algılandı (Ad: {entity_word})"
        except Exception as e:
            print(f"NER kontrolü başarısız: {e}")
        
        return None

    def _check_semantic_similarity(self, text: str) -> Optional[str]:
        """Semantik benzerlik ile prompt injection tespiti."""
        if not self._enable_semantic or not self.similarity_model or self.attack_embeddings is None:
            return None
            
        try:
            from sentence_transformers import util
            import torch
            
            # Text'i encode et
            text_embedding = self.similarity_model.encode(text, convert_to_tensor=True)
            
            # Tensor boyutlarını kontrol et ve düzelt
            if len(text_embedding.shape) == 1:
                text_embedding = text_embedding.unsqueeze(0)
            
            # Cosine similarity hesapla
            cosine_scores = util.cos_sim(text_embedding, self.attack_embeddings)
            
            # Max score'u güvenli şekilde al
            if isinstance(cosine_scores, torch.Tensor):
                max_score = torch.max(cosine_scores).item()
            else:
                max_score = float(cosine_scores.max())
            
            if max_score > self._similarity_threshold:
                return f"Kural atlama girişimi algılandı (Benzerlik Skoru: {max_score:.2f})"
                
        except Exception as e:
            print(f"Semantik benzerlik kontrolü başarısız: {e}")
        
        return None

    def _check_toxicity(self, text: str) -> Optional[str]:
        """Toksisite ve zararlı içerik tespiti."""
        if not self._enable_toxicity or not self.toxicity_pipeline:
            return None
            
        try:
            results = self.toxicity_pipeline(text)
            
            # Model çıktısına göre kontrol
            for result in results:
                label = result.get('label', '').lower()
                score = result.get('score', 0)
                
                # Türkçe sentiment modeli: negative sentiment yüksekse toksik olabilir
                if 'negative' in label and score > self._toxicity_threshold:
                    return f"Olumsuz/toksik içerik algılandı (Skor: {score:.2f})"
                elif 'toxic' in label and score > self._toxicity_threshold:
                    return f"Toksik içerik algılandı (Skor: {score:.2f})"
                elif 'hate' in label and score > self._toxicity_threshold:
                    return f"Nefret söylemi algılandı (Skor: {score:.2f})"
        except Exception as e:
            print(f"Toksisite kontrolü başarısız: {e}")
        
        return None


    def check_prompt_safety(self, text: str) -> Tuple[bool, str]:
        """
        Girdi metnini tüm güvenlik kontrollerinden geçirir.
        Herhangi bir kontrol başarısız olursa, işlemi durdurur ve nedenini döndürür.

        Returns:
            (is_safe, reason) tuple'ı.
        """
        if not text:
            return False, "Boş istek."
            
        # Gelen metni her seferinde temizle ve küçük harfe çevir
        processed_text = text.lower().strip()
        
        # Katman 1: Hızlı kontroller (uzunluk, temel anahtar kelimeler)
        basic_checks = [
            lambda t: self._check_length(t),
            lambda t: self._check_keywords(t),
        ]

        for check_func in basic_checks:
            reason = check_func(processed_text)
            if reason:
                return False, reason

        # Katman 2-4: NLP tabanlı akıllı kontroller (orijinal metin ile)
        nlp_checks = [
            lambda t: self._check_ner_entities(t),
            lambda t: self._check_semantic_similarity(t),
            lambda t: self._check_toxicity(t),
        ]

        for check_func in nlp_checks:
            reason = check_func(text)  # Orijinal metin (büyük/küçük harf duyarlı)
            if reason:
                return False, reason

        return True, "OK"