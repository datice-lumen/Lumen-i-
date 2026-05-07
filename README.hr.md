# Klasifikacija melanoma vođena pravednošću

[🇬🇧 English](README.md) | **🇭🇷 Hrvatski**

Pristup dubokog CNN-a s ujednačenim performansama po tonu kože, razvijen za **Lumen Data Science Challenge 2025**.

**Autori:** Jurica Jerinic, Filip Hlup, Tomislav Matanovic, Karlo Rastegorac
**Grupa:** Datice

## Pregled

Projekt razvija robustan model dubokog učenja koji klasificira dermatoskopske slike kožnih lezija kao **benigne ili maligne**, s naglaskom na **pravednost među tonovima kože**. Model je prilagođeni CNN (6.7M parametara) treniran od nule na [ISIC 2020 skupu podataka](https://challenge2020.isic-archive.com/), uz loss funkciju svjesnu pravednosti baziranu na Equalized Odds.

| Metrika | Rezultat na testu |
|--------|-----------|
| Točnost | 0.83 |
| AUC | 0.86 |
| TPR (osjetljivost) | 0.69 |
| FPR | 0.16 |
| Equalized Odds Gap | 0.51 |

Dostupna je i [web aplikacija uživo](https://lumen-i.onrender.com) za interaktivne predikcije s Grad-CAM objašnjivošću.

## Struktura repozitorija

```
src/lumen/                        # Glavni Python paket (pip install -e .)
  preprocessing.py                # Uklanjanje dlaka, izrezivanje, resize
  skin_tone.py                    # ITA i Fitzpatrick mapiranje
  model.py                        # CustomCNN + PretrainedEfficientNet arhitekture
  inference.py                    # Priprema tenzora, predikcija, Grad-CAM
  folding.py                      # Trostruko stratificirano k-struko dijeljenje
  training/                       # Moduli specifični za treniranje
    loss.py                       # Prilagođena loss funkcija svjesna pravednosti
    augmentation.py               # Transformacije za augmentaciju podataka
    dataset.py                    # PyTorch Dataset s paralelnim učitavanjem
    evaluation.py                 # Metrike, evaluacija pravednosti, crtanje
    trainer.py                    # Petlja treniranja s early stoppingom

scripts/                          # CLI ulazne točke
  preprocess_dataset.py           # Batch predobrada skupa podataka
  predict.py                      # Batch inferencija
  train.py                        # Treniranje modela

configs/default.yaml              # Konfiguracija treniranja/inferencije
web_app/
  Dockerfile                      # Multi-stage Docker build
  backend/                        # FastAPI + PyTorch backend
  frontend/                       # Vue.js + Naive UI frontend
```

## Brzi početak

### Preduvjeti

- Python 3.10+
- PyTorch 2.6+
- [ISIC 2020 trening skup](https://challenge2020.isic-archive.com/)

### Instaliraj ovisnosti

```bash
pip install -e .
```

Instalira `lumen` paket u editable modu sa svim ovisnostima.

### 1. Predobrada podataka

```bash
python scripts/preprocess_dataset.py \
    --images data/train \
    --labels data/ISIC_2020_Training_GroundTruth.csv \
    --duplicates data/2020_Challenge_duplicates.csv \
    --output preprocessed
```

Provodi uklanjanje duplikata, uklanjanje dlaka, ITA procjenu Fitzpatrick tona kože, ograničavanje broja slika po pacijentu i trostruko stratificirano k-struko dijeljenje. Izlaz je mapa `preprocessed/` s slikama po foldovima i `metadata.json`.

Bilo koji skup podataka u ISIC 2020 shemi radi (CSV s oznakama mora imati stupce `image_name`, `target`, `patient_id`). Detalji o zastavicama nalaze se u [USAGE.hr.md](USAGE.hr.md).

### 2. Treniranje modela

```bash
python scripts/train.py \
    --data-dir preprocessed \
    --metadata preprocessed/metadata.json \
    --output-dir models
```

Pokreće augmentaciju, treniranje s prilagođenom loss funkcijom svjesnom pravednosti, k-struku unakrsnu validaciju i spremanje modela. GPU se preporučuje.

**Konfiguracija treniranja:** AdamW optimizator, LR 3e-5, batch 128, do 35 epoha s early stoppingom.

### 3. Pokretanje inferencije

```bash
python scripts/predict.py \
    --images putanja/do/slika \
    --weights models/model_fold_0.pth \
    --output predictions.csv
```

Obrađuje mapu `.jpg` slika kroz isti cjevovod predobrade i ispisuje binarne predikcije (`image_name, target`) u CSV. Podržava paralelnu predobradu.

### 4. Pokretanje web aplikacije

```bash
cd web_app
docker build -t melanoma-detector .
docker run -p 8000:8000 melanoma-detector
```

Aplikacija automatski preuzima težine modela s Google Drivea pri prvom pokretanju. Pristupa se na `http://localhost:8000`.

## Konfiguracija

Zadane postavke za treniranje i inferenciju nalaze se u [`configs/default.yaml`](configs/default.yaml):

```yaml
model:
  name: "CustomCNN"          # ili "PretrainedEfficientNet"
  input_size: [224, 224]

training:
  epochs: 35
  batch_size: 128
  learning_rate: 0.00003     # 3e-5
  optimizer: "adamw"
  class_weights: [0.5, 8.0]
  target_ratio: 0.15
  early_stopping_patience: 5
  lr_reduction_patience: 2
  lr_reduction_factor: 0.5
```

## Ključne tehničke odluke

- **Prilagođeni CNN umjesto pretreniranog EfficientNeta:** bolja točnost uz manju kompleksnost za ovaj specifičan zadatak
- **Trostruko stratificirano dijeljenje:** sprječava curenje podataka tako što razdvaja pacijente između foldova
- **Loss funkcija svjesna pravednosti:** uključuje Equalized Odds Gap regularizaciju, kaznu za recall po klasi i težinu svjesnu augmentacije
- **Predobrada uklanjanjem dlaka:** morfološki black-hat filter + inpainting za smanjenje šuma od artefakata
- **ITA procjena tona kože:** računa se iz 8 perifernih podregija kako bi se izbjegao utjecaj lezije

## Dokumentacija

Cjelovita dokumentacija dostupna je na [project wiki](https://github.com/datice-lumen/Lumen-i-/wiki):

- [Project Documentation](https://github.com/datice-lumen/Lumen-i-/wiki/Project-Documentation) -- metodologija, rezultati, evaluacija pravednosti
- [Technical Documentation](https://github.com/datice-lumen/Lumen-i-/wiki/Technical-Documentation) -- detalji implementacije, opisi koda, deployment

## Licenca

Projekt je licenciran pod uvjetima navedenim u [LICENSE](LICENSE) datoteci.

## Napomena

Ovaj alat namijenjen je samo u edukativne i istraživačke svrhe. Nije validiran za kliničku upotrebu i ne smije zamijeniti profesionalnu medicinsku dijagnozu.
