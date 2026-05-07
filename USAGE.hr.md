# Upute za korištenje

[🇬🇧 English](USAGE.md) | **🇭🇷 Hrvatski**

## Postavljanje

```bash
pip install -e .
```

---

## 1. Predobrada skupa podataka

Uzima sirove ISIC 2020 slike i proizvodi predobrađene slike podijeljene u foldove, spremne za treniranje.

```bash
python scripts/preprocess_dataset.py \
    --images putanja/do/train \
    --labels putanja/do/ISIC_2020_Training_GroundTruth.csv \
    --duplicates putanja/do/2020_Challenge_duplicates.csv \
    --output preprocessed
```

| Zastavica | Obavezno | Zadano | Opis |
|------|----------|---------|-------------|
| `--images` | da | — | Mapa sa sirovim `.jpg` slikama |
| `--labels` | da | — | ISIC 2020 CSV s točnim oznakama |
| `--duplicates` | da | — | CSV s parovima duplih slika |
| `--output` | ne | `preprocessed` | Izlazna mapa |
| `--percent` | ne | `1.0` | Udio skupa podataka koji se koristi (0.0–1.0) |
| `--target-size` | ne | `200` | Veličina izlazne slike u pikselima |
| `--num-folds` | ne | `5` | Broj foldova |
| `--max-class0` | ne | `20` | Maks. broj slika klase 0 po pacijentu |
| `--no-parallel` | ne | — | Onemogući višeprocesnu obradu |
| `--seed` | ne | `42` | Sjeme generatora slučajnih brojeva |

**Izlaz:**
```
preprocessed/
├── 0_fold/          # Predobrađene slike po foldu
├── 1_fold/
├── ...
├── folds.csv        # Pridruživanje pacijenata foldovima
└── metadata.json    # ITA vrijednosti, oznake, ID-evi foldova
```

> **Koristite drugi skup podataka?** ISIC 2020 je zadan, ali bilo koji skup
> radi sve dok ulazi prate istu shemu:
> - `--labels` CSV mora imati stupce `image_name`, `target` (0/1) i `patient_id`.
> - `--duplicates` CSV mora imati stupac `ISIC_id_paired` (proslijedite prazan
>   CSV samo sa zaglavljem ako vaš skup nema poznatih duplikata).
> - `--images` mora biti mapa s `.jpg` datotekama čija imena odgovaraju `image_name`.

---

## 2. Treniranje

Pokreće k-struku unakrsnu validaciju s loss funkcijom svjesnom pravednosti.

```bash
python scripts/train.py \
    --data-dir preprocessed \
    --metadata preprocessed/metadata.json
```

| Zastavica | Obavezno | Zadano | Opis |
|------|----------|---------|-------------|
| `--data-dir` | da | — | Mapa s predobrađenim foldovima (iz koraka 1) |
| `--metadata` | da | — | Putanja do `metadata.json` (iz koraka 1) |
| `--output-dir` | ne | `models` | Gdje spremiti `.pth` težine |
| `--target-ratio` | ne | `0.15` | Ciljani udio klase 1 nakon augmentacije |
| `--num-folds` | ne | `5` | Broj foldova |
| `--test-fold` | ne | `4` | Fold rezerviran za završno testiranje |
| `--epochs` | ne | `35` | Maks. broj epoha |
| `--batch-size` | ne | `128` | Veličina batcha |
| `--lr` | ne | `3e-5` | Stopa učenja (AdamW) |
| `--seed` | ne | `42` | Sjeme generatora slučajnih brojeva |

**Izlaz:**
```
models/
├── model_fold_0.pth
├── model_fold_1.pth
├── model_fold_2.pth
└── model_fold_3.pth
```

---

## 3. Predviđanje

Pokreće batch predikciju nad mapom `.jpg` slika.

```bash
python scripts/predict.py \
    --images putanja/do/slika \
    --weights models/model_fold_0.pth \
    --output predictions.csv
```

| Zastavica | Obavezno | Zadano | Opis |
|------|----------|---------|-------------|
| `--images` | da | — | Mapa s `.jpg` slikama za klasifikaciju |
| `--weights` | da | — | Putanja do `.pth` datoteke s istreniranim težinama |
| `--output` | ne | `predictions.csv` | Putanja izlaznog CSV-a |
| `--threshold` | ne | `0.5` | Prag klasifikacije |
| `--no-parallel` | ne | — | Onemogući višeprocesnu obradu |

**Izlaz:** CSV sa stupcima `image_name, target` (0 = benigno, 1 = maligno).

---

## 4. Web aplikacija

```bash
cd web_app
docker build -t melanoma-detector .
docker run -p 8000:8000 melanoma-detector
```

Otvara se na `http://localhost:8000`. Težine modela preuzimaju se automatski pri prvom pokretanju.

---

## Python API

Sve je dostupno za uvoz iz paketa `lumen`:

```python
from lumen.preprocessing import preprocess, remove_hair, square_crop
from lumen.skin_tone import get_fitzpatrick, calculate_ita_subregions
from lumen.model import CustomCNN, PretrainedEfficientNet, load_model
from lumen.inference import prepare_tensor, predict, apply_gradcam
from lumen.training.loss import FairnessAwareLoss
from lumen.training.augmentation import augment
from lumen.training.dataset import SkinImageDataset
from lumen.training.evaluation import detailed_evaluation, evaluate_fairness
from lumen.training.trainer import k_fold_training
from lumen.folding import triple_stratified_fold
```

### Brza predikcija jedne slike

```python
import cv2
from lumen.model import CustomCNN, load_model
from lumen.preprocessing import preprocess
from lumen.inference import prepare_tensor, predict

model = load_model(CustomCNN, "models/model_fold_0.pth")
image = cv2.cvtColor(cv2.imread("lezija.jpg"), cv2.COLOR_BGR2RGB)
processed = preprocess(image, target_size=(224, 224), compute_ita=False)
prob, cls = predict(model, prepare_tensor(processed))
print(f"{'Maligno' if cls else 'Benigno'} ({prob:.1%})")
```

---

## Cijeli postupak: od sirovih slika do testiranog modela

Pretpostavlja se da imate ISIC 2020 skup podataka na disku:
- `data/train/` — mapa sa sirovim `.jpg` slikama
- `data/ISIC_2020_Training_GroundTruth.csv` — oznake
- `data/2020_Challenge_duplicates.csv` — popis duplikata

```bash
# 1. Instaliraj paket u editable modu
pip install -e .

# 2. Predobrada: uklanjanje dlaka, kvadratno izrezivanje, resize, ITA, k-fold
python scripts/preprocess_dataset.py \
    --images data/train \
    --labels data/ISIC_2020_Training_GroundTruth.csv \
    --duplicates data/2020_Challenge_duplicates.csv \
    --output preprocessed

# 3. Treniranje: 5-struka CV, fold 4 zadržan za test, težine se spremaju u models/
python scripts/train.py \
    --data-dir preprocessed \
    --metadata preprocessed/metadata.json \
    --output-dir models

# 4. Predikcija na zadržanom test foldu (fold 4) koristeći prvi istreniran model
python scripts/predict.py \
    --images preprocessed/4_fold \
    --weights models/model_fold_0.pth \
    --output predictions_fold4.csv

# 5. Pregled predikcija
head predictions_fold4.csv
```

Nakon koraka 3 dobijete `models/model_fold_{0..3}.pth`. Korak 4 pokreće
inferenciju nad zadržanim foldom; promijenite `--weights` da testirate model
svakog folda, ili usmjerite `--images` na bilo koju mapu novih `.jpg` slika
za stvarne predikcije.
