# Struktura projekta

[🇬🇧 English](STRUCTURE.md) | **🇭🇷 Hrvatski**

## Pregled

Sva zajednička logika nalazi se u Python paketu `lumen` (`src/lumen/`). Skripte i web aplikacija su tanki omotači koji uvoze iz njega. Nijedna funkcija nije definirana na više mjesta.

```
src/lumen/                        Glavna biblioteka
src/lumen/training/               Moduli samo za treniranje (loss, dataset, augmentacija, trainer)
scripts/                          CLI ulazne točke (uvoze iz lumen-a)
web_app/                          FastAPI backend + Vue frontend (uvozi iz lumen-a)
configs/default.yaml              Zadani hiperparametri (putanje podataka, model, predobrada, treniranje)
```

---

## `src/lumen/` — Glavni paket

### `preprocessing.py`

Cjevovod obrade dermatoskopskih slika.

| Funkcija | Što radi |
|----------|-------------|
| `remove_hair(image)` | Black-hat morfologija + inpainting za uklanjanje dlaka |
| `square_crop(image)` | Centralno izrezivanje u kvadrat (uklanja rubove dermatoskopa) |
| `check_resize(image)` | Provjerava 3:2 omjer i minimum 500px |
| `preprocess(image, ...)` | Cijeli cjevovod: izrezivanje → uklanjanje dlaka → resize → opcionalno ITA |
| `preprocess_from_path(path, ...)` | Učitavanje s diska + međurazina resize + cijeli cjevovod |
| `preprocess_for_inference(path, ...)` | Isto, ali vraća RGB, preskače ITA (za ulaz modela) |
| `parallel_preprocess(paths, ...)` | Višeprocesni omotač za batch predobradu |
| `parallel_preprocess_for_inference(names, folder, ...)` | Višeprocesni omotač za batch inferenciju |

### `skin_tone.py`

Procjena tona kože temeljena na ITA-i i Fitzpatrick klasifikacija.

| Funkcija | Što radi |
|----------|-------------|
| `calculate_ita_subregions(image)` | Računa ITA iz 8 perifernih podregija, vraća prosjek 2 najsvjetlije |
| `get_fitzpatrick(ita)` | Mapira ITA → Fitzpatrick tip (cijeli broj 1–6) |
| `get_fitzpatrick_label(ita)` | Mapira ITA → čitljiva oznaka tipa `"III (Intermediate)"` |
| `assign_fitz_group(ita)` | Mapira ITA → grupa za treniranje (12, 3, 4 ili 56) — grupe I–II i V–VI grupiraju se zajedno |

### `model.py`

Arhitekture neuronskih mreža i upravljanje težinama.

| Klasa/Funkcija | Što radi |
|----------------|-------------|
| `CustomCNN` | CNN s 6.7M parametara treniran od nule. 4 konv. bloka + klasifikator s 3 sloja. Glavni model. |
| `PretrainedEfficientNet` | EfficientNet-B0 osnova s prilagođenom klasifikacijskom glavom. Alternativni model. |
| `load_model(model_class, path)` | Instancira + učita težine + postavi eval mod. Radi s obje arhitekture. |
| `download_weights_from_gdrive(path, file_id)` | Preuzima `.pth` s Google Drivea (koristi web aplikacija pri prvom pokretanju) |

Oba modela primaju 224x224 RGB ulaz i vraćaju sirove logite (sami primijenite sigmoid ili koristite `inference.predict()`).

### `inference.py`

Pokretanje predikcija i objašnjivost.

| Funkcija | Što radi |
|----------|-------------|
| `prepare_tensor(img_np)` | Pretvara (H,W,3) numpy polje → (1,3,H,W) float tenzor normaliziran u [0,1] |
| `predict(model, tensor, threshold)` | Forward prolaz → vraća `(vjerojatnost, predviđena_klasa)` |
| `apply_gradcam(model, tensor, layer)` | Generira Grad-CAM toplinsku mapu za zadani sloj. Vraća (cam, class_idx) |

### `folding.py`

Stratificirano k-struko dijeljenje na razini pacijenta.

| Funkcija | Što radi |
|----------|-------------|
| `build_patient_dict(df)` | Gradi `{patient_id: [n_klasa0, n_klasa1]}` iz DataFramea |
| `triple_stratified_fold(patient_dict, df, num_folds)` | Round-robin dodjeljivanje foldova. Pacijenti klase 1 raspoređeni prvi, zatim klasa 0. Nijedan pacijent ne pripada u više foldova. |

---

## `src/lumen/training/` — Moduli za treniranje

### `loss.py`

| Klasa | Što radi |
|-------|-------------|
| `FairnessAwareLoss` | Prilagođena težinska BCE s tri komponente: (1) težine po klasi, (2) regularizacija ravnoteže recall-a koja kažnjava lošu recall vrijednost po klasi, (3) regularizacija pravednosti (equalized odds) koja kažnjava razlike u TPR/FPR između grupa tonova kože. Augmentirani uzorci dobivaju manju težinu. |

### `augmentation.py`

Sve funkcije rade na float32 numpy poljima u rasponu [0,1].

| Funkcija | Što radi |
|----------|-------------|
| `random_rotate(img)` | Rotacija za nasumični višekratnik 90° |
| `flip_vertical(img)` | Vertikalno zrcaljenje |
| `flip_horizontal(img)` | Horizontalno zrcaljenje |
| `contrast_change(img)` | Nasumični kontrast ±2–10% |
| `brightness_change(img)` | Nasumična svjetlina ±5–10% |
| `add_gaussian_noise(img)` | Gaussov šum, std 0.01–0.05 |
| `color_jitter(img)` | Nasumična promjena nijanse/zasićenja/svjetline u HSV prostoru |
| `augment(img, n)` | Proizvodi n augmentiranih kopija: nasumična rotacija + 3+ nasumične transformacije |

### `dataset.py`

| Klasa/Funkcija | Što radi |
|----------------|-------------|
| `SkinImageDataset` | PyTorch Dataset. Učitava sve slike u memoriju pri inicijalizaciji koristeći paralelne dretve. Augmentira klasu 1 i podzastupljene grupe. Vraća `(img_tensor, label, fitz_group, is_augmented)`. |
| `load_and_prepare_image(name, fold, dir)` | Učita pojedinu predobrađenu sliku iz strukture mapa po foldovima |
| `calculate_class_weights(loader, device)` | Računa balansirane težine klasa (sklearn) iz DataLoadera |

### `evaluation.py`

| Funkcija | Što radi |
|----------|-------------|
| `evaluate_loss(loader, model, criterion, device)` | Računa prosječni loss preko DataLoadera |
| `detailed_evaluation(loader, model, device, threshold)` | Klasifikacijski izvještaj + AUC |
| `evaluate_fairness(loader, model, criterion, device)` | Preciznost, recall, F1, točnost, FPR po Fitzpatrick grupi |
| `evaluate_thresholds(model, loader, device)` | Pretražuje pragove 0.2–0.8, crta krivulje, vraća najbolji prag/F1 |
| `plot_metrics(history)` | Crta 2x2 mrežu: loss, AUC, F1, recall klase 1 kroz epohe |

### `trainer.py`

| Funkcija | Što radi |
|----------|-------------|
| `train_epoch(loader, model, criterion, optimizer, device)` | Jedna epoha treniranja. Vraća (avg_loss, predikcije, oznake). |
| `k_fold_training(metadata, base_dir, train_folds, val_folds, ...)` | Cijelo treniranje: gradi datasetove, trenira uz early stopping + smanjivanje LR-a, bilježi metrike pravednosti po epohi, sprema najbolje težine. Vraća (model, history). |

---

## `scripts/` — CLI ulazne točke

| Skripta | Svrha | Glavni argumenti |
|--------|---------|----------|
| `preprocess_dataset.py` | Sirove slike → predobrađeni foldovi + metapodaci | `--images`, `--labels`, `--duplicates` |
| `train.py` | Metapodaci + foldovi → istrenirane težine modela | `--data-dir`, `--metadata` |
| `predict.py` | Slike + težine → CSV s predikcijama | `--images`, `--weights` |

Sve skripte prihvaćaju `--help` za potpunu dokumentaciju zastavica.

---

## `web_app/` — Web aplikacija

```
web_app/
├── Dockerfile              Multi-stage build (Node → Python → runtime)
├── backend/
│   ├── app.py              FastAPI postavljanje, učitavanje modela pri startu
│   └── router.py           SSE endpoint: upload → predobrada → predikcija → Grad-CAM
└── frontend/               Vue 3 + Naive UI single-page aplikacija
```

Backend uvozi `lumen.preprocessing`, `lumen.skin_tone`, `lumen.inference` i `lumen.model` — bez duplicirane logike.

---

## Tok podataka

```
Sirove .jpg slike
      │
      ▼
┌─────────────────────┐
│  preprocess_dataset │  scripts/preprocess_dataset.py
│  (preprocessing.py) │  uklanjanje dlaka, izrezivanje, resize, ITA
│  (skin_tone.py)     │  procjena Fitzpatrick tipa
│  (folding.py)       │  k-struko dijeljenje na razini pacijenta
└─────────┬───────────┘
          ▼
   Predobrađeni foldovi + metadata.json
          │
          ▼
┌─────────────────────┐
│       train         │  scripts/train.py
│  (dataset.py)       │  učitavanje slika, augmentacija klase 1 + grupe 56
│  (augmentation.py)  │  nasumične transformacije
│  (loss.py)          │  težinska BCE svjesna pravednosti
│  (trainer.py)       │  AdamW, early stopping, LR raspoređivanje
│  (evaluation.py)    │  metrike po epohi + izvještaj o pravednosti
└─────────┬───────────┘
          ▼
     model_fold_N.pth
          │
          ▼
┌─────────────────────┐
│      predict        │  scripts/predict.py
│  (preprocessing.py) │  predobrada novih slika
│  (inference.py)     │  priprema tenzora, forward prolaz
└─────────┬───────────┘
          ▼
    predictions.csv
```
