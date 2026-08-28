# Skin Check — multimodalni probir kožnih lezija

[🇬🇧 English](README.md) | **🇭🇷 Hrvatski**

Multimodalni sustav dubokog učenja koji procjenjuje vjerojatnost da je kožna lezija maligna na temelju **dermatoskopske snimke ili fotografije mobitelom** te neobaveznih podataka o pacijentu (dob, spol, lokacija na tijelu), upakiran u web aplikaciju koja korisniku prikazuje svaki korak obrade.

**Autori:** Filip Hlup, Jurica Jerinić, Tomislav Matanović, Karlo Raštegorac
**Mentor:** doc. dr. sc. Krešimir Križanović — Sveučilište u Zagrebu, Fakultet elektrotehnike i računarstva

🌐 **Aplikacija:** https://datice-skin-checker.onrender.com/ · 📖 **Cjelovit rad:** [wiki projekta](https://github.com/datice1/skin-check/wiki)

## Pregled

- **Podaci.** 67 085 dermatoskopskih slika objedinjenih iz skupova ISIC 2019, ISIC 2020 i MILK10k (24,6 % malignih), uz strogu podjelu na razini pacijenta (10/13 treniranje, 1/13 validacija, 2/13 test) za koju je provjereno da svaku leziju drži u jednom podskupu.
- **Model.** Zamrznuta okosnica **DINOv2-S** (globalni kontekst, 384 dim.) radi paralelno s malom treniranom mrežom **TinyCNN** (lokalna tekstura, 192 dim.); spojeni vektor od 576 dim. projicira se na 256 i ulančava sa 16-dimenzionalnim kodiranjem dobi, spola i lokacije iz **MetaMLP**-a. Trenira se samo 722 513 parametara.
- **Funkcija gubitka.** `BCE + λ·(meki FPR + w·(1 − meki TPR))` uz λ = 0,9 i w = 2,5 — meka, asimetrično ponderirana inačica Youdenova indeksa koja propušten malignom kažnjava 2,5 puta jače od lažnog alarma.
- **Prilagodba mobilnim slikama.** Na fotografijama mobitelom osjetljivost dermatoskopskog modela pada s 0,912 na 0,559. Isključivanjem uklanjanja dlačica i finim podešavanjem glava na mobilnim slikama iz MILK10k (DINOv2 ostaje zamrznut) osjetljivost na neviđenom mobilnom testu raste na 0,925.
- **Web aplikacija.** Vue 3 + FastAPI. Jedan zahtjev strujanjem (SSE) vraća korake obrade: kvadratni izrez → uklanjanje dlačica (samo dermatoskopski način) → procjena tona kože (ITA → Fitzpatrick) → predikcija → Grad-CAM. Jednoklasni detektor nad DINOv2 značajkama odbija slike koje nisu bliski snimak kože. Povijest lezija živi isključivo u pregledniku; na poslužitelju se ništa ne pohranjuje.

| Metrika | Dermatoskopski test (n = 10 326) | Mobilni test (n = 836) |
|---|---:|---:|
| Osjetljivost / TPR | **0,912** | **0,925** |
| FPR | 0,136 | 0,378 |
| Točnost | 0,876 | 0,843 |
| Preciznost | 0,687 | 0,869 |
| F1 | 0,784 | 0,896 |
| AUC | ≈ 0,94 | 0,844 |

## Struktura repozitorija

```
src/lumen/                  Glavni Python paket (pip install -e .)
  model_meta.py             Spojeni model DINOv2-S + TinyCNN + MetaMLP, Grad-CAM
  preprocessing.py          Središnji izrez, DullRazor uklanjanje dlačica, dermatoskopski / mobilni cjevovod
  skin_tone.py              ITA iz 8 perifernih područja, Fitzpatrickova ljestvica
  gating/                   Provjera je li slika koža (DINOv2 značajke + Mahalanobisov OOD detektor)
  training/fused.py         Skupovi podataka, BCEJLoss, optimizator, petlja epohe, checkpointi
scripts/                    CLI ulazne točke (predobrada, podjela, treniranje, dotreniranje, evaluacija)
web_app/
  Dockerfile                Višestupanjska izgradnja (Vue → Python)
  backend/                  FastAPI + SSE, isporučeni checkpointi
  frontend/                 Vue 3 + Naive UI jednostranična aplikacija
docs/training/              Zapisi treniranja (model_10_6, mobilno dotreniranje)
tests/                      pytest testovi
```

Opis modula nalazi se u [STRUCTURE.hr.md](STRUCTURE.hr.md), a sve CLI zastavice u [USAGE.hr.md](USAGE.hr.md).

## Brzi početak

### Instalacija

```bash
pip install -e .
```

Python 3.10+, PyTorch 2.x. GPU je potreban samo za treniranje.

### Treniranje dermatoskopskog modela

```bash
# 1. Predobrada na 448 px s uklanjanjem dlačica (jedan --images po izvornoj mapi)
python scripts/preprocess_fused_dataset.py \
    --metadata final_metadata.csv \
    --images data/2019/ISIC_2019_Training_Input \
    --images data/2020/train \
    --images data/MILK10k/MILK10k_Training_Input \
    --output preprocessed448

# 2. Stratificirana podjela grupirana po pacijentu (dodaje stupac "split")
python scripts/make_split.py --metadata final_metadata.csv

# 3. Treniranje (AdamW 3e-4, zagrijavanje + kosinus, skupina 64, rano zaustavljanje)
python scripts/train_fused.py \
    --metadata final_metadata.csv \
    --img-dir preprocessed448 \
    --output-dir runs/fused
```

### Dotreniranje i evaluacija mobilnog modela

```bash
python scripts/eval_mobile.py  --checkpoint runs/fused/checkpoint_<ts>.pt --eval-csv mobile_eval.csv --images data/MILK10k/MILK10k_Training_Input
python scripts/train_mobile.py --pretrained runs/fused/checkpoint_<ts>.pt --eval-csv mobile_eval.csv --images data/MILK10k/MILK10k_Training_Input --output-dir runs/mobile
```

### Pokretanje web aplikacije

```bash
# kontekst izgradnje je korijen repozitorija
docker build -f web_app/Dockerfile -t skin-check .
docker run -p 8000:8000 skin-check
# otvoriti http://localhost:8000
```

Težine modela isporučuju se u `web_app/backend/`; DINOv2-S se ugrađuje u sliku pri izgradnji. Za postavljanje na Render vidi `render.yaml` (potrebna je instanca s 2 GB memorije).

### Korištenje iz Pythona

```python
import cv2
import torch
from lumen.model_meta import load_fused_model, image_to_tensor, encode_metadata
from lumen.preprocessing import preprocess_mobile

model, meta_cfg = load_fused_model("web_app/backend/checkpoint_mobile_best.pt", device="cpu")
rgb = cv2.cvtColor(cv2.imread("lezija.jpg"), cv2.COLOR_BGR2RGB)
img = image_to_tensor(preprocess_mobile(rgb))
meta, meta_used = encode_metadata(54, "male", "torso", meta_cfg)  # meta_used govori koja su polja stvarno korištena
with torch.no_grad():
    prob = torch.sigmoid(model(img, meta)).item()
print(f"P(maligno) = {prob:.2f}")
```

## Dokumentacija

- [Dokumentacija projekta](https://github.com/datice1/skin-check/wiki/Dokumentacija-projekta) — motivacija, podaci, arhitektura, funkcija gubitka, eksperimenti, prilagodba mobilnim slikama, web aplikacija, rasprava (hrvatski)
- [Project Documentation](https://github.com/datice1/skin-check/wiki/Project-Documentation) — isti dokument na engleskom
- [Technical Documentation](https://github.com/datice1/skin-check/wiki/Technical-Documentation) — organizacija koda, API, naredbe za treniranje, SSE protokol, postavljanje

## Licenca

Vidi [LICENSE](LICENSE).

## Napomena

Ovo je istraživački i edukativni alat, a ne certificirani medicinski proizvod. Model nije klinički validiran; njegov izlaz je procjena vjerojatnosti, a ne dijagnoza, i ne zamjenjuje pregled kod dermatologa.
