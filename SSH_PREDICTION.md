# Upute za predikciju (inference) preko SSH-a

## 1. Spajanje na server

```bash
ssh username@server_address
# ili s SSH ključem
ssh -i ~/.ssh/id_rsa username@server_address
```

Provjeri imaš li GPU dostupan (inference radi i na CPU-u, ali je sporiji):
```bash
nvidia-smi
```

## 2. Postavljanje koda na server

**Opcija A — git clone (preporučeno):**
```bash
git clone <repo-url> Lumen-i-
cd Lumen-i-
git checkout feature/refactor
```

**Opcija B — kopiranje s lokalnog stroja (s tvog računala):**
```bash
rsync -avz --exclude='.git' --exclude='__pycache__' \
    /home/hlupek/Study/Lumen-i-/ username@server:~/Lumen-i-/
```

## 3. Python okruženje

```bash
cd ~/Lumen-i-
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Provjeri da PyTorch vidi GPU:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Ako CUDA nije dostupna, instaliraj odgovarajuću verziju PyTorcha (npr. za CUDA 12.1):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 4. Prijenos težina modela na server

Trebaš `.pth` datoteku s istreniranim težinama. Ako si trenirao na istom serveru, već je u `~/Lumen-i-/models/`.

**Inače, s lokalnog stroja:**
```bash
scp ./models/model_fold_0.pth username@server:~/Lumen-i-/models/
# ili više datoteka odjednom
rsync -avz --progress ./models/ username@server:~/Lumen-i-/models/
```

## 5. Prijenos slika za predikciju

**Opcija A — rsync s lokalnog stroja:**
```bash
rsync -avz --progress /lokalni/put/slike/ username@server:~/data/inference_images/
```

**Opcija B — pojedinačno preko scp:**
```bash
scp ./slike/*.jpg username@server:~/data/inference_images/
```

**Opcija C — preuzmi izravno na serveru** (ako su slike negdje online):
```bash
mkdir -p ~/data/inference_images
cd ~/data/inference_images
wget <url-do-slike>
```

## 6. Pokretanje predikcije

Slike moraju biti `.jpg` format. Skripta automatski koristi paralelnu predobradu (~70% CPU jezgri).

```bash
cd ~/Lumen-i-
source .venv/bin/activate

python scripts/predict.py \
    --images ~/data/inference_images \
    --weights ~/Lumen-i-/models/model_fold_0.pth \
    --output ~/Lumen-i-/predictions.csv
```

| Flag | Obavezno | Default | Opis |
|------|----------|---------|------|
| `--images` | da | — | Folder s `.jpg` slikama |
| `--weights` | da | — | Put do `.pth` težina |
| `--output` | ne | `predictions.csv` | Put do izlaznog CSV-a |
| `--threshold` | ne | `0.5` | Klasifikacijski prag |
| `--no-parallel` | ne | — | Isključi multiprocessing |

### Za veće setove slika — perzistentna sesija

Ako predikcija na velikom setu traje dugo, koristi `tmux`:

```bash
tmux new -s predict
# unutar tmux sesije:
cd ~/Lumen-i-
source .venv/bin/activate

python scripts/predict.py \
    --images ~/data/inference_images \
    --weights ~/Lumen-i-/models/model_fold_0.pth \
    --output ~/Lumen-i-/predictions.csv \
    2>&1 | tee predict.log
```

**Detach:** `Ctrl+B`, pa `D`
**Reattach:** `tmux attach -t predict`

## 7. Ensemble predikcija (više foldova)

Ako želiš kombinirati predikcije iz više foldova, pokreni svaki zasebno pa usrednji rezultate:

```bash
for fold in 0 1 2 3; do
    python scripts/predict.py \
        --images ~/data/inference_images \
        --weights ~/Lumen-i-/models/model_fold_${fold}.pth \
        --output ~/Lumen-i-/predictions_fold_${fold}.csv
done
```

## 8. Praćenje napretka

U novom terminalu/tmux prozoru:

```bash
# GPU iskorištenost
watch -n 1 nvidia-smi

# pratiti log
tail -f ~/Lumen-i-/predict.log
```

Skripta ispisuje progress svakih 500 slika.

## 9. Preuzimanje rezultata na lokalni stroj

S **lokalnog stroja**:
```bash
scp username@server:~/Lumen-i-/predictions.csv ./
# ili više datoteka
rsync -avz --progress username@server:~/Lumen-i-/predictions_*.csv ./
```

## Format izlaza

CSV sadrži stupce:
- `image_name` — naziv slike bez ekstenzije
- `target` — `0` (benign) ili `1` (malignant)

Primjer:
```csv
image_name,target
ISIC_0000001,0
ISIC_0000002,1
ISIC_0000003,0
```

## Često korisne preporuke

- **CPU-only inference (bez GPU-a):** radi bez izmjena, ali je sporiji
- **Specifični GPU ako server ima više:**
  ```bash
  CUDA_VISIBLE_DEVICES=0 python scripts/predict.py ...
  ```
- **Ako je RAM ograničen:** koristi `--no-parallel` da se smanji potrošnja memorije
- **Drugačiji prag osjetljivosti:** `--threshold 0.3` (niži = više malignih predikcija, viša osjetljivost)
- **Provjeri prostor na disku prije prijenosa:** `df -h ~`

Brzina inference-a: ~50-200 slika/sekundi na modernom GPU-u, ~5-20 slika/sekundi na CPU-u.
