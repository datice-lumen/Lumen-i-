# Upute za treniranje modela preko SSH-a

## 1. Spajanje na server

```bash
ssh username@server_address
# ili s SSH ključem
ssh -i ~/.ssh/id_rsa username@server_address
```

Provjeri imaš li GPU dostupan:
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

## 4. Prijenos dataseta

ISIC 2020 je velik (~30 GB), pa najbolje:

**Opcija A — preuzmi izravno na server:**
```bash
mkdir -p ~/data && cd ~/data
wget https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip
wget https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth.csv
wget https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_Duplicates.csv
unzip ISIC_2020_Training_JPEG.zip
```

**Opcija B — rsync s lokalnog stroja:**
```bash
rsync -avz --progress /lokalni/put/ISIC2020/ username@server:~/data/ISIC2020/
```

## 5. Predprocesiranje podataka

```bash
cd ~/Lumen-i-
source .venv/bin/activate

python scripts/preprocess_dataset.py \
    --images ~/data/ISIC2020/train \
    --labels ~/data/ISIC2020/ISIC_2020_Training_GroundTruth.csv \
    --duplicates ~/data/ISIC2020/ISIC_2020_Training_Duplicates.csv \
    --output ~/data/preprocessed
```

## 6. Treniranje u perzistentnoj sesiji

SSH konekcija može pasti — koristi `tmux` ili `screen` da treniranje preživi diskonekt.

```bash
tmux new -s train
# unutar tmux sesije:
cd ~/Lumen-i-
source .venv/bin/activate

python scripts/train.py \
    --data-dir ~/data/preprocessed \
    --metadata ~/data/preprocessed/metadata.json \
    --output-dir ~/Lumen-i-/models \
    --epochs 35 \
    --batch-size 128 \
    --lr 3e-5 \
    2>&1 | tee training.log
```

**Detach od tmux sesije:** `Ctrl+B`, pa `D`
**Reattach kasnije:**
```bash
tmux attach -t train
```

Korisne tmux komande:
- `tmux ls` — popis sesija
- `tmux kill-session -t train` — gasi sesiju

## 7. Praćenje treniranja

U novom terminalu/tmux prozoru:

```bash
# GPU iskorištenost (osvježava se svake 1s)
watch -n 1 nvidia-smi

# pratiti log
tail -f ~/Lumen-i-/training.log
```

## 8. Preuzimanje istreniranih težina natrag na lokalni stroj

S **lokalnog stroja**:
```bash
rsync -avz --progress username@server:~/Lumen-i-/models/ ./models/
# ili pojedinačno
scp username@server:~/Lumen-i-/models/model_fold_0.pth ./
```

## Često korisne preporuke

- **Manji batch ako pukne OOM:** `--batch-size 64` (ili 32)
- **Specifični GPU ako server ima više:**
  ```bash
  CUDA_VISIBLE_DEVICES=0 python scripts/train.py ...
  ```
- **Provjeri prostor na disku prije pokretanja:** `df -h ~`
- **Nohup alternativa tmuxu:**
  ```bash
  nohup python scripts/train.py --data-dir ... > training.log 2>&1 &
  ```

Treniranje s defaultnim postavkama (35 epoha, 4 folda) traje nekoliko sati ovisno o GPU-u — A100/RTX 4090 ~3-5h, slabiji GPU-ovi i preko 12h.
