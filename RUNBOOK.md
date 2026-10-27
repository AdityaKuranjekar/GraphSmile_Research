# Reproducing GraphSmile (and the no-text ablation)

## 1. Where the code lives

The code is on GitHub, branch `claude/gifted-galileo-l5kpj3`. The cloud session is temporary, so every change has to be pushed to that branch to survive.

```bash
git clone -b claude/gifted-galileo-l5kpj3 https://github.com/AdityaKuranjekar/GraphSmile_Research.git
cd GraphSmile_Research
```

Pull later updates with `git pull origin claude/gifted-galileo-l5kpj3`.

Datasets are **not** in the repo. The feature pickles are large and IEMOCAP needs a license, so download them yourself from the links in `README.md`.

## 2. Dataset setup (both machines)

1. Download the pickles from the README links.
2. Put the absolute path of each pickle into the matching constant at the top of `run.py`:
   `IEMOCAP_path`, `IEMOCAP4_path`, `MELD_path`, `CMUMOSEI7_path`.
3. Create the output folder if it is missing (it is also created automatically on the first run): `mkdir -p results`.

## 3. The institute GPU (IIT Patna)

I cannot reach your institute server from this cloud session, and I should not be given your password. You run the SSH steps from your own terminal:

```bash
# one-time, from your laptop: create a key so you don't need the password each time
ssh-keygen -t ed25519 -f ~/.ssh/iitp_gpu
ssh-copy-id -i ~/.ssh/iitp_gpu.pub <your_user>@<gpu_host>      # asks for the password once

# every time after that
ssh -i ~/.ssh/iitp_gpu <your_user>@<gpu_host>
```

Replace `<your_user>` and `<gpu_host>` with the details the institute gave you. If the host is only reachable on campus, connect to the institute VPN first.

On the GPU server:

```bash
git clone -b claude/gifted-galileo-l5kpj3 https://github.com/AdityaKuranjekar/GraphSmile_Research.git
cd GraphSmile_Research
python -m venv .venv && source .venv/bin/activate
pip install numpy==2.2.2 pandas==2.2.3 scikit_learn==1.6.1 tensorboardX==2.6.2.2
# torch 2.0.0 with CUDA 11.7 is not on PyPI; use the index that matches the server's CUDA
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu117
```

Run long jobs inside `tmux` or `nohup` so they survive a dropped SSH session:

```bash
tmux new -s graphsmile
python -u run.py ... 2>&1 | tee results/iemocap6_full.log
# detach with Ctrl-b d, reattach with: tmux attach -t graphsmile
```

Copy results back to your laptop:

```bash
scp -i ~/.ssh/iitp_gpu -r <your_user>@<gpu_host>:GraphSmile_Research/results ./
```

## 4. Full-model commands (paper settings)

The commands are the ones in `README.md`. Add `--modals tva` if you want to be explicit; it is now the default.

| Dataset | Command |
|---|---|
| IEMOCAP-6 | `python -u run.py --gpu 0 --port 1530 --classify emotion --dataset IEMOCAP --epochs 120 --textf_mode textf0 --loss_type emo_sen_sft --lr 1e-04 --batch_size 16 --hidden_dim 512 --win 17 17 --heter_n_layers 7 7 7 --drop 0.2 --shift_win 19 --lambd 1.0 1.0 0.7` |
| IEMOCAP-4 | `python -u run.py --gpu 0 --port 1531 --classify emotion --dataset IEMOCAP4 --epochs 120 --textf_mode textf0 --loss_type emo_sen_sft --lr 3e-04 --batch_size 16 --hidden_dim 256 --win 5 5 --heter_n_layers 4 4 4 --drop 0.2 --shift_win 10 --lambd 1.0 0.6 0.6` |
| MELD | `python -u run.py --gpu 0 --port 1532 --classify emotion --dataset MELD --epochs 50 --textf_mode textf0 --loss_type emo_sen_sft --lr 7e-05 --batch_size 16 --hidden_dim 384 --win 3 3 --heter_n_layers 5 5 5 --drop 0.2 --shift_win 3 --lambd 1.0 0.5 0.2` |
| CMU-MOSEI | `python -u run.py --gpu 0 --port 1534 --classify emotion --dataset CMUMOSEI7 --epochs 60 --textf_mode textf0 --loss_type emo_sen_sft --lr 8e-05 --batch_size 32 --hidden_dim 256 --win 5 5 --heter_n_layers 2 2 2 --drop 0.4 --shift_win 2 --lambd 1.0 0.8 1.0` |

Change `--gpu` to the index of a free GPU (`nvidia-smi` shows them).

## 5. No-text ablation (Table VI, "No Textual")

`--modals` selects which streams the model sees. Anything left out is replaced with zeros, so the architecture and parameter count stay the same and only the inputs change.

| Ablation | Flag |
|---|---|
| Full model | `--modals tva` |
| No textual | `--modals va` |
| No visual | `--modals ta` |
| No acoustic | `--modals tv` |

Example, IEMOCAP-6 without text (append to the full-model command and change only `--modals`):

```bash
python -u run.py --gpu 0 --port 1530 --classify emotion --dataset IEMOCAP --modals va --epochs 120 --textf_mode textf0 --loss_type emo_sen_sft --lr 1e-04 --batch_size 16 --hidden_dim 512 --win 17 17 --heter_n_layers 7 7 7 --drop 0.2 --shift_win 19 --lambd 1.0 1.0 0.7
```

The paper's no-text MERC numbers (Table VI, ACC / WF1):

| Dataset | Paper: no textual |
|---|---|
| IEMOCAP-6 | 55.70 / 54.23 |
| IEMOCAP-4 | 62.88 / 62.22 |
| MELD | 50.15 / 42.27 |
| CMU-MOSEI | 32.98 / 28.28 |

The paper does not say exactly how it removed the text stream. Zeroing is my interpretation. If the authors used a different method, the numbers may differ, so check their method section before you claim an exact match.

## 6. Laptop (CPU only)

You don't need a GPU to check the pipeline or run a shortened ablation. The code switches to CPU automatically when CUDA is not available, using the gloo backend and one process.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch numpy pandas scikit-learn            # plain CPU torch from PyPI works on the laptop
python -u run.py --classify emotion --dataset IEMOCAP --modals va --epochs 2 --textf_mode textf0 --loss_type emo_sen_sft --lr 1e-04 --batch_size 16 --hidden_dim 512 --win 17 17 --heter_n_layers 7 7 7 --drop 0.2 --shift_win 19 --lambd 1.0 1.0 0.7
```

A full 120-epoch run on CPU will take a very long time. Use `--epochs 2` to confirm the setup works, then decide whether to run the full ablation on the institute GPU.

## 7. Reading the output

- Each epoch prints `test_acc_emo` and `test_f1_emo`. The paper's ACC and WF1 correspond to these.
- At the end the script prints the best accuracy and F1. Results are also saved to `results/record_<date>.pk`.
- The script selects the best epoch by **test** F1, which is the protocol this code uses. It is not a clean validation split, so treat the numbers as a comparison with the paper, not as unbiased estimates.
- Expect small differences from the paper because of GPU non-determinism. Run several seeds if you need error bars.

## 8. Regenerate all tables in one go

`scripts/run_all.sh` runs the full model and the runnable ablations for all four datasets (28 runs, sequential), with L2 = 1e-3 as stated in the paper. Logs go to `results/logs/`. At the end it writes a comparison with the paper's numbers.

```bash
cd GraphSmile_Research
GPU=4 bash scripts/run_all.sh 2>&1 | tee results/run_all.out
cat results/comparison.md
```

To regenerate the comparison without rerunning anything: `python scripts/compare_to_paper.py > results/comparison.md`.

`results/comparison.md` contains:
- Table IV: per-class F1, ACC and WF1 for the four datasets, paper vs ours and the difference.
- Table V: sentiment (MSAC) per-class F1, ACC and WF1 with the Table III settings.
- Table VI: modality and loss ablations (MERC and MSAC).
- Table VII: seconds per epoch and memory.

Not in the comparison: No Res, No FC-Res and No Seg (Table VI), which the code has no switch for; the optimized-hyperparameter MSAC rows (‡); and t-SNE and confusion-matrix figures.

## 9. Extra checks and sweeps (IEMOCAP-6)

`scripts/run_sweeps.sh` runs:
- an L2 check at 1e-3 versus the repo default of 1e-4,
- network depth from 1 to 20 layers (paper Fig. 4, GraphSmile curve only),
- window sizes from 1 to 40 (paper Fig. 5).

```bash
GPU=4 bash scripts/run_sweeps.sh 2>&1 | tee results/sweeps.out
cat results/comparison.md
```
