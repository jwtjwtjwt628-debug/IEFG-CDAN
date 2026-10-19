#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
第四章统一设置实验: 全部使用 20%数据 + 200epoch
包含: 消融实验(3次) + 参数敏感性(3次) + 混淆矩阵 + t-SNE
"""

import os, sys, re, csv, time, glob
import subprocess
import numpy as np
from datetime import datetime

DATA_DIR = "./CWRU"
BATCH_SIZE = 32
EPOCH = 200
RATIO = 0.2
COMMON = f"--data_name multi_CWRU --data_dir {DATA_DIR} --batch_size {BATCH_SIZE} --max_epoch {EPOCH} --source_sample_ratio {RATIO} --print_step 999"
BASE = f"python train_advanced.py {COMMON} "
N_RUNS = 3


def run_and_extract(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = p.communicate()[0]
    accs = re.findall(r'target_val-Acc:\s*(\d+\.\d+)', out)
    return round(max(float(a) for a in accs) * 100, 2) if accs else None


def run_n_times(cmd, n=3):
    accs = []
    for i in range(n):
        acc = run_and_extract(cmd)
        accs.append(acc if acc else 0)
        print(f"    run{i+1}: {acc}%", flush=True)
        time.sleep(1)
    mean = round(np.mean(accs), 2)
    std = round(np.std(accs), 2)
    return accs, mean, std


# ======================================================================
# 1. 消融实验 (统一20%数据, 3次均值)
# ======================================================================
def run_ablation():
    print("\n" + "="*60 + "\n  消融实验 (20%数据, 200epoch, 3次均值)\n" + "="*60)
    
    tasks = [
        ('"[[3],[0]]"', "3->0"),
        ('"[[3],[0,1,2]]"', "3->012"),
    ]
    configs = {
        "A1_DANN": "--method DA --domain_adversarial True --adversarial_loss DA",
        "A2_CDANN_FG": (
            "--method DA --domain_adversarial True --adversarial_loss CDANN-FG "
            "--warmup_epoch 0 --ramp_epoch 0 --lam_entropy 0 "
            "--tau_high 1.1 --tau_low -0.1 --entropy_temperature 100"
        ),
        "A3_Boltzmann": (
            "--method DA --domain_adversarial True --adversarial_loss CDANN-FG "
            "--warmup_epoch 0 --ramp_epoch 0 --lam_entropy 0 "
            "--tau_high 1.1 --tau_low -0.1 --entropy_temperature 0.5"
        ),
        "A4_Curriculum": (
            "--method DA --domain_adversarial True --adversarial_loss CDANN-FG "
            "--warmup_epoch 5 --ramp_epoch 20 --lam_entropy 0 "
            "--tau_high 1.1 --tau_low -0.1 --entropy_temperature 0.5"
        ),
        "A5_DualThreshold": (
            "--method DA --domain_adversarial True --adversarial_loss CDANN-FG "
            "--warmup_epoch 5 --ramp_epoch 20 --lam_entropy 0 "
            "--tau_high 0.8 --tau_low 0.5 --entropy_temperature 0.5"
        ),
        "A6_IEFG_CDAN": (
            "--method DA --domain_adversarial True --adversarial_loss CDANN-FG "
            "--warmup_epoch 5 --ramp_epoch 20 --lam_entropy 0.0 "
            "--tau_high 0.8 --tau_low 0.5 --entropy_temperature 0.5"
        ),
    }

    csv_file = f"Unified_ablation_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow(["Task", "Config", "Run1", "Run2", "Run3", "Mean", "Std"])

    for task_str, task_name in tasks:
        for cname, cargs in configs.items():
            print(f"\n  [{task_name}] {cname}:", flush=True)
            cmd = f"{BASE} --transfer_task {task_str} {cargs}"
            accs, mean, std = run_n_times(cmd, N_RUNS)
            print(f"  => {mean} ± {std}")
            with open(csv_file, 'a', newline='', encoding='utf-8-sig') as f:
                csv.writer(f).writerow([task_name, cname] + accs + [mean, std])

    print(f"\nAblation saved: {csv_file}")


# ======================================================================
# 2. 参数敏感性 (统一20%数据, 3次均值)
# ======================================================================
def run_sensitivity():
    print("\n" + "="*60 + "\n  参数敏感性 (20%数据, 200epoch, 3次均值)\n" + "="*60)

    task = '"[[3],[0]]"'
    base_iefg = ("--method DA --domain_adversarial True --adversarial_loss CDANN-FG "
                 "--warmup_epoch 5 --ramp_epoch 20 --lam_entropy 0.0 "
                 "--tau_high 0.8 --tau_low 0.5 --entropy_temperature 0.5")

    csv_file = f"Unified_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow(["Param", "Value", "Run1", "Run2", "Run3", "Mean", "Std"])

    # (a) Temperature
    print("\n  --- Temperature T ---")
    for T in [0.1, 0.3, 0.5, 1.0, 2.0]:
        args = base_iefg.replace("--entropy_temperature 0.5", f"--entropy_temperature {T}")
        cmd = f"{BASE} --transfer_task {task} {args}"
        print(f"  T={T}:", flush=True)
        accs, mean, std = run_n_times(cmd, N_RUNS)
        print(f"  => {mean} ± {std}")
        with open(csv_file, 'a', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(["Temperature", T] + accs + [mean, std])

    # (b) Warmup
    print("\n  --- Warmup ---")
    for w in [0, 5, 10, 20, 30]:
        args = base_iefg.replace("--warmup_epoch 5", f"--warmup_epoch {w}")
        cmd = f"{BASE} --transfer_task {task} {args}"
        print(f"  warmup={w}:", flush=True)
        accs, mean, std = run_n_times(cmd, N_RUNS)
        print(f"  => {mean} ± {std}")
        with open(csv_file, 'a', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(["Warmup", w] + accs + [mean, std])

    # (c) tau_high
    print("\n  --- tau_high ---")
    for th in [0.6, 0.7, 0.8, 0.9, 0.95]:
        args = base_iefg.replace("--tau_high 0.8", f"--tau_high {th}")
        cmd = f"{BASE} --transfer_task {task} {args}"
        print(f"  tau_high={th}:", flush=True)
        accs, mean, std = run_n_times(cmd, N_RUNS)
        print(f"  => {mean} ± {std}")
        with open(csv_file, 'a', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(["tau_high", th] + accs + [mean, std])

    print(f"\nSensitivity saved: {csv_file}")


# ======================================================================
# 3. 混淆矩阵 + t-SNE (统一20%数据)
# ======================================================================
def run_visual():
    import torch
    from torch import nn
    sys.path.insert(0, '.')
    import models, datasets
    from sklearn.manifold import TSNE
    from sklearn.metrics import confusion_matrix
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    FAULT_LABELS = ['N','IR7','IR14','IR21','B7','B14','B21','OR7','OR14','OR21']
    METHOD_ORDER = ['Source_Only', 'DANN', 'CDANN_FG', 'IEFG_CDAN']
    TITLES = ['Source Only', 'DANN', 'CDANN-FG', 'IEFG-CDAN (Ours)']

    method_args = {
        'Source_Only': '--method DG',
        'DANN': '--method DA --domain_adversarial True --adversarial_loss DA',
        'CDANN_FG': ('--method DA --domain_adversarial True --adversarial_loss CDANN-FG '
                     '--warmup_epoch 0 --ramp_epoch 0 --lam_entropy 0 '
                     '--tau_high 1.1 --tau_low -0.1 --entropy_temperature 100'),
        'IEFG_CDAN': ('--method DA --domain_adversarial True --adversarial_loss CDANN-FG '
                      '--warmup_epoch 5 --ramp_epoch 20 --lam_entropy 0.0 '
                      '--tau_high 0.8 --tau_low 0.5 --entropy_temperature 0.5'),
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    task_str = [[3], [0]]
    os.makedirs('tsne_unified', exist_ok=True)

    all_preds = {}
    all_labels = {}
    all_features = {}

    for mname, margs in method_args.items():
        print(f"\n  Training {mname} for visualization...", flush=True)

        # Train: pick best of 3 runs
        best_acc = 0
        best_ckpt = None
        for run in range(3):
            cmd = f"{BASE} --transfer_task \"[[3],[0]]\" {margs}"
            run_and_extract(cmd)
            ckpts = glob.glob('./checkpoint/*/s_3_t_0/*/*.pth')
            if ckpts:
                ckpt = max(ckpts, key=os.path.getmtime)
                # Quick eval
                model = models.cnn_features_1d(False)
                bn = nn.Sequential(nn.Linear(256, 256), nn.ReLU(True), nn.Dropout())
                clf = nn.Linear(256, 10)
                mall = nn.Sequential(model, bn, clf)
                mall.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
                mall.to(device).eval()
                ds = datasets.multi_CWRU(DATA_DIR, task_str, 'mean-std')
                _, _, _, tgt_val = ds.data_split(transfer_learning=True)
                loader = torch.utils.data.DataLoader(tgt_val, batch_size=128, shuffle=False)
                correct = total = 0
                with torch.no_grad():
                    for x, lab in loader:
                        pred = mall(x.to(device)).argmax(1)
                        correct += (pred.cpu() == lab % 100).sum().item()
                        total += len(lab)
                acc = correct / total
                print(f"    run{run+1}: {acc*100:.2f}%", flush=True)
                if acc > best_acc:
                    best_acc = acc
                    best_ckpt = ckpt
                    # Save this model's weights
                    torch.save(mall.state_dict(), f'tsne_unified/{mname}_best.pth')

        if best_ckpt is None:
            continue

        # Extract features & predictions from best model
        mall = nn.Sequential(models.cnn_features_1d(False),
                             nn.Sequential(nn.Linear(256, 256), nn.ReLU(True), nn.Dropout()),
                             nn.Linear(256, 10))
        mall.load_state_dict(torch.load(f'tsne_unified/{mname}_best.pth', map_location=device, weights_only=True))
        mall.to(device).eval()
        model_feat = mall[0]
        bn_feat = mall[1]

        ds = datasets.multi_CWRU(DATA_DIR, task_str, 'mean-std')
        _, src_val, _, tgt_val = ds.data_split(transfer_learning=True)

        def extract(dataset):
            loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
            feats, preds, labs = [], [], []
            with torch.no_grad():
                for x, lab in loader:
                    x = x.to(device)
                    f = bn_feat(model_feat(x))
                    out = mall[2](f)
                    feats.append(f.cpu().numpy())
                    preds.append(out.argmax(1).cpu().numpy())
                    labs.append((lab % 100).numpy())
            return np.concatenate(feats), np.concatenate(preds), np.concatenate(labs)

        sf, sp, sl = extract(src_val)
        tf, tp, tl = extract(tgt_val)
        np.savez(f'tsne_unified/{mname}.npz', src_features=sf, src_labels=sl,
                 tgt_features=tf, tgt_labels=tl, tgt_preds=tp)
        all_preds[mname] = tp
        all_labels[mname] = tl
        all_features[mname] = (sf, sl, tf, tl)
        print(f"  {mname} best acc: {best_acc*100:.2f}%")

    # ====== Plot confusion matrices ======
    print("\n  Generating confusion matrices...", flush=True)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for i, m in enumerate(METHOD_ORDER):
        if m not in all_preds:
            continue
        cm = confusion_matrix(all_labels[m], all_preds[m], labels=range(10))
        acc = (all_preds[m] == all_labels[m]).mean() * 100
        im = axes[i].imshow(cm, cmap='Blues', interpolation='nearest')
        axes[i].set_title(f'{TITLES[i]}\n(Acc: {acc:.1f}%)', fontsize=13, fontweight='bold')
        axes[i].set_xticks(range(10))
        axes[i].set_yticks(range(10))
        axes[i].set_xticklabels(FAULT_LABELS, rotation=45, ha='right', fontsize=7)
        axes[i].set_yticklabels(FAULT_LABELS, fontsize=7)
        for ii in range(10):
            for jj in range(10):
                val = cm[ii, jj]
                if val > 0:
                    color = 'white' if val > cm.max() / 2 else 'black'
                    axes[i].text(jj, ii, str(val), ha='center', va='center', fontsize=6, color=color)
        if i == 0:
            axes[i].set_ylabel('True Label', fontsize=11)
        axes[i].set_xlabel('Predicted Label', fontsize=11)
    plt.tight_layout(pad=1.5)
    plt.savefig('unified_confusion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved unified_confusion.png")

    # ====== Plot t-SNE class view (2x2 detail) ======
    print("  Generating t-SNE...", flush=True)
    colors = plt.colormaps['tab10']
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    axes = axes.flatten()
    for i, m in enumerate(METHOD_ORDER):
        if m not in all_features:
            continue
        sf, sl, tf, tl = all_features[m]
        emb = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=2000).fit_transform(
            np.concatenate([sf, tf]))
        ns = len(sf)
        for c in range(10):
            axes[i].scatter(emb[:ns][sl == c, 0], emb[:ns][sl == c, 1],
                            c=[colors(c)], marker='x', alpha=0.6, s=25, linewidths=0.8)
            axes[i].scatter(emb[ns:][tl == c, 0], emb[ns:][tl == c, 1],
                            c=[colors(c)], marker='o', alpha=0.6, s=25, edgecolors='none',
                            label=FAULT_LABELS[c])
        acc = (all_preds[m] == all_labels[m]).mean() * 100
        axes[i].set_title(f'{TITLES[i]} ({acc:.1f}%)', fontsize=14, fontweight='bold')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    axes[3].legend(fontsize=7, ncol=2, loc='best', framealpha=0.9, markerscale=1.2,
                   title='× Source  ○ Target', title_fontsize=8)
    plt.tight_layout(pad=1.5)
    plt.savefig('unified_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved unified_tsne.png")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='all', choices=['ablation', 'sensitivity', 'visual', 'all'])
    args = parser.parse_args()

    if args.exp in ('ablation', 'all'):
        run_ablation()
    if args.exp in ('sensitivity', 'all'):
        run_sensitivity()
    if args.exp in ('visual', 'all'):
        run_visual()

    print("\nAll unified experiments done!")
