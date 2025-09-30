#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YAML で指定した設定に基づき、データ分割（train/val）を作成 or 再生するスクリプト。
- 症例ディレクトリ直下に *-t1n.nii.gz がある前提（pattern で変更可）
- デフォルトはシンボリックリンク（Windows 等で不可なら copy: true で実体コピー）
- "create" モード: 乱数分割（seed 固定・リスト保存）
- "replay" モード: 既存の train.txt / val.txt を読んで同じ分割を再現
- 任意で層化分割（サイト/ベンダ等の CSV による）対応
"""

import os, glob, argparse, random, math, csv, sys
from typing import List, Dict, Tuple, Optional

try:
    import yaml
except ImportError as e:
    print("[ERROR] PyYAML が必要です: pip install pyyaml", file=sys.stderr)
    raise

# ------------------------------
# 探索 & ユーティリティ
# ------------------------------
def subjects_under(root: str, pattern: str) -> List[str]:
    subs = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        if glob.glob(os.path.join(d, pattern)):
            subs.append(os.path.basename(d))
    return subs

def safe_unlink(path: str):
    try:
        if os.path.lexists(path):
            if os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                # ディレクトリの実体は消さない（上書き時は先に rmtree など使う）
                pass
            else:
                os.unlink(path)
    except OSError:
        pass

def link_tree(src_root: str, subjs: List[str], dst_root: str, copy: bool = False):
    os.makedirs(dst_root, exist_ok=True)
    if copy:
        import shutil
    for s in subjs:
        src = os.path.join(src_root, s)
        dst = os.path.join(dst_root, s)
        # 既存を掃除
        if os.path.lexists(dst):
            if os.path.islink(dst):
                os.unlink(dst)
            elif os.path.isdir(dst):
                if copy:
                    import shutil
                    shutil.rmtree(dst)
            else:
                os.unlink(dst)
        if copy:
            import shutil
            shutil.copytree(src, dst)
        else:
            try:
                os.symlink(src, dst, target_is_directory=True)
            except OSError:
                # ディレクトリ symlink 不可な環境 → 中身のファイルに symlink
                os.makedirs(dst, exist_ok=True)
                for f in glob.glob(os.path.join(src, "*")):
                    ln = os.path.join(dst, os.path.basename(f))
                    if os.path.lexists(ln):
                        os.unlink(ln)
                    os.symlink(f, ln)

def write_lists(out_dir: str, train_subs: List[str], val_subs: List[str], seed: int, src_root: str, all_discovered: List[str]):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_subs) + "\n")
    with open(os.path.join(out_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_subs) + "\n")
    with open(os.path.join(out_dir, "split_manifest.txt"), "w") as f:
        f.write(f"mode=create\nseed={seed}\nsrc_root={src_root}\n")
        f.write(f"train={len(train_subs)} val={len(val_subs)} total={len(train_subs)+len(val_subs)}\n")
    with open(os.path.join(out_dir, "discovered_all.txt"), "w") as f:
        f.write("\n".join(all_discovered) + "\n")

def read_list(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

# ------------------------------
# 層化分割（任意）
# ------------------------------
def load_groups_from_csv(csv_path: str, subject_col: str, group_col: str) -> Dict[str, str]:
    mapping = {}
    with open(csv_path, newline="") as cf:
        rdr = csv.DictReader(cf)
        assert subject_col in rdr.fieldnames and group_col in rdr.fieldnames, \
            f"CSV に必要な列がありません: {subject_col}, {group_col}"
        for row in rdr:
            sid = row[subject_col].strip()
            grp = row[group_col].strip()
            if sid:
                mapping[sid] = grp
    return mapping

def largest_remainder_alloc(total_target: int, counts: Dict[str, int], ratios: Dict[str, float]) -> Dict[str, int]:
    # グループごとに target = total_target * ratio を割当（端数は最大剰余法）
    raw = {g: total_target * ratios[g] for g in counts}
    base = {g: min(counts[g], int(math.floor(raw[g]))) for g in counts}
    rem = total_target - sum(base.values())
    if rem <= 0:
        return base
    # 端数の大きい順に +1
    fracs = sorted(((g, raw[g]-base[g]) for g in counts), key=lambda x: x[1], reverse=True)
    i = 0
    while rem > 0 and i < len(fracs):
        g = fracs[i][0]
        if base[g] < counts[g]:
            base[g] += 1
            rem -= 1
        i += 1
    return base

def stratified_split(subs: List[str], groups: Dict[str, str], train_target: int, seed: int) -> Tuple[List[str], List[str]]:
    # subs をグループで分け、各グループ内をシャッフルして目標数を配分
    by_grp: Dict[str, List[str]] = {}
    for s in subs:
        g = groups.get(s, "__UNGROUPED__")
        by_grp.setdefault(g, []).append(s)
    # グループごとの比率 = 件数 / 全体
    total = len(subs)
    ratios = {g: len(lst)/total for g, lst in by_grp.items()}
    counts = {g: len(lst) for g, lst in by_grp.items()}
    alloc = largest_remainder_alloc(train_target, counts, ratios)

    rnd = random.Random(seed)
    train, val = [], []
    for g, lst in by_grp.items():
        rnd.shuffle(lst)
        k = min(alloc.get(g, 0), len(lst))
        train.extend(lst[:k])
        val.extend(lst[k:])
    return train, val

# ------------------------------
# メイン
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML 設定ファイル")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    split = cfg.get("split", {})
    mode = split.get("mode", "create")  # create | replay

    # 共通
    src_root = split.get("src_root", "")
    pattern  = split.get("pattern", "*-t1n.nii.gz")
    train_out = split.get("train_out", "")
    val_out   = split.get("val_out", "")
    copy_flag = bool(split.get("copy", False))

    assert src_root and train_out and val_out, "src_root/train_out/val_out は必須です"

    # すべての症例（安定のため必ず sorted）
    all_subs = subjects_under(src_root, pattern)
    if len(all_subs) == 0:
        raise SystemExit(f"[ERROR] src_root={src_root} に pattern={pattern} を満たす症例が見つかりません")

    if mode == "replay":
        # ------------------ 再生モード ------------------
        rp = split.get("replay", {})
        from_dir = rp.get("from_dir", None)
        train_list = rp.get("train_list", None)
        val_list   = rp.get("val_list", None)
        if from_dir:
            train_list = train_list or os.path.join(from_dir, "train.txt")
            val_list   = val_list   or os.path.join(from_dir, "val.txt")
        assert train_list and val_list and os.path.exists(train_list) and os.path.exists(val_list), \
            "replay: train_list/val_list が見つかりません"

        train_subs = read_list(train_list)
        val_subs   = read_list(val_list)

        # 欠損があっても実在のみで構築（警告）
        all_set = set(all_subs)
        miss_train = [s for s in train_subs if s not in all_set]
        miss_val   = [s for s in val_subs   if s not in all_set]
        if miss_train or miss_val:
            print("[WARN] 一部症例が src_root に見つかりません。")
            if miss_train: print("  missing train:", miss_train[:10], "..." if len(miss_train)>10 else "")
            if miss_val:   print("  missing val  :", miss_val[:10], "..." if len(miss_val)>10 else "")

        link_tree(src_root, [s for s in train_subs if s in all_set], train_out, copy=copy_flag)
        link_tree(src_root, [s for s in val_subs   if s in all_set], val_out,   copy=copy_flag)
        print(f"[replayed] train={len(train_subs)} -> {train_out} , val={len(val_subs)} -> {val_out}")
        return

    # ------------------ 作成モード ------------------
    seed = int(split.get("seed", 42))
    train_size = split.get("train_size", None)
    train_ratio = split.get("train_ratio", None)

    if train_size is None and train_ratio is None:
        raise SystemExit("create: train_size か train_ratio のどちらかを指定してください")

    if train_size is None and train_ratio is not None:
        train_size = int(round(len(all_subs) * float(train_ratio)))
    train_size = int(train_size)
    assert 0 < train_size < len(all_subs), f"train_size が不正です: {train_size} / total {len(all_subs)}"

    # 層化オプション
    strat = split.get("stratify", None)

    if strat and strat.get("metadata_csv"):
        csv_path = strat["metadata_csv"]
        subj_col = strat.get("subjects_column", "subject_id")
        group_col = strat.get("group_column", "site")
        groups = load_groups_from_csv(csv_path, subj_col, group_col)
        train_subs, val_subs = stratified_split(all_subs, groups, train_size, seed)
    else:
        rnd = random.Random(seed)
        subs = all_subs[:]  # copy
        rnd.shuffle(subs)
        train_subs = subs[:train_size]
        val_subs   = subs[train_size:]

    # リスト保存（lists_dir 未指定なら train_out の親に保存）
    lists_dir = split.get("lists_dir") or os.path.dirname(os.path.abspath(train_out))
    write_lists(lists_dir, train_subs, val_subs, seed, src_root, all_subs)

    # リンク作成
    link_tree(src_root, train_subs, train_out, copy=copy_flag)
    link_tree(src_root, val_subs,   val_out,   copy=copy_flag)

    print(f"[created] train={len(train_subs)} -> {train_out} , val={len(val_subs)} -> {val_out}")
    print(f"[lists saved] {os.path.join(lists_dir, 'train.txt')} , {os.path.join(lists_dir, 'val.txt')}")
    print(f"[discovered_all] {os.path.join(lists_dir, 'discovered_all.txt')}")
    
if __name__ == "__main__":
    main()
