from __future__ import annotations

import argparse
from pathlib import Path
import re
import json
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Helfer ----------

def find_latest_results_dir(base: Path) -> Optional[Path]:
    if not base.exists():
        return None
    dirs = [p for p in base.iterdir() if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.name, reverse=True)
    return dirs[0]


def ensure_fig_dir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


def savefig(fig, outdir: Path, name: str):
    outdir = ensure_fig_dir(outdir)
    path = outdir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[saved] {path}")


def load_csv_maybe(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[warn] Konnte {path} nicht laden: {e}")
    return None

def load_summary(results_dir: Path) -> Optional[pd.DataFrame]:
    return load_csv_maybe(results_dir / "experiment_summary.csv")


def load_all_train(results_dir: Path) -> Optional[pd.DataFrame]:
    frames = []
    for f in results_dir.glob("train_metrics_*_*.csv"):
        df = load_csv_maybe(f)
        if df is not None:
            frames.append(df)
    if frames:
        df = pd.concat(frames, ignore_index=True)
        return df
    return None


def load_all_eval(results_dir: Path) -> Optional[pd.DataFrame]:
    frames = []
    for f in results_dir.glob("eval_metrics_*_*.csv"):
        df = load_csv_maybe(f)
        if df is not None:
            frames.append(df)
    if frames:
        df = pd.concat(frames, ignore_index=True)
        return df
    return None


def load_actions_and_profiles(results_dir: Path) -> Optional[pd.DataFrame]:
    """Join actions_* mit zugehörigem profiles_* anhand von (variant, seed)."""
    profile_map: Dict[Tuple[str, int], pd.DataFrame] = {}
    for pf in results_dir.glob("profiles_*_*.csv"):
        m = re.match(r"profiles_(?P<variant>[^_]+)_(?P<seed>\d+)\.csv$", pf.name)
        if not m:
            continue
        v = m.group("variant"); s = int(m.group("seed"))
        pdf = load_csv_maybe(pf)
        if pdf is not None:
            profile_map[(v, s)] = pdf.rename(columns={"profile_idx":"pidx"})

    frames = []
    for af in results_dir.glob("actions_*_*.csv"):
        m = re.match(r"actions_(?P<variant>[^_]+)_(?P<seed>\d+)\.csv$", af.name)
        if not m:
            continue
        v = m.group("variant"); s = int(m.group("seed"))
        adf = load_csv_maybe(af)
        if adf is None:
            continue
        adf = adf.rename(columns={"profile_idx":"pidx"})
        pdf = profile_map.get((v, s))
        if pdf is None:
            print(f"[warn] Kein profiles_ für {v} seed {s} gefunden – Skip Join")
            adf["variant"] = v
            adf["seed"] = s
            frames.append(adf)
        else:
            j = adf.merge(pdf, on="pidx", how="left")
            j["variant"] = v
            j["seed"] = s
            frames.append(j)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return None


# ---------- Plotter ----------

def plot_learning_curves(summary: pd.DataFrame, out: Path, rolling: int = 20):
    df = summary.copy()
    df = df.sort_values(["variant", "seed", "episode"]) 

    fig, ax = plt.subplots(figsize=(10, 6))
    for (variant, seed), g in df.groupby(["variant", "seed"]):
        ax.plot(g["episode"], g["return"], alpha=0.25, linewidth=1, label=f"{variant} (seed {seed})")
    ax.set_title("Lernkurven – Returns je Folge (Seeds als transparente Linien)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([], [], frameon=False)
    savefig(fig, out, "learning_curves_all_seeds")

    grp = df.groupby(["variant", "episode"])['return']
    mean_df = grp.mean().reset_index(name='mean')
    std_df = grp.std(ddof=0).reset_index(name='std')
    agg = mean_df.merge(std_df, on=["variant", "episode"], how="left")

    fig, ax = plt.subplots(figsize=(10, 6))
    for variant, g in agg.groupby("variant"):
        g = g.sort_values("episode").copy()
        g["mean_roll"] = g["mean"].rolling(rolling, min_periods=max(1, rolling//2)).mean()
        ax.plot(g["episode"], g["mean_roll"].fillna(g["mean"]), linewidth=2, label=variant)
        ax.fill_between(g["episode"], (g["mean"]-g["std"]).values, (g["mean"]+g["std"]).values, alpha=0.15)
    ax.set_title(f"Lernkurven – Mittelwert ± Std (Rolling={rolling})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.legend()
    savefig(fig, out, "learning_curves_mean_std")

    last_n = int(df["episode"].max() * 0.2) if len(df) else 50
    last_n = max(10, min(200, last_n))
    tail = df[df["episode"] >= df["episode"].max() - last_n + 1]
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant, g in tail.groupby("variant"):
        ax.hist(g["return"], bins=30, alpha=0.5, label=variant)
    ax.set_title(f"Return-Verteilung (letzte {last_n} Folgen)")
    ax.set_xlabel("Return")
    ax.set_ylabel("Häufigkeit")
    ax.legend()
    savefig(fig, out, "returns_distribution_lastN")


def plot_train_metrics(train: pd.DataFrame, out: Path, rolling: int = 20):
    df = train.copy().sort_values(["variant", "seed", "episode"]) 

    def lineplot_metric(metric: str, title: str, fname: str):
        fig, ax = plt.subplots(figsize=(10, 6))
        g = df.groupby(["variant", "episode"])[metric].mean().reset_index()
        for variant, gg in g.groupby("variant"):
            gg = gg.sort_values("episode").copy()
            gg["roll"] = gg[metric].rolling(rolling, min_periods=max(1, rolling//2)).mean()
            ax.plot(gg["episode"], gg["roll"].fillna(gg[metric]), linewidth=2, label=variant)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(metric)
        ax.legend()
        savefig(fig, out, fname)

    lineplot_metric("return", "Training: Return (Ø über Seeds)", "train_return")
    lineplot_metric("accept_rate", "Training: Accept-Rate", "train_accept_rate")
    lineplot_metric("avg_premium", "Training: Durchschnittsprämie (nur angenommene)", "train_avg_premium")
    lineplot_metric("loss_paid_sum", "Training: Summe gezahlter Schäden je Folge", "train_loss_paid_sum")
    lineplot_metric("claims_count", "Training: Schadenzahl je Folge", "train_claims_count")
    lineplot_metric("min_capital", "Training: Minimales Kapital je Folge", "train_min_capital")
    lineplot_metric("final_capital", "Training: Endkapital je Folge", "train_final_capital")
    lineplot_metric("exploration_rate", "Training: Explorations-Rate je Folge", "train_exploration_rate")

    fig, ax = plt.subplots(figsize=(10, 6))
    tmp = df.groupby(["variant", "episode"])['bankrupt'].mean().reset_index(name='rate')
    for variant, gg in tmp.groupby("variant"):
        gg = gg.sort_values("episode").copy()
        gg["roll"] = gg["rate"].rolling(rolling, min_periods=max(1, rolling//2)).mean()
        ax.plot(gg["episode"], gg["roll"].fillna(gg["rate"]), linewidth=2, label=variant)
    ax.set_title("Training: Bankruptcy-Rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Anteil bankrupt")
    ax.legend()
    savefig(fig, out, "train_bankruptcy_rate")

    ez = df[df["variant"] == "EZ"].copy()
    if not ez.empty:
        for m, title in [
            ("ez_steps", "EZ: Explorations-Schritte je Folge"),
            ("ez_phases", "EZ: Explorations-Phasen je Folge"),
            ("ez_repeats", "EZ: Wiederhol-Schritte je Folge"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 6))
            g = ez.groupby(["episode"])[m].mean().reset_index()
            g = g.sort_values("episode")
            g["roll"] = g[m].rolling(rolling, min_periods=max(1, rolling//2)).mean()
            ax.plot(g["episode"], g["roll"].fillna(g[m]), linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("Episode")
            ax.set_ylabel(m)
            savefig(fig, out, f"train_{m}")


def plot_eval_metrics(eval_df: pd.DataFrame, out: Path):
    df = eval_df.copy()
    df.loc[df['time_to_ruin'] < 0, 'time_to_ruin'] = np.nan

    fig, ax = plt.subplots(figsize=(10, 6))
    g = df.groupby(["variant", "block_episode"])['return'].mean().reset_index()
    for variant, gg in g.groupby("variant"):
        gg = gg.sort_values("block_episode")
        ax.plot(gg["block_episode"], gg["return"], linewidth=2, label=variant)
    ax.set_title("Evaluation: Return je Block (Ø)")
    ax.set_xlabel("Block-Episode")
    ax.set_ylabel("Return")
    ax.legend()
    savefig(fig, out, "eval_return_by_block")

    fig, ax = plt.subplots(figsize=(10, 6))
    g = df.groupby(["variant", "block_episode"])['bankrupt'].mean().reset_index(name='rate')
    for variant, gg in g.groupby("variant"):
        gg = gg.sort_values("block_episode")
        ax.plot(gg["block_episode"], gg["rate"], linewidth=2, label=variant)
    ax.set_title("Evaluation: Bankruptcy-Rate je Block")
    ax.set_xlabel("Block-Episode")
    ax.set_ylabel("Anteil bankrupt")
    ax.legend()
    savefig(fig, out, "eval_bankruptcy_rate_by_block")

    fig, ax = plt.subplots(figsize=(10, 6))
    g = df.groupby(["variant", "block_episode"])['time_to_ruin'].mean().reset_index()
    for variant, gg in g.groupby("variant"):
        gg = gg.sort_values("block_episode")
        ax.plot(gg["block_episode"], gg["time_to_ruin"], linewidth=2, label=variant)
    ax.set_title("Evaluation: Time-to-Ruin (Ø, nur Ruins)")
    ax.set_xlabel("Block-Episode")
    ax.set_ylabel("Schritte bis Ruin")
    ax.legend()
    savefig(fig, out, "eval_time_to_ruin_by_block")

    for metric, title in [
        ("final_capital", "Evaluation: Endkapital (Ø)") ,
        ("min_capital",   "Evaluation: Minimales Kapital (Ø)") ,
        ("liabilities_end","Evaluation: Verbindlichkeiten am Ende (Ø)"),
        ("terminal_paid",  "Evaluation: Terminal Paid (Ø)")
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        g = df.groupby(["variant", "block_episode"])[metric].mean().reset_index()
        for variant, gg in g.groupby("variant"):
            gg = gg.sort_values("block_episode")
            ax.plot(gg["block_episode"], gg[metric], linewidth=2, label=variant)
        ax.set_title(title)
        ax.set_xlabel("Block-Episode")
        ax.set_ylabel(metric)
        ax.legend()
        savefig(fig, out, f"eval_{metric}_by_block")


def plot_actions_profiles(actions_profiles: pd.DataFrame, out: Path):
    df = actions_profiles.copy()
    df["accepted"] = (df["action"] > 0).astype(int)

    fig, ax = plt.subplots(figsize=(9, 5))
    dist = df.groupby(["variant", "action"]).size().reset_index(name='count')
    for variant, g in dist.groupby("variant"):
        ax.plot(g["action"], g["count"], marker='o', linewidth=2, label=variant)
    ax.set_title("Aktionsverteilung (Häufigkeit je Action-ID)")
    ax.set_xlabel("Action-ID")
    ax.set_ylabel("Anzahl")
    ax.legend()
    savefig(fig, out, "actions_distribution_by_variant")

    if {'risk_score'}.issubset(df.columns):
        q = pd.qcut(df['risk_score'], 10, duplicates='drop')
        tmp = df.copy()
        tmp["risk_decile"] = q
        grp = tmp.groupby(["variant", "risk_decile"])['accepted'].mean().reset_index(name='accept_rate')
        fig, ax = plt.subplots(figsize=(10, 6))
        for variant, g in grp.groupby("variant"):
            x = np.arange(len(g))
            ax.plot(x, g['accept_rate'], linewidth=2, marker='o', label=variant)
        ax.set_title("Annahme-Rate nach Risk-Score-Deciles")
        ax.set_xlabel("Risk-Score-Decile (niedrig → hoch)")
        ax.set_ylabel("Accept-Rate")
        ax.legend()
        savefig(fig, out, "accept_rate_by_risk_decile")

    if {'region'}.issubset(df.columns):
        grp = df.groupby(["variant", "region"])['accepted'].mean().reset_index(name='accept_rate')
        order = grp.groupby('region')['accept_rate'].mean().sort_values().index
        fig, ax = plt.subplots(figsize=(10, 6))
        for variant, g in grp.groupby("variant"):
            g = g.set_index('region').loc[order].reset_index()
            ax.plot(np.arange(len(g)), g['accept_rate'], marker='o', linewidth=2, label=variant)
        ax.set_title("Annahme-Rate nach Region")
        ax.set_xlabel("Region (sortiert)")
        ax.set_ylabel("Accept-Rate")
        ax.legend()
        savefig(fig, out, "accept_rate_by_region")

    if {'age'}.issubset(df.columns):
        bins = [0, 25, 35, 45, 55, 65, 200]
        labels = ["<25","25-34","35-44","45-54","55-64","65+"]
        tmp = df.copy()
        tmp['age_bin'] = pd.cut(tmp['age'], bins=bins, labels=labels, right=False)
        grp = tmp.groupby(["variant", "age_bin"])['accepted'].mean().reset_index(name='accept_rate')
        order = labels
        fig, ax = plt.subplots(figsize=(10, 6))
        for variant, g in grp.groupby("variant"):
            g = g.set_index('age_bin').loc[order].reset_index()
            ax.plot(np.arange(len(g)), g['accept_rate'], marker='o', linewidth=2, label=variant)
        ax.set_title("Annahme-Rate nach Altersgruppen")
        ax.set_xlabel("Altersgruppe")
        ax.set_ylabel("Accept-Rate")
        ax.legend()
        savefig(fig, out, "accept_rate_by_age")


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', type=str, default=None, help='Pfad zum results-Ordner (Standard: neuester unter ./results)')
    ap.add_argument('--rolling', type=int, default=20, help='Fenstergröße für gleitende Mittelwerte')
    ap.add_argument('--no-actions', action='store_true', help='Actions/Profile-Auswertung überspringen')
    args = ap.parse_args()

    base = Path('results')
    if args.dir:
        results_dir = Path(args.dir)
    else:
        results_dir = find_latest_results_dir(base)
    if results_dir is None or not results_dir.exists():
        print('[error] Kein gültiger results-Ordner gefunden. Nutze --dir <pfad> oder führe zuerst run_experiment.py aus.')
        return

    print(f"[info] results-dir: {results_dir}")
    fig_dir = ensure_fig_dir(results_dir / 'plots')

    meta_path = results_dir / 'run_meta.json'
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            print('[meta]', json.dumps(meta, indent=2))
        except Exception:
            pass

    summary = load_summary(results_dir)
    if summary is not None and not summary.empty:
        plot_learning_curves(summary, fig_dir, rolling=args.rolling)
    else:
        print('[warn] experiment_summary.csv nicht gefunden oder leer – überspringe Lernkurven')

    train_df = load_all_train(results_dir)
    if train_df is not None and not train_df.empty:
        plot_train_metrics(train_df, fig_dir, rolling=args.rolling)
    else:
        print('[warn] train_metrics_*.csv nicht gefunden – überspringe Trainingsplots')

    eval_df = load_all_eval(results_dir)
    if eval_df is not None and not eval_df.empty:
        plot_eval_metrics(eval_df, fig_dir)
    else:
        print('[warn] eval_metrics_*.csv nicht gefunden – überspringe Eval-Plots')

    if not args.no_actions:
        ap_df = load_actions_and_profiles(results_dir)
        if ap_df is not None and not ap_df.empty:
            plot_actions_profiles(ap_df, fig_dir)
        else:
            print('[warn] actions_* oder profiles_* nicht gefunden – überspringe Actions/Profile-Auswertung')

    print(f"Fertig. Alle Grafiken liegen unter: {fig_dir}")


if __name__ == '__main__':
    main()
