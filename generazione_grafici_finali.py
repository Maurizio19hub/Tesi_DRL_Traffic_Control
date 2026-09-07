"""
genera_tutti_grafici.py
─────────────────────────────────────────────────────────────
Script unico per generare tutti i grafici finali per due modelli:
- PPO_energy_pedwaiting
- PPO_energy_pedwaiting_collisions

Ogni modello avrà la sua cartella principale in risultati_finali/
con le sottocartelle: training (solo per pedwaiting), reward,
confronto_PPO_baseline-opt e confronto_PPO_baseline.
─────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3.common.monitor import load_results
import warnings
warnings.filterwarnings("ignore")

# ── Stile e palette ─────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.0)
palette = sns.color_palette("Set2")
COLOR_TRAIN = palette[0]
COLOR_REWARD = palette[1]
COLOR_BASE = palette[2]
COLOR_PPO = palette[3]

BASE_OUT = "risultati_finali"
os.makedirs(BASE_OUT, exist_ok=True)
print(f"📁 Cartella principale creata in: {BASE_OUT}")


# ════════════════════════════════════════════════════════════════
#  1. TRAINING METRICS
# ════════════════════════════════════════════════════════════════
def generate_training_metrics(output_base):
    """Genera i grafici delle metriche fisiche di training, suddivisi in due figure."""
    print("\n📈 Generazione metriche di training...")
    CSV_PATH = "models/t20energy-pedwaiting_f2_trstats/training_physical_metrics.csv"
    if not os.path.exists(CSV_PATH):
        print(f"⚠️  File non trovato: {CSV_PATH}, salto.")
        return

    df = pd.read_csv(CSV_PATH)
    window = 20

    # Converte emissioni da mg a g
    df['total_CO2'] = df['total_CO2'] / 1000.0
    df['total_NOx'] = df['total_NOx'] / 1000.0
    df['total_PMx'] = df['total_PMx'] / 1000.0
    df['total_fuel'] = df['total_fuel'] / 1000.0

    # Gruppo 1: metriche di traffico
    traffic_metrics = [
        'episode_reward', 'mean_waiting_time', 'mean_queue',
        'max_queue', 'mean_speed', 'arrived_vehicles'
    ]

    # Gruppo 2: metriche ambientali e di sicurezza
    env_safety_metrics = [
        'total_CO2', 'total_NOx', 'total_PMx', 'total_fuel',
        'collisions', 'teleports', 'phase_changes', 'pedestrian_waiting'
    ]

    out_dir = os.path.join(BASE_OUT, output_base, "training")
    os.makedirs(out_dir, exist_ok=True)

    _plot_metrics_group(
        df, traffic_metrics, window,
        out_path=os.path.join(out_dir, "training_metrics_traffic.png")
    )
    _plot_metrics_group(
        df, env_safety_metrics, window,
        out_path=os.path.join(out_dir, "training_metrics_env_safety.png")
    )


def _plot_metrics_group(df, metrics, window, out_path, n_cols=2):
    """Genera una figura con subplot per il gruppo di metriche indicato."""
    metrics = [m for m in metrics if m in df.columns]
    n_metrics = len(metrics)
    if n_metrics == 0:
        print(f"⚠️  Nessuna metrica trovata per {out_path}, salto.")
        return

    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5*n_cols, 3.3*n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        smoothed = df[metric].rolling(window=window, min_periods=1).mean()
        std = df[metric].rolling(window=window, min_periods=1).std()
        ax.plot(df['episode'], smoothed, color=COLOR_TRAIN, linewidth=1.5,
                 label=f'media mobile {window}')
        ax.fill_between(df['episode'], smoothed - std, smoothed + std,
                          alpha=0.2, color=COLOR_TRAIN)
        ax.set_xlabel('Episodio', fontsize=9)
        ax.set_ylabel(metric, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(metric, fontsize=10)

    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Salvato: {out_path}")


# ════════════════════════════════════════════════════════════════
#  2. REWARD CURVE
# ════════════════════════════════════════════════════════════════
def generate_reward_curve(monitor_dir, output_base):
    """Genera la curva di reward dal monitor.csv del modello."""
    print(f"\n📊 Generazione curva reward per {output_base}...")
    if not os.path.exists(monitor_dir):
        print(f"⚠️  Directory monitor non trovata: {monitor_dir}, salto.")
        return

    try:
        df = load_results(monitor_dir)
    except Exception as e:
        print(f"   Impossibile caricare i risultati: {e}, salto.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['r'], alpha=0.4, color=COLOR_REWARD, label='Reward per episodio')
    window = 10
    rolling = df['r'].rolling(window, min_periods=1).mean()
    ax.plot(rolling, color='#D35400', linewidth=2, label=f'Media mobile ({window} ep)')
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Reward')
    ax.set_title('Andamento della Reward durante il Training')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.join(BASE_OUT, output_base, "reward")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "training_reward.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Salvato: {out_path}")


# ════════════════════════════════════════════════════════════════
#  3. CONFRONTO PPO vs BASELINE (generico)
# ════════════════════════════════════════════════════════════════
def generate_comparison(baseline_csv, baseline_npz, ppo_csv, ppo_npz, output_subdir, output_base):
    """
    Genera tutti i grafici di confronto per una data coppia baseline/PPO.
    """
    print(f"\n📊 Generazione confronto per {output_base}/{output_subdir}")
    OUTPUT_DIR = os.path.join(BASE_OUT, output_base, output_subdir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Carica dati
    try:
        df_base = pd.read_csv(baseline_csv)
        df_ppo  = pd.read_csv(ppo_csv)
    except FileNotFoundError as e:
        print(f"⚠️  File CSV mancante: {e}, salto questo confronto.")
        return

    # Carica serie temporali
    HAS_TIMESERIES = False
    try:
        ts_base = np.load(baseline_npz)
        ts_ppo  = np.load(ppo_npz)
        HAS_TIMESERIES = True
    except FileNotFoundError:
        pass

    METRICS = [
        ("avg_waiting_time", "Tempo medio di attesa veicoli", "s",   True),
        ("avg_queue_length",  "Lunghezza media code",          "m",   True),
        ("max_queue_length",  "Lunghezza massima code",        "m",   True),
        ("avg_speed",         "Velocità media",                "m/s", False),
        ("total_arrived",     "Veicoli arrivati",              "n°", False),
        ("avg_ped_waiting",   "Numero medio pedoni in attesa",           "n°",   True),
        ("total_co2",         "Emissioni CO2 totali",          "g",  True),
        ("total_nox",         "Emissioni NOx totali",          "g",  True),
        ("total_pmx",         "Emissioni PMx totali",          "g",  True),
        ("total_fuel",        "Consumo carburante",            "g",  True),
        ("total_collisions",  "Collisioni totali",             "n°",  True),
        ("total_teleports",   "Teleport totali",               "n°",  True),
        ("phase_changes",     "Cambi fase semaforica",         "n°",  True),
    ]

    # Converte il consumo di carburante da mg a g
    df_base['total_fuel'] = df_base['total_fuel'] / 1000.0
    df_ppo['total_fuel']  = df_ppo['total_fuel'] / 1000.0

    df_base['total_co2'] = df_base['total_co2'] / 1000.0
    df_ppo['total_co2'] = df_ppo['total_co2'] / 1000.0

    df_base['total_nox'] = df_base['total_nox'] / 1000.0
    df_ppo['total_nox'] = df_ppo['total_nox'] / 1000.0

    df_base['total_pmx'] = df_base['total_pmx'] / 1000.0
    df_ppo['total_pmx'] = df_ppo['total_pmx'] / 1000.0


    # ---- 3.1 Barre confronto ----
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    for ax, (col, label, unit, _) in zip(axes, METRICS):
        base_mean, base_std = df_base[col].mean(), df_base[col].std()
        ppo_mean,  ppo_std  = df_ppo[col].mean(),  df_ppo[col].std()
        data = pd.DataFrame({
            "Gruppo": ["Baseline", "PPO"],
            "Media": [base_mean, ppo_mean],
            "Std":   [base_std,  ppo_std]
        })
        sns.barplot(x="Gruppo", y="Media", data=data, ax=ax,
                    palette=[COLOR_BASE, COLOR_PPO], ci=None)
        for bar, err in zip(ax.patches, data["Std"]):
            y = bar.get_height()
            low_err = min(err, y)   # clip a zero
            high_err = err
            ax.errorbar(bar.get_x() + bar.get_width()/2, y,
                        yerr=[[low_err], [high_err]], 
                        fmt='none', capsize=5, color='black', elinewidth=1.2)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel(unit, fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    for ax in axes[len(METRICS):]:
        ax.axis('off')
    fig.suptitle("Confronto PPO vs Baseline — Media ± Deviazione Standard", fontsize=14)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "bar_comparison_all.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✅ Salvato: {out}")

    # ---- 3.2 Boxplot ----
    def melt_metrics(df, label):
        melted = []
        for col, name, unit, _ in METRICS:
            for val in df[col]:
                melted.append({"Metrica": name, "Valore": val, "Gruppo": label})
        return pd.DataFrame(melted)
    df_melt_base = melt_metrics(df_base, "Baseline")
    df_melt_ppo  = melt_metrics(df_ppo, "PPO")
    df_melt_all  = pd.concat([df_melt_base, df_melt_ppo], ignore_index=True)

    fig, axes = plt.subplots(5, 3, figsize=(12, 14))
    axes = axes.flatten()
    for ax, (col, label, unit, _) in zip(axes, METRICS):
        data = df_melt_all[df_melt_all["Metrica"] == label]
        sns.boxplot(x="Gruppo", y="Valore", data=data, ax=ax,
                    palette=[COLOR_BASE, COLOR_PPO], width=0.6)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel(unit, fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    for ax in axes[len(METRICS):]:
        ax.axis('off')
    fig.suptitle("Distribuzione delle Metriche sui Seed — Boxplot", fontsize=14)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "boxplot_comparison_all.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✅ Salvato: {out}")

    # ---- 3.3 Violin plot ----
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    for ax, (col, label, unit, _) in zip(axes, METRICS):
        data = df_melt_all[df_melt_all["Metrica"] == label]
        sns.violinplot(x="Gruppo", y="Valore", data=data, ax=ax,
                       palette=[COLOR_BASE, COLOR_PPO], inner="quart", cut=0)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel(unit, fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    for ax in axes[len(METRICS):]:
        ax.axis('off')
    fig.suptitle("Distribuzione delle Metriche sui Seed — Violin Plot", fontsize=14)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "violin_comparison_all.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✅ Salvato: {out}")

    # ---- 3.4 Miglioramento percentuale ----
    labels_plot, improvements, colors_plot = [], [], []
    for col, label, unit, lower_better in METRICS:
        if lower_better is None:
            continue
        base_mean = df_base[col].mean()
        ppo_mean  = df_ppo[col].mean()
        if base_mean == 0:
            continue
        if lower_better:
            pct = (base_mean - ppo_mean) / abs(base_mean) * 100
        else:
            pct = (ppo_mean - base_mean) / abs(base_mean) * 100
        labels_plot.append(label)
        improvements.append(pct)
        colors_plot.append("#2E7D32" if pct >= 0 else "#C62828")

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5*len(labels_plot))))
    df_imp = pd.DataFrame({"Metrica": labels_plot, "Miglioramento %": improvements})
    sns.barplot(data=df_imp, y="Metrica", x="Miglioramento %", ax=ax,
                palette=colors_plot, orient='h')
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel("Miglioramento percentuale PPO rispetto a Baseline (%)")
    ax.set_title("Riepilogo Miglioramenti — PPO vs Baseline", fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(improvements):
        ax.text(v + (0.5 if v >= 0 else -0.5), i, f"{v:+.1f}%",
                va='center', ha='left' if v >= 0 else 'right', fontsize=9)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "improvement_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✅ Salvato: {out}")
    # Salva CSV
    summary_df = pd.DataFrame({
        "metrica": labels_plot,
        "baseline_mean": [df_base[c].mean() for c, l, u, lb in METRICS if lb is not None and df_base[c].mean() != 0],
        "ppo_mean": [df_ppo[c].mean() for c, l, u, lb in METRICS if lb is not None and df_base[c].mean() != 0],
        "miglioramento_%": improvements,
    })
    out_csv = os.path.join(OUTPUT_DIR, "improvement_summary.csv")
    summary_df.to_csv(out_csv, index=False)

    # ---- 3.5 Serie temporali ----
    if HAS_TIMESERIES:
        ts_metrics = [
            ("waiting", "Tempo medio di attesa veicoli", "s"),
            ("queue_len", "Lunghezza media code", "m"),
            ("speed", "Velocità media veicoli", "m/s"),
            ("co2", "Emissioni CO₂ per step", "g"),
            ("collisions", "Collisioni per step", "n°"),
            ("arrived", "Veicoli arrivati (cumulativo)", "n°"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        
        for ax, (key, label, unit) in zip(axes, ts_metrics):
            base_data = ts_base[key]
            ppo_data  = ts_ppo[key]
            if key == "arrived":
                base_data = np.cumsum(base_data, axis=1)
                ppo_data  = np.cumsum(ppo_data, axis=1)

            # rimozione ultimo punto (artefatto di fine episodio)
            base_data = base_data[:, :-1]
            ppo_data  = ppo_data[:, :-1]

            T = base_data.shape[1]
            steps = np.arange(T)
            mean_base, std_base = base_data.mean(axis=0), base_data.std(axis=0)
            mean_ppo,  std_ppo  = ppo_data.mean(axis=0),  ppo_data.std(axis=0)
            # Per collisioni (e teleport) la banda inferiore non deve scendere sotto 0
            if key in ["collisions", "teleports"]:
                low_base = np.clip(mean_base - std_base, 0, None)
                high_base = mean_base + std_base
                low_ppo = np.clip(mean_ppo - std_ppo, 0, None)
                high_ppo = mean_ppo + std_ppo
            else:
                low_base = mean_base - std_base
                high_base = mean_base + std_base
                low_ppo = mean_ppo - std_ppo
                high_ppo = mean_ppo + std_ppo

            ax.plot(steps, mean_base, color=COLOR_BASE, linewidth=1.5, label="Baseline")
            ax.fill_between(steps, low_base, high_base, alpha=0.2, color=COLOR_BASE)
            ax.plot(steps, mean_ppo, color=COLOR_PPO, linewidth=1.5, label="PPO")
            ax.fill_between(steps, low_ppo, high_ppo, alpha=0.2, color=COLOR_PPO)
            ax.set_xlabel("Step simulazione")
            ax.set_ylabel(f"{label} ({unit})")
            ax.set_title(label)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
        for ax in axes[len(ts_metrics):]:
            ax.axis('off')
        fig.suptitle("Confronto Serie Temporali — Media ± Deviazione Standard", fontsize=14)
        
        


        fig.text(0.5, 0.01, 
                 "La linea rappresenta la media sui seed di test, l'area ombreggiata la deviazione standard",
                 ha='center', fontsize=10, style='italic')
        
        plt.tight_layout(rect=[0, 0.03, 1, 1]) 
        out = os.path.join(OUTPUT_DIR, "timeseries_comparison.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   ✅ Salvato: {out}")

    # ---- 3.6 Tabella riassuntiva ----
    print("\n   Generazione tabella riassuntiva...")
    rows = []
    for col, label, unit, lower_better in METRICS:
        if col not in df_base.columns or col not in df_ppo.columns:
            continue
        base_mean, base_std = df_base[col].mean(), df_base[col].std()
        ppo_mean,  ppo_std  = df_ppo[col].mean(),  df_ppo[col].std()
        if base_mean == 0:
            delta_pct = float("nan")
        else:
            if lower_better is None:
                delta_pct = float("nan")
            elif lower_better:
                delta_pct = (base_mean - ppo_mean) / abs(base_mean) * 100
            else:
                delta_pct = (ppo_mean - base_mean) / abs(base_mean) * 100
        rows.append({
            "Metrica": label,
            "Unità":   unit,
            "Baseline (media)": round(base_mean, 3),
            "Baseline (std)":   round(base_std, 3),
            "PPO (media)":      round(ppo_mean, 3),
            "PPO (std)":        round(ppo_std, 3),
            "Δ% PPO vs Baseline": round(delta_pct, 1) if not np.isnan(delta_pct) else None,
        })
    summary = pd.DataFrame(rows)

    summary.to_csv(os.path.join(OUTPUT_DIR, "final_summary_table.csv"), index=False)
    summary["Baseline"] = summary.apply(
        lambda r: f"{r['Baseline (media)']:.2f} ± {r['Baseline (std)']:.2f}", axis=1)
    summary["PPO"] = summary.apply(
        lambda r: f"{r['PPO (media)']:.2f} ± {r['PPO (std)']:.2f}", axis=1)
    compact = summary[["Metrica", "Unità", "Baseline", "PPO", "Δ% PPO vs Baseline"]]
    compact.to_csv(os.path.join(OUTPUT_DIR, "final_summary_table_compact.csv"), index=False)

    # LaTeX
    latex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Confronto delle metriche tra Baseline e PPO, media $\pm$ std su " + str(len(df_base)) + r" seed.}",
        r"\label{tab:final_comparison}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Metrica} & \textbf{Baseline} & \textbf{PPO} & \textbf{$\Delta\%$ Miglioramento} \\",
        r"\midrule"
    ]
    for _, r in summary.iterrows():
        if pd.isna(r["Δ% PPO vs Baseline"]):
            continue
        metrica = f"{r['Metrica']} ({r['Unità']})"
        base_str = f"{r['Baseline (media)']:.2f} $\\pm$ {r['Baseline (std)']:.2f}"
        ppo_str  = f"{r['PPO (media)']:.2f} $\\pm$ {r['PPO (std)']:.2f}"
        delta_str = f"{r['Δ% PPO vs Baseline']:+.1f}\\%"
        latex_lines.append(f"{metrica} & {base_str} & {ppo_str} & {delta_str} \\\\")
    latex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(OUTPUT_DIR, "final_summary_table.tex"), "w") as f:
        f.write("\n".join(latex_lines))

    # Immagine tabella
    table_data = []
    for _, r in summary.iterrows():
        if pd.isna(r["Δ% PPO vs Baseline"]):
            continue
        table_data.append([
            f"{r['Metrica']} ({r['Unità']})",
            f"{r['Baseline (media)']:.2f} ± {r['Baseline (std)']:.2f}",
            f"{r['PPO (media)']:.2f} ± {r['PPO (std)']:.2f}",
            f"{r['Δ% PPO vs Baseline']:+.1f}%",
        ])

    col_labels = ["Metrica", "Baseline (media ± std)", "PPO (media ± std)", "Δ% Miglioramento PPO vs Baseline"]
    fig_height = 0.45 * len(table_data) + 1.2
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.axis("off")
    tbl = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor("#1565C0")
        cell.set_text_props(color="white", fontweight="bold")

    for i, row in enumerate(table_data, start=1):
        for j in range(len(col_labels)):
            cell = tbl[i, j]
            if i % 2 == 0:
                cell.set_facecolor("#F2F2F2")
        delta_str = row[-1]
        delta_val = float(delta_str.replace("%", "").replace("+", ""))
        delta_cell = tbl[i, len(col_labels) - 1]
        if delta_val >= 0:
            delta_cell.set_text_props(color="#2E7D32", fontweight="bold")
        else:
            delta_cell.set_text_props(color="#C62828", fontweight="bold")

    plt.title("Confronto Baseline vs PPO — Media ± Deviazione Standard",
              fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    out_img = os.path.join(OUTPUT_DIR, "final_summary_table.png")
    plt.savefig(out_img, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✅ Salvato: {out_img}")

    print(f"✅ Confronto completato per {output_base}/{output_subdir}")


# ════════════════════════════════════════════════════════════════
#  4. FUNZIONE PRINCIPALE PER UN MODELLO
# ════════════════════════════════════════════════════════════════
def process_model(model_name, ppo_csv, ppo_npz, monitor_dir, generate_training=True):
    """
    Genera tutti i grafici per un dato modello:
    - reward
    - training (opzionale)
    - confronti con baseline (standard e ottimizzata)
    """
    print(f"\n{'='*60}")
    print(f"🔄 Elaborazione modello: {model_name}")
    print(f"{'='*60}")

    # Reward curve
    generate_reward_curve(monitor_dir, model_name)

    # Training metrics
    if generate_training:
        generate_training_metrics(model_name)

    # Confronti
    # Baseline standard
    generate_comparison(
        baseline_csv="csv_generati/baseline/metrics_baseline.csv",
        baseline_npz="csv_generati/baseline/timeseries_baseline.npz",
        ppo_csv=ppo_csv,
        ppo_npz=ppo_npz,
        output_subdir="confronto_PPO_baseline",
        output_base=model_name
    )

    # Baseline ottimizzata
    generate_comparison(
        baseline_csv="csv_generati/baseline_opt/metrics_baseline.csv",
        baseline_npz="csv_generati/baseline_opt/timeseries_baseline.npz",
        ppo_csv=ppo_csv,
        ppo_npz=ppo_npz,
        output_subdir="confronto_PPO_baseline-opt",
        output_base=model_name
    )


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Modello 1: PPO_energy_pedwaiting (con training metrics)
    process_model(
        model_name="PPO_energy_pedwaiting",
        ppo_csv="csv_generati/PPO/metrics_ppo.csv",
        ppo_npz="csv_generati/PPO/timeseries_ppo.npz",
        monitor_dir="models/t20energy-pedwaiting_f2",
        generate_training=True
    )

    # Modello 2: PPO_energy_pedwaiting_collisions (senza training metrics)
    process_model(
        model_name="PPO_energy_pedwaiting_collisions",
        ppo_csv="csv_generati/PPO-collisions/metrics_ppo.csv",
        ppo_npz="csv_generati/PPO-collisions/timeseries_ppo.npz",
        monitor_dir="models/t20energy-pedwaiting-collisions_f2",
        generate_training=False
    )

    print("\n🎉 Tutti i grafici sono stati generati nella cartella:", BASE_OUT)
    print("   - PPO_energy_pedwaiting/")
    print("   - PPO_energy_pedwaiting_collisions/")
    plt.close('all')