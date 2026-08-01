"""Day 4 follow-up (not part of the main pipeline): does feeding
temperature into the autoencoder itself, as a genuine model input, help
detection or localisation, compared to the current approach (fit the
model first, then subtract a temperature-fitted trend from the features
afterward, see scripts/05_lumo_transfer.py's temperature_detrend)?

Idea #2 from the Eltouny/Gomaa/Liang 2023 SHM review (Ozdagli &
Koutsoukos 2019; Gu et al. 2017): give the model temperature as an extra
input, so it can learn "what does healthy look like at this temperature"
from the start, instead of correcting for it after the fact.

LUMO already records one temperature reading per 10-minute block
(src/data/lumo.py's block_mean_temperature). This script scales it (fit
on healthy TRAIN blocks only, same as the feature scaler) and broadcasts
the single value to all 18 sensor nodes as an extra feature column, so
every node's input includes "what temperature was it". Worth being
upfront about a simplification this makes: train_autoencoder's loss is
one MSE over the whole tensor, so the model is also scored on
reconstructing this extra column, not just conditioned on it. Since the
same value is copied to all 18 nodes, reconstructing it should be close
to trivial for every node's decoder, so it should not meaningfully
corrupt the anomaly signal built from the other features, but this is
not the same as a model strictly conditioned on temperature without
being scored on reproducing it.

Trains GCN and per-sensor, each with and without the temperature column,
back to back in the same run (same split, same seed) for a fair, paired
comparison, avoiding the run-to-run PyTorch CPU non-determinism already
documented in DAY_4.md's "An honest complication".

Run from the repo root:
    python scripts/day4_lumo_temperature_input_experiment.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from src.data.lumo import (
    ACCEL_CHANNELS,
    DAM_POSITION_MLS,
    ML_HEIGHTS_M,
    build_feature_table,
    list_blocks,
    lumo_sensor_graph,
    nearest_levels_for_dam,
)
from src.evaluate.metrics import detection_roc_auc, localization_rank, localization_topk_accuracy
from src.models.gcn_ae import GCNAutoencoder, per_node_error
from src.models.per_sensor_ae import PerSensorAutoencoder
from src.models.train_utils import train_autoencoder

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"

N_LEVELS = len(ML_HEIGHTS_M) - 1  # ML1-ML9 have accelerometers; ML10 does not


def load_config() -> dict:
    with open(REPO_ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def split_healthy_blocks(meta: pd.DataFrame, test_fraction: float, val_fraction: float, rng) -> tuple:
    healthy_blocks = np.sort(meta.loc[meta["is_healthy"], "block_id"].unique())
    n_total = len(healthy_blocks)
    n_test = max(1, round(n_total * test_fraction))
    n_val = max(1, round(n_total * val_fraction))
    shuffled = rng.permutation(healthy_blocks)
    test_blocks = set(shuffled[:n_test].tolist())
    val_blocks = set(shuffled[n_test:n_test + n_val].tolist())
    train_blocks = set(shuffled[n_test + n_val:].tolist())
    return train_blocks, val_blocks, test_blocks


def build_feature_tensor(table: pd.DataFrame, feature_cols: list, n_sensors: int):
    table_sorted = table.sort_values(["window_id", "sensor_idx"])
    window_ids = table_sorted["window_id"].unique()
    x = table_sorted[feature_cols].to_numpy().reshape(len(window_ids), n_sensors, len(feature_cols))
    return x, window_ids


def evaluate(model, prop, x_val, x_eval, is_damaged_eval, dam_position_eval, k=3):
    model.eval()
    with torch.no_grad():
        x_val_t = torch.tensor(x_val, dtype=torch.float32)
        x_eval_t = torch.tensor(x_eval, dtype=torch.float32)
        prop_t = torch.tensor(prop, dtype=torch.float32)
        val_node_err = per_node_error(x_val_t, model(x_val_t, prop_t)).numpy()
        node_err = per_node_error(x_eval_t, model(x_eval_t, prop_t)).numpy()

    scores = node_err.mean(axis=1)
    pooled_auc = detection_roc_auc(is_damaged_eval, scores)

    sensor_error_mean, sensor_error_std = val_node_err.mean(axis=0), val_node_err.std(axis=0)
    zscored_node_err = (node_err - sensor_error_mean) / sensor_error_std

    by_position = {}
    for pos in sorted(set(dam_position_eval[is_damaged_eval])):
        pos_mask = dam_position_eval == pos
        level_err_raw = node_err[pos_mask].reshape(-1, N_LEVELS, 2).mean(axis=2)
        level_err_z = zscored_node_err[pos_mask].reshape(-1, N_LEVELS, 2).mean(axis=2)
        expected_levels = nearest_levels_for_dam(int(pos))

        best_raw_topk = max(localization_topk_accuracy(level_err_raw, e, k=k) for e in expected_levels)
        best_raw_rank = min(np.mean([localization_rank(r, e) for r in level_err_raw]) for e in expected_levels)
        best_z_topk = max(localization_topk_accuracy(level_err_z, e, k=k) for e in expected_levels)
        best_z_rank = min(np.mean([localization_rank(r, e) for r in level_err_z]) for e in expected_levels)

        healthy_eval_mask = ~is_damaged_eval
        mask = healthy_eval_mask | pos_mask
        pos_auc = detection_roc_auc(pos_mask[mask].astype(int), scores[mask])

        by_position[int(pos)] = {
            "auc": pos_auc, "raw_topk": best_raw_topk, "raw_rank": best_raw_rank,
            "z_topk": best_z_topk, "z_rank": best_z_rank,
        }

    return pooled_auc, by_position


def main() -> None:
    config = load_config()
    model_cfg = config["model"]
    lumo_cfg = config["lumo"]
    seed = model_cfg["random_seed"]
    rng = np.random.default_rng(seed)

    data_dirs = [REPO_ROOT / d for d in lumo_cfg["data_dirs"]]
    missing = [d for d in data_dirs if not d.exists()]
    if missing:
        raise FileNotFoundError(f"LUMO data not found at {missing}. See DAY_4.md for download instructions.")

    blocks = pd.concat([list_blocks(d) for d in data_dirs], ignore_index=True)
    sampling_rate_hz = 1651.6129032258063
    table = build_feature_table(blocks, lumo_cfg["window_length"], sampling_rate_hz, config["features"]["band_edges_hz"])
    non_feature_cols = ("window_id", "sensor_idx", "block_id", "is_healthy", "dam_position", "temperature_c")
    feature_cols = [c for c in table.columns if c not in non_feature_cols]
    print(f"Feature table: {table.shape[0]} rows, {len(feature_cols)} features")

    train_blocks, val_blocks, test_blocks = split_healthy_blocks(
        table, lumo_cfg["healthy_test_fraction"], lumo_cfg["healthy_val_fraction"], rng,
    )
    train_mask = table["block_id"].isin(train_blocks).to_numpy()
    scaler = StandardScaler().fit(table.loc[train_mask, feature_cols])
    scaled_table = table.copy()
    scaled_table[feature_cols] = scaler.transform(table[feature_cols])
    x_scaled, window_ids = build_feature_tensor(scaled_table, feature_cols, len(ACCEL_CHANNELS))

    window_meta = table[["window_id", "block_id", "is_healthy", "dam_position", "temperature_c"]] \
        .drop_duplicates().set_index("window_id").loc[window_ids]
    train_window_mask = window_meta["block_id"].isin(train_blocks).to_numpy()
    val_window_mask = window_meta["block_id"].isin(val_blocks).to_numpy()
    eval_window_mask = (
        window_meta["block_id"].isin(test_blocks) | ~window_meta["is_healthy"].astype(bool)
    ).to_numpy()

    adjacency, a_hat = lumo_sensor_graph()
    n_sensors = len(ACCEL_CHANNELS)
    n_features = len(feature_cols)

    # Scale temperature the same way as the other features (fit on healthy
    # TRAIN windows only), then broadcast the one block-level reading to
    # all 18 sensor nodes as an extra feature column.
    temperature_all = window_meta["temperature_c"].to_numpy()
    temp_mean = temperature_all[train_window_mask].mean()
    temp_std = temperature_all[train_window_mask].std()
    temperature_scaled = (temperature_all - temp_mean) / temp_std
    temp_channel = np.repeat(temperature_scaled[:, None, None], n_sensors, axis=1)
    x_with_temp = np.concatenate([x_scaled, temp_channel], axis=-1)

    is_damaged_eval = ~window_meta.loc[eval_window_mask, "is_healthy"].astype(bool).to_numpy()
    dam_position_eval = window_meta.loc[eval_window_mask, "dam_position"].to_numpy()
    print(f"Training on {train_window_mask.sum()} healthy windows, evaluating on {eval_window_mask.sum()} "
          f"({is_damaged_eval.sum()} damaged)\n")

    input_variants = {
        "baseline (no temperature)": (x_scaled, n_features),
        "with temperature input": (x_with_temp, n_features + 1),
    }
    architecture_builders = {
        "GCN": lambda nf: GCNAutoencoder(nf, model_cfg["hidden_dim"], model_cfg["latent_dim"]),
        "Per-sensor (no graph)": lambda nf: PerSensorAutoencoder(
            n_sensors, nf, model_cfg["hidden_dim"], model_cfg["latent_dim"],
        ),
    }

    results = {}
    for arch_name, build_fn in architecture_builders.items():
        for variant_name, (x_full, nf) in input_variants.items():
            x_train = x_full[train_window_mask]
            x_val = x_full[val_window_mask]
            x_eval = x_full[eval_window_mask]

            print(f"Training {arch_name}, {variant_name}...")
            model, train_history, val_history, best_epoch = train_autoencoder(
                build_fn(nf), x_train, x_val, a_hat,
                learning_rate=model_cfg["learning_rate"], patience=model_cfg["early_stopping_patience"],
                min_delta=model_cfg["early_stopping_min_delta"], max_epochs=model_cfg["max_epochs"], seed=seed,
            )
            stopped_early = len(train_history) < model_cfg["max_epochs"]
            print(f"  {len(train_history)} epochs ({'stopped early' if stopped_early else 'hit max_epochs cap'})")
            pooled_auc, by_position = evaluate(model, a_hat, x_val, x_eval, is_damaged_eval, dam_position_eval,
                                                k=model_cfg["localization_top_k"])
            results[(arch_name, variant_name)] = {"pooled_auc": pooled_auc, "by_position": by_position}
            print(f"  Pooled detection AUC: {pooled_auc:.3f}")
            for pos, m in by_position.items():
                ml_pair = DAM_POSITION_MLS[pos]
                print(f"  DAM{pos} (between ML{ml_pair[0]} and ML{ml_pair[1]}): AUC {m['auc']:.3f}, "
                      f"raw top-3 {m['raw_topk']:.1%}/rank {m['raw_rank']:.1f}, "
                      f"z-score top-3 {m['z_topk']:.1%}/rank {m['z_rank']:.1f}")
            print()

    chance_topk = model_cfg["localization_top_k"] / N_LEVELS
    chance_rank = (N_LEVELS - 1) / 2
    print(f"Chance: AUC 0.500, top-{model_cfg['localization_top_k']} {chance_topk:.1%}, mean rank {chance_rank:.1f}\n")

    print("Summary, baseline vs with-temperature-input, averaged across the 3 DAM positions:")
    for (arch_name, variant_name), r in results.items():
        positions = r["by_position"].values()
        print(f"  {arch_name}, {variant_name}: AUC {r['pooled_auc']:.3f}, "
              f"z-score rank {np.mean([m['z_rank'] for m in positions]):.1f}, "
              f"z-score top-3 {np.mean([m['z_topk'] for m in positions]):.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    arch_names = list(architecture_builders.keys())
    x_pos = np.arange(len(arch_names))
    width = 0.35
    for ax, metric, ylabel, chance_line, agg in [
        (axes[0], "pooled_auc", "Pooled detection AUC", 0.5, None),
        (axes[1], "z_rank", "Mean rank, z-score, avg over DAM positions", chance_rank, "rank"),
    ]:
        for i, variant_name in enumerate(input_variants.keys()):
            if agg is None:
                values = [results[(a, variant_name)]["pooled_auc"] for a in arch_names]
            else:
                values = [np.mean([m["z_rank"] for m in results[(a, variant_name)]["by_position"].values()])
                          for a in arch_names]
            ax.bar(x_pos + (i - 0.5) * width, values, width, label=variant_name)
        ax.axhline(chance_line, color="black", linestyle="--", linewidth=1, label="Chance")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(arch_names)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("LUMO: temperature as a model input, baseline vs with-temperature")
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    out_path = FIGURES_DIR / "day4_lumo_temperature_input.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
