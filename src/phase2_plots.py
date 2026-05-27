"""
Fase 2 (plots) — Funciones de diagnóstico para modelos KGE y CLI para
regenerar los plots de un modelo ya entrenado, sin reentrenar.

Contenido del módulo:

  Utilidades de tipos
    _entity_type(label)            → clasifica por prefijo
    _TYPE_COLORS                   → paleta consistente con todo el proyecto

  Funciones de plot (reutilizables desde otros módulos)
    _plot_loss_curve(losses, ...)            curva de pérdida del entrenamiento
    _plot_tsne_embeddings_random(...)        t-SNE con muestreo aleatorio uniforme
                                             (la "antigua": refleja la distribución
                                              real del grafo)
    _plot_tsne_embeddings(...)               t-SNE con muestreo ESTRATIFICADO por
                                             tipo (la "actual": balanceada, ideal
                                              para ver si el KGE separa tipos)

  CLI standalone
    python src/phase2_plots.py --kge-model TransE
    python src/phase2_plots.py --kge-model RotatE --n-per-type 5000
    python src/phase2_plots.py --kge-model TransE --skip-tsne
    python src/phase2_plots.py --kge-model TransE --skip-loss

Lee de disco lo que phase2_kge_train.py dejó:
  out/models/<modelo>/results.json                (losses por época)
  out/models/<modelo>/training_triples/           (factory PyKEEN binario)
  out/embeddings/<modelo>/entity_embeddings.pt    (embeddings)

Genera con timestamp nuevo:
  out/figures/<modelo>/loss_curve_<modelo>_<ts>.png
  out/figures/<modelo>/tsne_entities_<modelo>_<ts>.png
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


# ---------------------------------------------------------------------------
# Clasificación de entidades por prefijo del label
# ---------------------------------------------------------------------------

def _entity_type(label: str) -> str:
    """Clasifica una entidad por prefijo de su label."""
    prefixes = {
        "incident_":       "incident",
        "intervention_":   "intervention",
        "company":         "company",
        "employee":        "employee",
        "supportGroup":    "supportGroup",
        "supportTeam":     "supportTeam",
        "supportCategory": "supportCategory",
        "statusIncident":  "status",
        "typeIncident":    "type",
        "incidentOrigin":  "origin",
        "person_":         "person",
    }
    for pref, etype in prefixes.items():
        if label.startswith(pref):
            return etype
    return "other"


_TYPE_COLORS = {
    "incident":        "#1f77b4",
    "intervention":    "#bcbd22",
    "company":         "#ff7f0e",
    "employee":        "#2ca02c",
    "supportGroup":    "#d62728",
    "supportTeam":     "#9467bd",
    "supportCategory": "#17becf",
    "status":          "#8c564b",
    "type":            "#e377c2",
    "origin":          "#7f7f7f",
    "person":          "#aec7e8",
    "other":           "#cccccc",
}


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------

def _plot_loss_curve(
    losses,
    model_name: str,
    out_dir: Path,
    timestamp: str,
) -> None:
    """
    Guarda la curva de loss en
      out_dir/loss_curve_<modelo>_<timestamp>.png

    `losses` admite:
      - Lista de floats (caso CLI: leyendo results.json).
      - Objeto PipelineResult de PyKEEN (con atributo .losses).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # backend sin display (servidor)
        import matplotlib.pyplot as plt
    except ImportError:
        print("      [!] matplotlib no instalado; se omite plot de loss.")
        return

    # Permitir tanto lista directa como objeto PipelineResult
    if hasattr(losses, "losses"):
        losses = losses.losses
    losses = list(losses or [])
    if not losses:
        print("      [!] No hay losses registradas.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"loss_curve_{model_name.lower()}_{timestamp}.png"

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o", color="steelblue")
    plt.title(f"Curva de pérdida — {model_name}")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"      Loss curve → {out_path}")


# ---------------------------------------------------------------------------
# t-SNE — muestreo aleatorio uniforme (versión antigua)
# ---------------------------------------------------------------------------

def _plot_tsne_embeddings_random(
    entity_embs,
    factory,
    model_name: str,
    out_dir: Path,
    timestamp: str,
    n_sample: int = 2000,
    seed: int = 42,
) -> None:
    """
    t-SNE 2D con muestreo aleatorio uniforme sobre TODAS las entidades.

    Refleja la distribución REAL del grafo: si el 99% son
    incidents+interventions, en el plot ves prácticamente solo eso.
    Útil cuando se quiere mostrar el desequilibrio del grafo, NO cuando se
    quiere comprobar si el modelo separa los tipos minoritarios.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import numpy as np
    except ImportError as e:
        print(f"      [!] Falta dependencia para t-SNE ({e}); se omite.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tsne_entities_{model_name.lower()}_{timestamp}.png"

    embs_np = entity_embs.detach().cpu().numpy() if hasattr(entity_embs, "detach") \
        else np.asarray(entity_embs)
    if np.iscomplexobj(embs_np):
        embs_np = embs_np.real

    n_total = embs_np.shape[0]
    n = min(n_sample, n_total)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, n, replace=False)
    sample = embs_np[idx]

    id_to_label = {v: k for k, v in factory.entity_to_id.items()}
    labels = [id_to_label.get(int(i), "other") for i in idx]
    types  = [_entity_type(l) for l in labels]

    print(f"      Calculando t-SNE (muestreo aleatorio) sobre {n:,} entidades ...")
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30,
                init="pca", learning_rate="auto")
    embs_2d = tsne.fit_transform(sample)

    plt.figure(figsize=(12, 8))
    for etype, color in _TYPE_COLORS.items():
        mask = [i for i, t in enumerate(types) if t == etype]
        if mask:
            plt.scatter(
                embs_2d[mask, 0], embs_2d[mask, 1],
                c=color, label=f"{etype} ({len(mask)})",
                alpha=0.6, s=15,
            )
    plt.title(f"t-SNE de embeddings — {model_name}  (n={n:,}, aleatorio)",
              fontsize=13)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"      t-SNE      → {out_path}")


# ---------------------------------------------------------------------------
# t-SNE — muestreo estratificado por tipo (versión actual)
# ---------------------------------------------------------------------------

def _plot_tsne_embeddings(
    entity_embs,
    factory,
    model_name: str,
    out_dir: Path,
    timestamp: str,
    n_per_type: int = 500,
    seed: int = 42,
) -> None:
    """
    t-SNE 2D con muestreo ESTRATIFICADO: hasta `n_per_type` entidades de
    cada tipo (incident, company, employee, supportGroup, ...).

    Garantiza que todos los tipos aparezcan en el plot aunque sean
    minoritarios. Es la vista recomendada para evaluar si el KGE separa
    los tipos en su espacio de embeddings.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import numpy as np
    except ImportError as e:
        print(f"      [!] Falta dependencia para t-SNE ({e}); se omite.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tsne_entities_{model_name.lower()}_{timestamp}.png"

    embs_np = entity_embs.detach().cpu().numpy() if hasattr(entity_embs, "detach") \
        else np.asarray(entity_embs)
    if np.iscomplexobj(embs_np):
        embs_np = embs_np.real

    # Muestreo estratificado por tipo
    rng = np.random.default_rng(seed)
    id_to_label = {v: k for k, v in factory.entity_to_id.items()}

    ids_by_type: dict[str, list[int]] = {}
    for eid, label in id_to_label.items():
        ids_by_type.setdefault(_entity_type(label), []).append(eid)

    selected_ids: list[int] = []
    print(f"      Muestreo estratificado por tipo (hasta {n_per_type:,} por tipo):")
    for etype in sorted(ids_by_type):
        all_ids = ids_by_type[etype]
        n_total_type = len(all_ids)
        if n_total_type <= n_per_type:
            picks_ids = all_ids
        else:
            picks_idx = rng.choice(n_total_type, n_per_type, replace=False)
            picks_ids = [all_ids[i] for i in picks_idx]
        selected_ids.extend(picks_ids)
        print(f"        {etype:<16} total={n_total_type:>7,}  → muestreado={len(picks_ids):,}")

    idx = np.array(selected_ids, dtype=np.int64)
    sample = embs_np[idx]
    labels = [id_to_label[int(i)] for i in idx]
    types  = [_entity_type(l) for l in labels]
    n = len(idx)

    print(f"      Calculando t-SNE sobre {n:,} entidades ...")
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30,
                init="pca", learning_rate="auto")
    embs_2d = tsne.fit_transform(sample)

    plt.figure(figsize=(12, 8))
    for etype, color in _TYPE_COLORS.items():
        mask = [i for i, t in enumerate(types) if t == etype]
        if mask:
            plt.scatter(
                embs_2d[mask, 0], embs_2d[mask, 1],
                c=color, label=f"{etype} ({len(mask)})",
                alpha=0.6, s=15,
            )
    plt.title(f"t-SNE de embeddings — {model_name}  "
              f"(n={n:,}, estratificado)", fontsize=13)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"      t-SNE      → {out_path}")


# ---------------------------------------------------------------------------
# CLI: regenerar plots de un modelo ya entrenado
# ---------------------------------------------------------------------------

def run(
    kge_model_name: str = "TransE",
    n_per_type:     int | None = None,
    do_loss:        bool = True,
    do_tsne:        bool = True,
) -> None:
    model_lower    = kge_model_name.lower()
    out_model_dir  = cfg.model_dir(kge_model_name)
    out_embed_path = cfg.entity_embeddings_path(kge_model_name)
    results_json   = out_model_dir / "results.json"
    factory_dir    = out_model_dir / "training_triples"

    out_figures_dir = cfg.OUT_DIR / "figures" / model_lower
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print(f"Phase 2 (plots) — {kge_model_name}")
    print(f"  Lee de: {out_model_dir}")
    print(f"  Sale a: {out_figures_dir}")
    print("=" * 60)

    # ---------- Loss curve ----------
    if do_loss:
        if not results_json.exists():
            print(f"  [!] No encontrado: {results_json}\n"
                  "      ¿Entrenaste este modelo con phase2_kge_train.py?")
        else:
            with open(results_json, encoding="utf-8") as f:
                data = json.load(f)
            losses = data.get("losses", [])
            if not losses:
                print("  [!] results.json no contiene 'losses'.")
            else:
                print(f"  [1/2] Generando loss curve ({len(losses)} épocas) ...")
                _plot_loss_curve(losses, kge_model_name, out_figures_dir, ts)
    else:
        print("  [1/2] Loss curve omitido (--skip-loss).")

    # ---------- t-SNE (estratificado) ----------
    if do_tsne:
        if not out_embed_path.exists():
            print(f"  [!] No encontrado: {out_embed_path}")
            return
        if not factory_dir.exists():
            print(f"  [!] No encontrado: {factory_dir}\n"
                  "      Se necesita el factory binario para mapear ids→labels.")
            return

        from pykeen.triples import TriplesFactory

        print(f"  [2/2] Cargando embeddings y factory ...")
        entity_embs = torch.load(out_embed_path, map_location="cpu",
                                 weights_only=True)
        factory = TriplesFactory.from_path_binary(factory_dir)
        print(f"        entity_embs shape: {list(entity_embs.shape)}")
        print(f"        Entidades en factory: {len(factory.entity_to_id):,}")

        _plot_tsne_embeddings(
            entity_embs, factory, kge_model_name, out_figures_dir, ts,
            n_per_type=n_per_type or cfg.TSNE_N_PER_TYPE,
        )
    else:
        print("  [2/2] t-SNE omitido (--skip-tsne).")

    print("\n✓ Plots regenerados.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenera plots (loss curve + t-SNE) de un modelo KGE ya entrenado."
    )
    parser.add_argument("--kge-model", default="TransE",
                        help=f"Modelo KGE (default: TransE). Opciones: {cfg.KGE_MODELS}")
    parser.add_argument("--n-per-type", type=int, default=None,
                        help=f"Muestras por tipo en t-SNE (default: cfg.TSNE_N_PER_TYPE={cfg.TSNE_N_PER_TYPE})")
    parser.add_argument("--skip-loss", action="store_true",
                        help="No regenerar loss curve")
    parser.add_argument("--skip-tsne", action="store_true",
                        help="No regenerar t-SNE")
    args = parser.parse_args()

    run(
        kge_model_name=args.kge_model,
        n_per_type=args.n_per_type,
        do_loss=not args.skip_loss,
        do_tsne=not args.skip_tsne,
    )
