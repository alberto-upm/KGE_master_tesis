"""
Análisis previo del grafo de incidencias.

Recorre el TTL/N3 fuente y reporta:
  · Totales: tripletas, entidades únicas, relaciones únicas.
  · Conteo por tipo de entidad (incident, intervention, employee, company,
    supportGroup, supportTeam, supportCategory, status, type, origin, person,
    other).
  · Distribución de propiedades clave de las incidencias:
      hasStateIncident, hasTypeInc, incident_hasOrigin,
      hasSupportCategory, hasSupportGroup, hasSupportTeam,
      hasTechnician, hasExternalTechnician, int_hasCustomer.
  · Distribución de relaciones (todas) coloreadas en stacked-bar por el
    dominio del sujeto (incident vs. intervention vs. employee vs. other).

Salida:
  out/figures/analisis_previo/relation_distribution.png
  out/figures/analisis_previo/entity_type_distribution.png
  out/figures/analisis_previo/metagraph.png        ← schema visual
  out/figures/analisis_previo/conteos.txt

Uso:
  python scripts/analisis_previo.py
  python scripts/analisis_previo.py --input data/incident_triplets.ttl
  python scripts/analisis_previo.py --top-n 20

Notas:
  · El TTL es grande (≈7M tripletas). El parseo con rdflib tarda varios
    minutos; muestra avance.
  · Si modificas los prefijos de tipos en el grafo, ajusta _entity_type()
    o la lista PROP_PROPS.
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Permitir importar `config` desde src/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
import config as cfg


# ---------------------------------------------------------------------------
# Clasificación de entidades por prefijo del label
# ---------------------------------------------------------------------------

_TYPE_PREFIXES = [
    ("incident_",       "incident"),
    ("intervention_",   "intervention"),
    ("employee",        "employee"),
    ("company",         "company"),
    ("supportGroup",    "supportGroup"),
    ("supportTeam",     "supportTeam"),
    ("supportCategory", "supportCategory"),
    ("statusIncident",  "status"),
    ("typeIncident",    "type"),
    ("incidentOrigin",  "origin"),
    ("person_",         "person"),
    ("bu_",             "businessUnit"),
]


def _entity_type(label: str) -> str:
    for pref, etype in _TYPE_PREFIXES:
        if label.startswith(pref):
            return etype
    return "other"


_TYPE_COLORS = {
    "incident":        "#1f77b4",
    "intervention":    "#bcbd22",
    "employee":        "#2ca02c",
    "company":         "#ff7f0e",
    "supportGroup":    "#d62728",
    "supportTeam":     "#9467bd",
    "supportCategory": "#17becf",
    "status":          "#8c564b",
    "type":            "#e377c2",
    "origin":          "#7f7f7f",
    "person":          "#aec7e8",
    "businessUnit":    "#98df8a",
    "other":           "#cccccc",
}


# Propiedades cuyo valor mostramos en la tabla de distribuciones,
# agrupadas por el tipo del sujeto. Solo se cuentan las tripletas
# (s, p, o) donde el sujeto es de ese tipo.
PROPS_BY_SUBJECT_TYPE: dict[str, list[str]] = {
    "incident": [
        "hasStateIncident",
        "hasTypeInc",
        "incident_hasOrigin",
        "hasSupportCategory",
        "hasSupportGroup",
        "hasSupportTeam",
        "hasTechnician",
        "hasExternalTechnician",
        "int_hasCustomer",
        "hasIntervention",  # multi-valor (IDs de intervención)
    ],
    "intervention": [
        "hasSupportTeam",
        "hasTechnician",
    ],
    "employee": [
        "hasBusinessUnit",
        "hasCompany",
        "hasOrganizationalUnit",
    ],
}


# ---------------------------------------------------------------------------
# Parseo del grafo
# ---------------------------------------------------------------------------

def extract_label(uri) -> str:
    """Devuelve la parte local del URI (sin namespace)."""
    s = str(uri)
    if "#" in s:
        return s.split("#")[-1]
    return s.split("/")[-1]


def load_graph(path: Path):
    """Carga el TTL/N3 con rdflib."""
    from rdflib import Graph
    print(f"[1/3] Cargando grafo desde {path} ...")
    g = Graph()
    fmt = "n3" if path.suffix == ".n3" else "turtle"
    g.parse(str(path), format=fmt)
    print(f"      {len(g):,} tripletas cargadas.")
    return g


def _is_literal(o) -> bool:
    """True si el objeto rdflib es un Literal (dato escalar)."""
    try:
        from rdflib.term import Literal
        return isinstance(o, Literal)
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Contadores
# ---------------------------------------------------------------------------

def collect_counts(g):
    """
    Recorre el grafo y devuelve un dict con todos los conteos.
    """
    print("[2/3] Contando entidades, relaciones y distribuciones ...")

    entities      = set()
    relations     = set()
    n_triples     = 0

    # Tipo declarado de cada sujeto (vía repcon:type / rdf:type)
    subject_type: dict[str, str] = {}

    # Conteo de tripletas por relación
    rel_count: Counter = Counter()

    # Para stacked bar: relation -> dom -> count
    rel_by_domain: defaultdict[str, Counter] = defaultdict(Counter)

    # Distribución de valores por (tipo_sujeto, propiedad)
    #   prop_value_count[stype][prop] = Counter(valores)
    prop_value_count: dict[str, dict[str, Counter]] = {
        stype: {p: Counter() for p in props}
        for stype, props in PROPS_BY_SUBJECT_TYPE.items()
    }

    # Aristas del meta-grafo: (tipo_sujeto, relación, tipo_objeto) → count
    meta_edges: Counter = Counter()

    # ---------- Pase 1: tipos declarados ----------
    for s, p, o in g:
        pred = extract_label(p)
        if pred == "type":
            subject_type[extract_label(s)] = extract_label(o)

    # ---------- Pase 2: el resto ----------
    for s, p, o in g:
        n_triples += 1
        s_lbl = extract_label(s)
        p_lbl = extract_label(p)
        o_lbl = extract_label(o)

        entities.add(s_lbl)
        if not _is_literal(o):
            entities.add(o_lbl)
        relations.add(p_lbl)
        rel_count[p_lbl] += 1

        # Dominio del sujeto: prioriza tipo declarado; si no, por prefijo
        s_type = subject_type.get(s_lbl) or _entity_type(s_lbl)
        # Normalizar al esquema de _TYPE_COLORS
        if s_type not in _TYPE_COLORS:
            s_type = _entity_type(s_lbl)
        rel_by_domain[p_lbl][s_type] += 1

        # Distribución de propiedades, según el tipo del sujeto
        bucket = prop_value_count.get(s_type)
        if bucket is not None and p_lbl in bucket:
            bucket[p_lbl][o_lbl] += 1

        # Aristas del meta-grafo (sólo si el objeto es URI, no literal)
        if not _is_literal(o):
            o_type = subject_type.get(o_lbl) or _entity_type(o_lbl)
            if o_type not in _TYPE_COLORS:
                o_type = _entity_type(o_lbl)
            meta_edges[(s_type, p_lbl, o_type)] += 1

    # Conteo de entidades por tipo
    entity_type_count: Counter = Counter()
    for ent in entities:
        etype = subject_type.get(ent) or _entity_type(ent)
        if etype not in _TYPE_COLORS:
            etype = _entity_type(ent)
        entity_type_count[etype] += 1

    return {
        "n_triples":         n_triples,
        "n_entities":        len(entities),
        "n_relations":       len(relations),
        "rel_count":         rel_count,
        "rel_by_domain":     rel_by_domain,
        "entity_type_count": entity_type_count,
        "prop_value_count":  prop_value_count,
        "meta_edges":        meta_edges,
    }


# ---------------------------------------------------------------------------
# Salida por consola + fichero
# ---------------------------------------------------------------------------

def format_report(counts: dict, top_n: int = 10) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("  ANÁLISIS PREVIO DEL GRAFO DE INCIDENCIAS")
    lines.append("=" * 70)
    lines.append(f"")
    lines.append(f"Totales:")
    lines.append(f"  Tripletas:        {counts['n_triples']:>10,}")
    lines.append(f"  Entidades únicas: {counts['n_entities']:>10,}")
    lines.append(f"  Relaciones únicas:{counts['n_relations']:>10,}")
    lines.append("")

    # Entidades por tipo
    lines.append("Entidades por tipo:")
    for etype, n in counts["entity_type_count"].most_common():
        lines.append(f"  {etype:<16} {n:>10,}")
    lines.append("")

    # Relaciones por frecuencia
    lines.append("Relaciones (frecuencia de tripletas):")
    for rel, n in counts["rel_count"].most_common():
        lines.append(f"  {rel:<28} {n:>10,}")
    lines.append("")

    # Distribución de propiedades clave (top_n valores) agrupadas por tipo
    for stype, props in PROPS_BY_SUBJECT_TYPE.items():
        lines.append(f"Distribución de propiedades de '{stype}' (top-{top_n}):")
        bucket = counts["prop_value_count"].get(stype, {})
        for prop in props:
            ctr = bucket.get(prop)
            if not ctr:
                lines.append(f"\n  [{prop}]  (sin datos)")
                continue
            total = sum(ctr.values())
            lines.append(f"\n  [{prop}]  total tripletas: {total:,}  "
                         f"|  valores únicos: {len(ctr):,}")
            for val, n in ctr.most_common(top_n):
                lines.append(f"    {val:<48} {n:>8,}")
        lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def plot_relation_distribution(counts: dict, out_path: Path) -> None:
    """Stacked bar: cada relación, dividida por dominio del sujeto."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[!] matplotlib no instalado; se omite la figura.")
        return

    relations = [r for r, _ in counts["rel_count"].most_common()]
    # Dominios presentes ordenados por frecuencia total
    dom_total: Counter = Counter()
    for r in relations:
        for d, n in counts["rel_by_domain"][r].items():
            dom_total[d] += n
    domains = [d for d, _ in dom_total.most_common()]

    x = np.arange(len(relations))
    bottom = np.zeros(len(relations))

    plt.figure(figsize=(max(10, len(relations) * 0.7), 6))
    for dom in domains:
        vals = np.array(
            [counts["rel_by_domain"][r].get(dom, 0) for r in relations],
            dtype=float,
        )
        plt.bar(
            x, vals, bottom=bottom,
            color=_TYPE_COLORS.get(dom, "#cccccc"),
            label=f"{dom} ({int(vals.sum()):,})",
            edgecolor="white", linewidth=0.3,
        )
        bottom += vals

    plt.xticks(x, relations, rotation=45, ha="right", fontsize=9)
    plt.ylabel("Nº tripletas")
    plt.title("Distribución de relaciones por dominio del sujeto")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"      Relation distribution → {out_path}")


def plot_metagraph(counts: dict, out_path: Path) -> None:
    """
    Meta-grafo del schema: nodos = tipos de entidad, aristas = relaciones
    agregadas entre tipos (con peso = nº de tripletas). Permite ver la
    estructura del grafo sin plotear las 972k entidades reales.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np
    except ImportError as e:
        print(f"      [!] Falta dependencia para meta-grafo ({e}); se omite.")
        return

    # Agrupar todas las relaciones entre el mismo par (s_type, o_type)
    # en una única arista con etiqueta concatenada.
    grouped: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for (s_t, rel, o_t), n in counts["meta_edges"].items():
        if rel == "type":
            continue  # arista estructural; no aporta información
        grouped.setdefault((s_t, o_t), []).append((rel, n))

    G = nx.DiGraph()
    for s_t in counts["entity_type_count"]:
        G.add_node(s_t, n_entities=counts["entity_type_count"][s_t])

    edge_labels = {}
    edge_weights = {}
    for (s_t, o_t), rels in grouped.items():
        rels_sorted = sorted(rels, key=lambda x: -x[1])
        total = sum(n for _, n in rels_sorted)
        label = "\n".join(f"{r}" for r, _ in rels_sorted)
        G.add_edge(s_t, o_t, label=label, weight=total)
        edge_labels[(s_t, o_t)] = label
        edge_weights[(s_t, o_t)] = total

    # Layout (spring con semilla fija para reproducibilidad)
    pos = nx.spring_layout(G, seed=42, k=2.5, iterations=80)

    # Tamaños de nodo proporcionales (log) al nº de entidades
    node_sizes = []
    node_colors = []
    for node in G.nodes():
        n_ent = counts["entity_type_count"].get(node, 1)
        node_sizes.append(300 + 600 * np.log10(max(n_ent, 1) + 1))
        node_colors.append(_TYPE_COLORS.get(node, "#cccccc"))

    # Anchura de arista proporcional (log) al nº de tripletas
    max_w = max(edge_weights.values()) if edge_weights else 1
    edge_widths = [
        1 + 5 * np.log10(edge_weights[e] + 1) / np.log10(max_w + 1)
        for e in G.edges()
    ]

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.9, edgecolors="black", linewidths=1)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    nx.draw_networkx_edges(G, pos,
                           width=edge_widths,
                           edge_color="#666666",
                           arrows=True, arrowsize=18,
                           connectionstyle="arc3,rad=0.12",
                           alpha=0.6)

    # Etiquetas de aristas (relaciones agregadas + total entre paréntesis)
    edge_label_full = {
        e: f"{edge_labels[e]}\n({edge_weights[e]:,})"
        for e in G.edges()
    }
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_label_full,
        font_size=7, bbox=dict(boxstyle="round", facecolor="white",
                                edgecolor="none", alpha=0.85),
    )

    plt.title("Meta-grafo del schema  (nodos = tipos · aristas = relaciones agregadas)",
              fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"      Meta-graph → {out_path}")


def plot_entity_type_distribution(counts: dict, out_path: Path) -> None:
    """Bar chart de nº entidades por tipo."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    items = counts["entity_type_count"].most_common()
    if not items:
        return
    types  = [t for t, _ in items]
    values = [n for _, n in items]
    colors = [_TYPE_COLORS.get(t, "#cccccc") for t in types]

    plt.figure(figsize=(10, 5))
    plt.bar(types, values, color=colors, edgecolor="white", linewidth=0.5)
    plt.yscale("log")  # log para ver tipos minoritarios
    plt.ylabel("Nº entidades (escala log)")
    plt.title("Entidades únicas por tipo")
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"      Entity type distribution → {out_path}")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run(input_path: Path, out_dir: Path, top_n: int = 10) -> None:
    g       = load_graph(input_path)
    counts  = collect_counts(g)

    print("[3/3] Generando informe y figuras ...")
    report = format_report(counts, top_n=top_n)
    print("\n" + report)

    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / "conteos.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"      Informe → {txt_path}")

    plot_relation_distribution(counts, out_dir / "relation_distribution.png")
    plot_entity_type_distribution(counts, out_dir / "entity_type_distribution.png")
    plot_metagraph(counts, out_dir / "metagraph.png")

    print("\n✓ Análisis previo completado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Análisis previo del grafo de incidencias."
    )
    parser.add_argument(
        "--input", type=Path,
        default=cfg.DATA_DIR / "incident_triplets.ttl",
        help="Fichero TTL/N3 a analizar (default: data/incident_triplets.ttl)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=cfg.OUT_DIR / "figures" / "analisis_previo",
        help="Directorio de salida para figuras y conteos.txt",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Top-N valores a mostrar por propiedad (default: 10)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"No se encontró el fichero: {args.input}\n"
            "Ejecuta primero phase 0 para generar incident_triplets.ttl"
        )

    run(args.input, args.output_dir, top_n=args.top_n)
