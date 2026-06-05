"""
Genera reglas con AnyBURL para cada fichero de data/train_splits/.

Por cada train_full_*.ttl:
  1. Convierte .ttl -> .tsv (AnyBURL no lee Turtle, necesita suj<TAB>pred<TAB>obj)
  2. Escribe un config-learn.properties con las propiedades pedidas
  3. Ejecuta:  java -Xmx12G -cp AnyBURL-23-1x.jar de.unima.ki.anyburl.Learn <config>
  4. Almacena las reglas en data/reglas/<nombre_del_split>/

Uso:
    python scripts/learn_rules_splits.py
"""

import subprocess
import sys
from pathlib import Path

# Reutiliza la conversion N3->TSV y la localizacion de java ya existentes
from learn_rules_anyburl import n3_to_tsv, download_jar, _find_java

BASE_DIR   = Path(__file__).parent.parent
SPLITS_DIR = BASE_DIR / "data" / "train_splits"
REGLAS_DIR = BASE_DIR / "data" / "reglas"
JAR_FILE   = BASE_DIR / "AnyBURL-23-1x.jar"

# --- Propiedades de AnyBURL pedidas ---
THRESHOLD_CORRECT_PREDICTIONS = 10
THRESHOLD_CONFIDENCE          = 0.7
SNAPSHOTS_AT                  = "100,500,1000"
WORKER_THREADS                = 7
XMX                           = "12G"

# Predicados que NO deben entrar en el TSV de aprendizaje de reglas.
# hasIntervention se excluye: las intervenciones se introducen a mano en el
# incident creator, no se predicen con reglas, y sólo añaden ruido al aprendizaje.
EXCLUDE_PREDS = {"hasDedicationTimeMin", "createdOn", "hasIntervention"}


def write_config(config_path: Path, tsv_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = f"""\
PATH_TRAINING = {tsv_path}
PATH_OUTPUT   = {out_dir}/rules

THRESHOLD_CORRECT_PREDICTIONS = {THRESHOLD_CORRECT_PREDICTIONS}
THRESHOLD_CONFIDENCE = {THRESHOLD_CONFIDENCE}

SNAPSHOTS_AT = {SNAPSHOTS_AT}

SAFE_PREFIX_MODE = false
WORKER_THREADS = {WORKER_THREADS}
"""
    config_path.write_text(config, encoding="utf-8")


def main():
    if not SPLITS_DIR.exists():
        print(f"[!] No existe {SPLITS_DIR}")
        sys.exit(1)

    # Si se pasan nombres de split por la línea de comandos, procesar solo esos.
    # Ej.:  python scripts/learn_rules_splits.py train_full_incidents train_full_incidents_interventions
    requested = [a.removesuffix(".ttl") for a in sys.argv[1:]]
    if requested:
        ttl_files = [SPLITS_DIR / f"{name}.ttl" for name in requested]
        missing = [t for t in ttl_files if not t.exists()]
        if missing:
            print("[!] No existen estos splits:")
            for t in missing:
                print(f"    {t}")
            sys.exit(1)
    else:
        ttl_files = sorted(SPLITS_DIR.glob("*.ttl"))
    if not ttl_files:
        print(f"[!] No hay .ttl en {SPLITS_DIR}")
        sys.exit(1)

    download_jar(JAR_FILE)
    java = _find_java()

    print(f"\n{'='*60}")
    print(f"  AnyBURL sobre {len(ttl_files)} split(s)")
    print(f"  Soporte>={THRESHOLD_CORRECT_PREDICTIONS}  Confianza>={THRESHOLD_CONFIDENCE}")
    print(f"  Snapshots={SNAPSHOTS_AT}  Threads={WORKER_THREADS}  -Xmx{XMX}")
    print(f"{'='*60}")

    for i, ttl in enumerate(ttl_files, 1):
        name = ttl.stem                       # p.ej. train_full_employees
        out_dir = REGLAS_DIR / name           # data/reglas/train_full_employees/
        tsv_path = out_dir / f"{name}.tsv"
        config_path = out_dir / "config-learn.properties"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i}/{len(ttl_files)}] === {ttl.name} -> {out_dir.relative_to(BASE_DIR)}/ ===")
        n3_to_tsv(ttl, tsv_path, exclude_preds=EXCLUDE_PREDS)
        write_config(config_path, tsv_path, out_dir)

        cmd = [
            java, f"-Xmx{XMX}", "-cp", str(JAR_FILE),
            "de.unima.ki.anyburl.Learn", str(config_path),
        ]
        try:
            subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            print("\n[!] Interrumpido por el usuario.")
            break

        finales = sorted(out_dir.glob("rules-*"))
        if finales:
            for r in finales:
                n = sum(1 for _ in open(r))
                print(f"  OK {r.relative_to(BASE_DIR)}: {n:,} reglas")
        else:
            print(f"  [!] No se generaron reglas en {out_dir}")

    print(f"\n{'='*60}")
    print(f"  Reglas en: data/reglas/<nombre_split>/rules-100, rules-500, rules-1000")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
