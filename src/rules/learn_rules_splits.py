"""
Genera reglas con AnyBURL para cada fichero de data/train_splits/.

Por cada train_full_*.ttl:
  1. Convierte .ttl -> .tsv (AnyBURL no lee Turtle, necesita suj<TAB>pred<TAB>obj)
  2. Escribe un config-learn.properties con las propiedades pedidas
  3. Ejecuta:  java -Xmx12G -cp AnyBURL-23-1x.jar de.unima.ki.anyburl.Learn <config>
  4. Almacena las reglas en data/reglas/<nombre_del_split>/

Uso:
    python src/run_pipeline.py --phase 4
    python src/rules/learn_rules_splits.py                       # todos los splits
    python src/rules/learn_rules_splits.py train_full_incidents  # solo algunos
"""

import subprocess
import sys
import urllib.request
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parents[2]   # src/rules/ → src/ → raíz del repo
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

# --- AnyBURL: conversión N3→TSV, descarga del JAR y localización de Java ---
# (antes vivían en scripts/learn_rules_anyburl.py)
PREFIX      = "repcon:"
ANYBURL_URL = "https://web.informatik.uni-mannheim.de/AnyBURL/AnyBURL-23-1x.jar"


def n3_to_tsv(n3_path: Path, out_path: Path, exclude_preds: set[str] | None = None) -> int:
    """
    Convierte un fichero N3 con prefijo 'repcon:' al formato plano de AnyBURL:
        sujeto<TAB>predicado<TAB>objeto

    Maneja la sintaxis Turtle compacta:
      repcon:X repcon:p1 repcon:v1 ;
               repcon:p2 repcon:v2 .

    exclude_preds: conjunto de predicados (nombre local, sin prefijo) a omitir.
                   Las tripletas con esos predicados no se escriben en el TSV.
    """
    print(f"[1/4] Convirtiendo {n3_path.name} → {out_path.name} ...")

    exclude = exclude_preds or set()

    def strip_prefix(token: str) -> str:
        token = token.rstrip(" ;.,")
        if token.startswith(PREFIX):
            return token[len(PREFIX):]
        return token

    n_triples = 0
    n_excluded = 0
    current_subject = None

    with open(n3_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        for raw_line in fin:
            line = raw_line.strip()

            # Saltar líneas vacías, comentarios y @prefix
            if not line or line.startswith("#") or line.startswith("@prefix"):
                continue

            # Detectar si la línea empieza con un sujeto (no con espacio)
            if not raw_line[0].isspace():
                parts = line.split(None, 2)
                if len(parts) < 3:
                    continue
                current_subject = strip_prefix(parts[0])
                pred  = strip_prefix(parts[1])
                obj   = strip_prefix(parts[2])
                if current_subject and pred and obj:
                    if pred in exclude:
                        n_excluded += 1
                    else:
                        fout.write(f"{current_subject}\t{pred}\t{obj}\n")
                        n_triples += 1
            else:
                # Continuación con el mismo sujeto (línea indentada)
                if current_subject is None:
                    continue
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                pred = strip_prefix(parts[0])
                obj  = strip_prefix(parts[1])
                if pred and obj:
                    if pred in exclude:
                        n_excluded += 1
                    else:
                        fout.write(f"{current_subject}\t{pred}\t{obj}\n")
                        n_triples += 1

    print(f"        {n_triples:,} triples escritos.")
    if exclude:
        print(f"        {n_excluded:,} triples excluidos (predicados: {', '.join(sorted(exclude))}).")
    return n_triples


def download_jar(jar_path: Path) -> None:
    if jar_path.exists():
        print(f"[2/4] Jar ya existe: {jar_path.name}")
        return
    print(f"[2/4] Descargando AnyBURL desde {ANYBURL_URL} ...")
    try:
        urllib.request.urlretrieve(ANYBURL_URL, jar_path)
        print(f"        Descargado ({jar_path.stat().st_size / 1e6:.1f} MB).")
    except Exception as e:
        print(f"[!] Error descargando AnyBURL: {e}")
        print(f"    Descárgalo manualmente de https://web.informatik.uni-mannheim.de/AnyBURL/")
        print(f"    y colócalo en: {jar_path}")
        sys.exit(1)


def _find_java() -> str:
    """Devuelve la ruta a java, instalándolo con install-jdk si no está disponible."""
    import shutil

    java = shutil.which("java")
    if java:
        return java

    print("  [!] 'java' no encontrado. Instalando OpenJDK 17 via pip (install-jdk) ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-user", "-q", "install-jdk"],
        check=True,
    )
    import jdk as _jdk
    print("      Descargando JDK 17 (puede tardar un momento) ...")
    jdk_dir = _jdk.install("17")
    java_bin = Path(jdk_dir) / "bin" / "java"
    if not java_bin.exists():
        print(f"  [!] Binario no encontrado en {java_bin}")
        sys.exit(1)
    print(f"        Java listo: {java_bin}")
    return str(java_bin)


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


def main(requested: list[str] | None = None):
    """
    Aprende reglas para los splits indicados.

    requested: lista de nombres de split (con o sin '.ttl'). Si es None se leen
    de sys.argv (modo CLI). Pasa [] para procesar TODOS los splits sin depender
    de argv — así lo invoca la Fase 4 (phase4_learn_rules.run()).
    """
    if not SPLITS_DIR.exists():
        print(f"[!] No existe {SPLITS_DIR}")
        sys.exit(1)

    # Nombres de split: por parámetro (Fase 4) o por línea de comandos (CLI).
    # Ej. CLI:  python src/rules/learn_rules_splits.py train_full_incidents ...
    if requested is None:
        requested = sys.argv[1:]
    requested = [a.removesuffix(".ttl") for a in requested]
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
