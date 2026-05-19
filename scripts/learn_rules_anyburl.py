"""
Aprende reglas sobre incident_triplets.ttl usando AnyBURL.

Criterios duros:
  - THRESHOLD_CORRECT_PREDICTIONS = 10   (mínimo 10 instancias)
  - THRESHOLD_CONFIDENCE = 0.75          (confianza >= 75%)

Pasos:
  1. Convierte incident_triplets.ttl → triples.tsv (formato AnyBURL: suj pred obj)
  2. Descarga AnyBURL-23-1x.jar si no existe
  3. Escribe config-learn.properties con los umbrales pedidos
  4. Ejecuta AnyBURL y guarda reglas en data/reglas/
"""

import re
import subprocess
import sys
import urllib.request
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
RULES_DIR  = DATA_DIR / "reglas"
SCRIPTS_DIR = BASE_DIR / "scripts"

INPUT_FILE    = DATA_DIR / "incident_triplets.ttl"
TSV_FILE   = DATA_DIR / "anyburl_triples.tsv"
JAR_FILE   = BASE_DIR / "AnyBURL-23-1x.jar"
CONFIG_FILE = SCRIPTS_DIR / "config-learn.properties"

ANYBURL_URL = "https://web.informatik.uni-mannheim.de/AnyBURL/AnyBURL-23-1x.jar"

PREFIX = "repcon:"

# Tiempo de aprendizaje (segundos). Genera reglas en snapshots: 10s, 50s, 100s, …
LEARNING_TIME = 1000
# Prefijo que SAFE_PREFIX_MODE añade internamente a identificadores que empiecen por dígito
# (lo gestiona AnyBURL, no necesitamos hacerlo manualmente)


# ---------------------------------------------------------------------------
# Paso 1: N3 → TSV
# ---------------------------------------------------------------------------

def n3_to_tsv(n3_path: Path, out_path: Path) -> int:
    """
    Convierte un fichero N3 con prefijo 'repcon:' al formato plano de AnyBURL:
        sujeto<TAB>predicado<TAB>objeto

    Maneja la sintaxis Turtle compacta:
      repcon:X repcon:p1 repcon:v1 ;
               repcon:p2 repcon:v2 .
    """
    print(f"[1/4] Convirtiendo {n3_path.name} → {out_path.name} ...")

    def strip_prefix(token: str) -> str:
        token = token.rstrip(" ;.,")
        if token.startswith(PREFIX):
            return token[len(PREFIX):]
        return token

    n_triples = 0
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
                    fout.write(f"{current_subject}\t{pred}\t{obj}\n")
                    n_triples += 1

    print(f"        {n_triples:,} triples escritos.")
    return n_triples


# ---------------------------------------------------------------------------
# Paso 2: Descargar AnyBURL jar
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Paso 3: Escribir config-learn.properties
# ---------------------------------------------------------------------------

def write_config(config_path: Path, tsv_path: Path, rules_dir: Path) -> None:
    print(f"[3/4] Escribiendo configuración AnyBURL ...")
    rules_dir.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config = f"""\
# AnyBURL — configuración de aprendizaje de reglas
# Generado por learn_rules_anyburl.py

# --- Entrada / Salida ---
PATH_TRAINING   = {tsv_path}
PATH_OUTPUT     = {rules_dir}/rules

# --- Umbrales (criterios duros) ---
# Mínimo 10 instancias correctas (soporte mínimo)
THRESHOLD_CORRECT_PREDICTIONS = 10
# Confianza mínima 0.75
THRESHOLD_CONFIDENCE = 0.75

# --- Solo generar el snapshot final ---
# AnyBURL escribirá únicamente rules-1000 (sin intermedios rules-10, rules-100, ...)
SNAPSHOTS_AT = {LEARNING_TIME}

# --- Longitud máxima de reglas ---
MAX_LENGTH_CYCLIC          = 3
MAX_LENGTH_ACYCLIC         = 1
MAX_LENGTH_GROUNDED_CYCLIC = 1

# --- Modo seguro para identificadores que empiezan por dígito ---
# Necesario porque el N3 tiene entidades como repcon:999
SAFE_PREFIX_MODE = true

# --- Paralelismo ---
WORKER_THREADS = 4
"""
    config_path.write_text(config, encoding="utf-8")
    print(f"        Config guardada en {config_path}")


# ---------------------------------------------------------------------------
# Paso 4: Ejecutar AnyBURL
# ---------------------------------------------------------------------------

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


def run_anyburl(jar_path: Path, config_path: Path) -> None:
    java = _find_java()

    print(f"[4/4] Ejecutando AnyBURL (tiempo máx: {LEARNING_TIME}s) ...")
    print(f"      Snapshots intermedios en: 10s, 50s, 100s, 500s, {LEARNING_TIME}s")
    print(f"      Puedes interrumpir con Ctrl+C — los snapshots ya generados se conservan.\n")

    target = RULES_DIR / f"rules-{LEARNING_TIME}"
    cmd = [
        java, "-Xmx4G", "-cp", str(jar_path),
        "de.unima.ki.anyburl.Learn",
        str(config_path),
    ]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[!] Interrumpido por el usuario.")

    if target.exists():
        n = sum(1 for _ in open(target))
        print(f"  ✓ {target.name}: {n:,} reglas con soporte≥10 y confianza≥0.75")
    else:
        print(f"  [!] No se generó {target.name}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  Aprendizaje de reglas — AnyBURL")
    print(f"  Soporte mínimo : 10 instancias")
    print(f"  Confianza mín. : 0.75")
    print(f"  Tiempo máx.    : {LEARNING_TIME}s")
    print(f"{'='*60}\n")

    if not INPUT_FILE.exists():
        print(f"[!] No se encontró {INPUT_FILE}")
        sys.exit(1)

    n3_to_tsv(INPUT_FILE, TSV_FILE)
    download_jar(JAR_FILE)
    write_config(CONFIG_FILE, TSV_FILE, RULES_DIR)
    run_anyburl(JAR_FILE, CONFIG_FILE)

    print(f"\n{'='*60}")
    print(f"  Reglas guardadas en: {RULES_DIR}/")
    print(f"  Ficheros generados: rules-10, rules-50, rules-100, ...")
    print(f"{'='*60}\n")
