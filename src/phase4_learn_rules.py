"""
Fase 4 (opcional) — Aprendizaje de reglas Horn con AnyBURL.

Orquesta el toolchain de reglas (paquete src/rules/) en dos pasos:
  1. rules.split_train_full    → divide train_full.ttl en data/train_splits/
                                 (incidents, interventions, employees, ...)
  2. rules.learn_rules_splits  → AnyBURL aprende reglas por split en
                                 data/reglas/<split>/

Requiere Java 17+ (AnyBURL lo descarga el propio toolchain si falta). Es OPCIONAL
y NO forma parte de `--phase all`: las reglas cambian poco y el aprendizaje es
lento. Las reglas resultantes las consumen create_incident (fase 5) y la
evaluación (fase 6) mediante PyClause.

Uso:
  python src/run_pipeline.py --phase 4
  python src/phase4_learn_rules.py
"""

from rules import split_train_full, learn_rules_splits


def run() -> None:
    """Genera los splits temáticos y aprende reglas AnyBURL sobre cada uno."""
    print("[Fase 4] Paso 1/2 — generando splits temáticos en data/train_splits/ ...")
    split_train_full.main()

    print("\n[Fase 4] Paso 2/2 — aprendiendo reglas Horn con AnyBURL ...")
    # requested=[] → procesa TODOS los splits sin leer sys.argv (que aquí
    # contiene los args de run_pipeline, p.ej. ['--phase', '4']).
    learn_rules_splits.main(requested=[])

    print("\n[Fase 4] ✓ Reglas generadas en data/reglas/<split>/")


if __name__ == "__main__":
    run()
