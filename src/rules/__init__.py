"""
Toolchain de reglas Horn con AnyBURL (Fase 4, opcional).

Submódulos:
  split_train_full     — divide train_full.ttl en splits temáticos (data/train_splits/)
  learn_rules_splits   — AnyBURL aprende reglas por split (data/reglas/<split>/)

Orquestado por src/phase4_learn_rules.py. Requiere Java 17+ (lo instala el
propio toolchain si falta).
"""
