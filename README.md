# XAI dla wykrywania włamań sieciowych (SPZC, Zespół 13)

**Autorzy:** Dawid Budzyński, Filip Budzyński

- Artykuł (Overleaf): https://www.overleaf.com/project/6a25a45128d3e2f25a8e1c66
- Repozytorium kodu (GitHub): https://github.com/FilipBudzynski/SPZC-Projekt
- Gotowy artykuł (PDF): `paper/Budzynski-Budzynski-SPZC-projekt.pdf`
- Kod eksperymentów: `src/`, `run.py`, `run_baseline.py`
- Wyniki (metryki, wykresy SHAP, macierze pomyłek): `results/`

Implementacja wyjaśnialnego wykrywania włamań opartego na modelach **Random Forest** i **XGBoost**
oraz metodę **SHAP**, inspirowana pracą Arreche i in. (2024), *XAI-IDS* (DOI: 10.3390/app14104170).
Eksperymenty na zbiorach **UNSW-NB15** oraz **CIC-IoT2023** (klasyfikacja wieloklasowa).
Dodatkowo, dla walidacji poprawności implementacji, ten sam potok uruchamiamy na zbiorze
**NSL-KDD** (użytym w pracy bazowej) — odtwarzamy raportowaną dokładność ~0,99, co potwierdza,
że niższe wyniki na UNSW-NB15/CIC-IoT2023 wynikają z trudności tych zbiorów, a nie z błędu kodu.

## Struktura

```
src/data.py      wczytywanie i preprocessing obu zbiorów
src/train.py     modele RF/XGBoost, trening, metryki
src/explain.py   wyjaśnienia SHAP (TreeSHAP), ważność cech
run.py           pełny eksperyment (UNSW-NB15, CIC-IoT2023) -> results/
run_baseline.py  baseline reprodukcyjny na NSL-KDD -> dopisanie do results/
paper/           artykuł naukowy (LaTeX, IEEEtran)
data/            zbiory danych (pobierane, patrz niżej)
results/         metryki, rankingi SHAP, wykresy
```

## Uruchomienie

```bash
uv sync                     # tworzy .venv i instaluje zależności z pyproject.toml/uv.lock
uv run python run.py            # UNSW-NB15 + CIC-IoT2023
uv run python run_baseline.py   # NSL-KDD (baseline; dopisuje wyniki do results/)
```

## Dane

Skrypt oczekuje plików w katalogu `data/`:

- `unsw_train.csv`, `unsw_test.csv` — UNSW-NB15 (oficjalny podział train/test).
- `ciciot_train.parquet`, `ciciot_test.parquet` — CIC-IoT2023 (podział losowy,
  etykieta `attack_class` z 8 klasami).
- `KDDTrain+.txt`, `KDDTest+.txt` — NSL-KDD (baseline; łączone i dzielone losowo 70/30
  w `src/data.py`, etykiety mapowane na 5 klas: Normal/DoS/Probe/R2L/U2R).

Pobranie:

```bash
curl -L -o data/unsw_train.csv https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv
curl -L -o data/unsw_test.csv  https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv
curl -L -o data/ciciot_train.parquet https://huggingface.co/datasets/lacg030175/CIC-IoT-2023/resolve/main/random/train-00000-of-00001.parquet
curl -L -o data/ciciot_test.parquet  https://huggingface.co/datasets/lacg030175/CIC-IoT-2023/resolve/main/random/test-00000-of-00001.parquet
curl -L -o data/KDDTrain+.txt https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt
curl -L -o data/KDDTest+.txt  https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt
```

## Wyniki

`run.py` zapisuje do `results/`: `metrics.json` (skuteczność, w tym ablacja 15 cech),
`comparison.json` (zgodność rankingów SHAP RF vs XGBoost), `per_class_top.json`
(najważniejsze cechy per klasa), `timing.json` (czasy treningu/inferencji) oraz wykresy
SHAP i macierze pomyłek (`*.png`).

## Artykuł

Źródło artykułu znajduje się w `paper/main.tex` (szablon IEEEtran, język polski przez
`\babelprovide[main,import]{polish}`.
