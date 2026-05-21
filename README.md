# MarketScout DL: Predicción del valor de mercado de futbolistas con Deep Learning

**Proyecto final – Inteligencia Artificial (Universidad)**

**Autores:** Eduardo Turriza & Alejandro Jiménez

---

## 1. Resumen ejecutivo

MarketScout DL es un sistema de aprendizaje automático que predice el **siguiente** valor de mercado de un futbolista a partir de su perfil histórico, su rendimiento reciente y el contexto de su club y competición. El proyecto compara modelos base (persistencia *no-change*, Ridge, HistGradientBoosting) con dos enfoques de Deep Learning: una MLP de regresión directa y una MLP **residual** (predice el delta sobre la línea base *no-change*).

**Resultado principal:** la MLP residual obtiene el mejor MAE en espacio logarítmico (**0.205**) y el mejor error porcentual absoluto mediano (**13.4 %**), superando a la persistencia (MAE_log = 0.216) e igualando prácticamente el R² de HistGradientBoosting (0.965 vs 0.966, < 0.1 % de diferencia).

| Modelo | MAE (log) | R² (log) | MAE (EUR M) | APE mediano (%) |
|---|---|---|---|---|
| `previous_value` (no-change) | 0.216 | 0.956 | €0.90 M | 16.7 % |
| HistGradientBoosting | 0.209 | **0.966** | €1.04 M | 14.3 % |
| MLP directa | 0.296 | 0.937 | €0.61 M | 22.7 % |
| **MLP residual (ganador)** | **0.205** | 0.965 | €0.81 M | **13.4 %** |

---

## 2. Planteamiento del problema

Dado un registro de valoración `(player_id, valuation_date)` en el instante $t$, el objetivo es predecir el valor de mercado del jugador en la **siguiente** valoración registrada $t + \Delta t$:

$$
y = \log\!\bigl(1 + \text{next\_market\_value\_in\_eur}\bigr)
$$

Se trata, por tanto, de una **regresión temporal** sobre el logaritmo del valor. La transformación `log1p`:

- Estabiliza la varianza en un rango que va de < 100 000 € a 200 000 000 €.
- Pone el énfasis en errores **relativos** (porcentaje) y no absolutos.
- Es simétrica: `expm1(ŷ)` reconstruye el valor en euros.

### Casos de uso

- **Equipos de scouting** – detectar jugadores infravalorados / sobrevalorados.
- **Plataformas analíticas** – contrastar valoraciones de mercado con las del modelo.
- **Aficionados / medios** – explicar variaciones de valor a partir de variables interpretables.

---

## 3. Dataset

### 3.1 Fuente

[Transfermarkt](https://www.transfermarkt.com/) – seis tablas públicas de jugadores, valoraciones, partidos y competiciones.

### 3.2 Tablas originales

| Tabla | Filas | Columnas | Descripción |
|---|---|---|---|
| `players` | 47 637 | 24 | Datos demográficos (edad, posición, nacionalidad, contrato) |
| `player_valuations` | 507 815 | 4 | Histórico de valoraciones (`player_id`, fecha, valor) |
| `appearances` | 1 877 839 | 10 | Estadísticas por partido (goles, asistencias, minutos, tarjetas) |
| `games` | 88 271 | 17 | Metadatos del partido (fecha, equipos, competición) |
| `clubs` | 796 | 24 | Plantilla, edad media, capacidad de estadio |
| `competitions` | 67 | 7 | Liga / copa, confederación, país |

### 3.3 Dataset final de entrenamiento

- **476 307 filas** (registros jugador-valoración).
- **100 columnas** (64 numéricas + 36 categóricas).
- **30 637 jugadores únicos**.
- **2 750 clubes** y **32 competiciones**.
- **Rango temporal:** 2000-01-20 – 2025-12-19.
- **Mediana del objetivo:** 600 000 €. **Media:** 2.68 M €. **Máx.:** 200 M €.

### 3.4 Cobertura por competición (top 10)

| Competición | Filas |
|---|---|
| premier-liga (Rusia) | 54 013 |
| super-lig (Turquía) | 48 867 |
| serie-a (Italia) | 47 661 |
| super-league-1 (Grecia) | 36 287 |
| laliga (España) | 35 278 |
| bundesliga (Alemania) | 33 966 |
| liga-portugal | 29 317 |
| premier-league (Inglaterra) | 28 527 |
| ligue-1 (Francia) | 27 171 |
| eredivisie (Países Bajos) | 26 517 |

---

## 4. Ingeniería de variables

Las variables se construyen en cinco bloques **estrictamente históricos** (todos los agregados usan `game_date < valuation_date`):

### 4.1 Perfil del jugador (12 features)
`age_at_valuation`, `age_bucket` (U18 / 18-21 / 22-25 / 26-29 / 30-33 / 34+), `position`, `sub_position`, `foot`, `height_in_cm`, `contract_days_remaining`, etc.

### 4.2 Ventanas de rendimiento 90 / 180 / 365 días (33 features)

Para cada ventana `W ∈ {90, 180, 365}`:

`appearances_W`, `minutes_W`, `goals_W`, `assists_W`, `yellow_cards_W`, `red_cards_W`, `goal_contributions_W`, `goals_per_90_W`, `assists_per_90_W`, `goal_contributions_per_90_W`, `minutes_per_appearance_W`, `played_any_W`.

### 4.3 Contexto de club (12 features)
Tamaño de plantilla, edad media, número y % de extranjeros, capacidad del estadio, saldo neto de fichajes.

### 4.4 Contexto de competición (8 features)
Nombre, código, país, confederación, tipo (liga doméstica / copa / continental).

### 4.5 Valoración actual (2 features)
`market_value_in_eur` y `log_market_value = log1p(market_value_in_eur)`.

### 4.6 Variables descartadas por *leakage*

- `players.market_value_in_eur` (snapshot, no histórico).
- `highest_market_value_in_eur` (puede incluir futuro).
- `clubs.total_market_value` (100 % nulo) y `clubs.coach_name` (88 % nulo).
- Cualquier columna derivada del objetivo (`next_*`, `value_change_*`).
- `days_to_next_valuation` (no se conoce en producción).

---

## 5. División temporal (sin shuffle)

Para evitar fuga temporal se utiliza un *split* estrictamente cronológico:

| Partición | Regla | Filas | Rango |
|---|---|---|---|
| **Entrenamiento** | `valuation_date < 2022-01-01` | 379 522 | 2008 – 2021 |
| **Validación** | `2022-01-01 ≤ valuation_date < 2024-01-01` | 72 646 | 2022 – 2023 |
| **Test** | `valuation_date ≥ 2024-01-03` | 24 139 | 2024 – presente |

---

## 6. Modelos entrenados

### 6.1 Modelos base (Sprint 6)

- **`mean` / `median`** – constantes (sanity check).
- **`previous_value`** – predice `log_market_value` (no-change). Es la línea base más fuerte: R²_log = 0.956.
- **Ridge** – regresión lineal regularizada (numéricas + categóricas de baja cardinalidad vía OneHot).
- **HistGradientBoostingRegressor** – boosting sobre las 58 variables numéricas.

### 6.2 MLP directa (Sprint 7)

`Dense(256) → BN → Dropout(0.25) → Dense(128) → BN → Dropout(0.20) → Dense(64) → Dropout(0.10) → Dense(1)`

- **Pérdida:** Huber.
- **Optimizador:** Adam.
- **Regularización:** EarlyStopping + ReduceLROnPlateau.
- Preprocesado: `SimpleImputer(median)` + `StandardScaler` para numéricas; `OneHotEncoder(min_frequency=50)` para categóricas en la variante `mlp_tabular`.
- **Resultado:** MAE_log = 0.296 → **peor que la línea base**.

### 6.3 MLP residual (Sprint 8) – modelo ganador

Reformula el problema como predicción del **delta** en espacio logarítmico:

$$
y_{\text{residual}} = \log(\text{next\_value}) - \log(\text{current\_value})
$$

$$
\widehat{\log(\text{next})} = \log(\text{current}) + \hat{y}_{\text{residual}}
$$

Esto convierte a la línea base *no-change* en el caso trivial $\hat{y}=0$, lo que obliga al modelo a centrarse exclusivamente en patrones que **predicen el cambio**.

Arquitectura más pequeña y regularizada:

`Dense(128, L2=1e-4) → BN → Dropout(0.15) → Dense(64, L2=1e-4) → BN → Dropout(0.10) → Dense(32) → Dense(1)`

- **Pérdida:** `Huber(delta=0.5)`.
- **Optimizador:** `Adam(lr=5e-4)`.
- **Batch size:** 2 048. **Épocas máx.:** 100. EarlyStopping sobre `val_loss`.
- **Sólo features numéricas** (54), para evitar problemas de memoria con OHE.
- Estadísticas del residual en entrenamiento: media = 0.074, mediana = 0.000, σ = 0.445.

---

## 7. Resultados detallados

### 7.1 Conjunto de test (24 139 predicciones)

| Modelo | MAE (log) | RMSE (log) | R² (log) | MAE (EUR) | RMSE (EUR) | Median Abs Err (EUR) | APE mediano (%) |
|---|---|---|---|---|---|---|---|
| `mean` | 1.493 | 1.883 | -0.208 | 5.50 M | 13.64 M | 627 k | 83.1 |
| `median` | 1.530 | 1.937 | -0.277 | 5.52 M | 13.67 M | 600 k | 85.0 |
| `previous_value` | 0.216 | 0.358 | 0.956 | 898 k | 2.30 M | 100 k | 16.7 |
| Ridge | 0.218 | 0.325 | 0.964 | 1.03 M | 3.98 M | 170 k | 15.1 |
| HistGB | 0.209 | 0.317 | **0.966** | 1.04 M | 3.96 M | 166 k | 14.3 |
| MLP directa | 0.296 | – | 0.937 | 0.61 M | – | – | 22.7 |
| **MLP residual** | **0.205** | 0.320 | 0.965 | 0.81 M | 2.10 M | 158 k | **13.4** |

### 7.2 Conjunto de validación

| Modelo | MAE (log) | R² (log) | APE mediano (%) |
|---|---|---|---|
| `previous_value` | 0.213 | 0.957 | 16.4 |
| HistGB | 0.206 | 0.967 | 13.8 |
| **MLP residual** | **0.202** | 0.966 | 13.1 |

La similitud entre validación y test indica **buena generalización**, sin sobreajuste apreciable.

### 7.3 Ganador por métrica

| Métrica | 1.º | 2.º | 3.º | 4.º |
|---|---|---|---|---|
| MAE (log) | **MLP residual** 0.205 | HistGB 0.209 | previous_value 0.216 | MLP directa 0.296 |
| R² (log) | HistGB 0.966 | **MLP residual** 0.965 | previous_value 0.956 | MLP directa 0.937 |
| APE mediano | **MLP residual** 13.4 % | HistGB 14.3 % | previous_value 16.7 % | MLP directa 22.7 % |
| MAE (EUR) | MLP directa 0.61 M | MLP residual 0.81 M | previous_value 0.90 M | HistGB 1.04 M |

> **Nota metodológica:** el MAE en euros está fuertemente sesgado por jugadores top (cuyo valor llega a 200 M €) frente a canteranos (< 1 M €). Por eso se utilizan **MAE_log** y **APE mediano** como métricas principales; el MAE en euros sólo se reporta como referencia.

---

## 8. Discusión y hallazgos clave

1. **La persistencia es casi imbatible.** El valor actual explica por sí solo el 95.6 % de la varianza del valor siguiente: las valoraciones son muy *sticky*.
2. **La regresión directa es ineficiente.** La MLP directa malgasta capacidad aprendiendo la identidad `current ≈ next`. Su MAE_log (0.296) es **peor** que la línea base trivial.
3. **El planteamiento residual desbloquea el aprendizaje útil.** Al pedir al modelo sólo el *delta*, éste se concentra en los patrones realmente predictivos (rendimiento → cambio de valor).
4. **HistGB y MLP residual son prácticamente equivalentes en R²** (0.966 vs 0.965, < 0.1 %), pero la MLP residual gana en MAE_log y en APE mediano.
5. **Las métricas cuentan historias distintas.** MAE en euros se ve distorsionado por outliers; usar MAE_log + APE mediano evita ese sesgo.

---

## 9. Estructura del repositorio

```
football_market_value-dl/
├── app.py                          # Demo Streamlit (predicción individual)
├── requirements.txt                # Dependencias del proyecto
├── data/
│   ├── raw/                        # CSVs originales de Transfermarkt (no versionado)
│   ├── interim/                    # Datasets intermedios por sprint (no versionado)
│   └── processed/                  # Dataset final 476 307 × 100
├── notebooks/                      # 13 notebooks de análisis y entrenamiento
├── src/                            # Pipeline en módulos Python
│   ├── data_understanding.py       # Sprint 1
│   ├── make_targets.py             # Sprint 2
│   ├── make_player_features.py     # Sprint 3
│   ├── make_performance_features.py# Sprint 4
│   ├── make_final_dataset.py       # Sprint 5
│   ├── train_baselines.py          # Sprint 6
│   ├── train_dl.py                 # Sprint 7
│   ├── train_residual_dl.py        # Sprint 8
│   ├── predict.py / predict_single.py # Inferencia
│   ├── evaluate.py                 # Métricas
│   ├── features.py / build_dataset.py
│   ├── load_data.py                # Lectura de CSVs
│   └── config.py                   # Rutas y constantes
├── models/                         # Artefactos entrenados (.keras / .joblib)
└── reports/
    ├── figures/                    # Curvas, residuos, real vs predicho (10 PNG)
    ├── metrics/                    # Métricas por sprint (9 JSON)
    ├── final_model_comparison.md
    ├── project_summary.md
    └── presentation_outline.md
```

---

## 10. Reproducibilidad

### 10.1 Requisitos previos

- Python 3.10 / 3.11.
- ~3 GB de RAM libres para los pasos pesados (Sprint 4-5).
- (Opcional) TensorFlow CPU para los modelos de DL.

### 10.2 Instalación

```bash
git clone <url-del-repo>
cd football_market_value-dl
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 10.3 Datos de entrada

Los CSV originales **no se versionan** (>200 MB). Descárgalos del dataset público de Transfermarkt en [Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores) y colócalos en `data/raw/`:

```
data/raw/
├── players.csv(.gz)
├── player_valuations.csv(.gz)
├── appearances.csv(.gz)
├── games.csv(.gz)
├── clubs.csv(.gz)
└── competitions.csv(.gz)
```

### 10.4 Pipeline completo

```bash
python -m src.data_understanding           # Sprint 1: exploración
python -m src.make_targets                 # Sprint 2: definición del objetivo
python -m src.make_player_features         # Sprint 3: perfil de jugador
python -m src.make_performance_features    # Sprint 4: ventanas de rendimiento
python -m src.make_final_dataset           # Sprint 5: contexto club + competición
python -m src.train_baselines              # Sprint 6: modelos base
python -m src.train_dl                     # Sprint 7: MLP directa
python -m src.train_residual_dl            # Sprint 8: MLP residual (ganador)
```

O, de forma equivalente, ejecutar los notebooks en `notebooks/` en orden numérico.

### 10.5 Demo interactiva (Streamlit)

```bash
streamlit run app.py
```

La app carga el modelo residual entrenado (`models/mlp_residual.keras`) y permite editar el perfil de un jugador ficticio, su valor de mercado actual, su rendimiento reciente y su contexto de club para obtener una predicción del **siguiente** valor de mercado:

```
pred_log_next  = log_market_value + predicted_residual
pred_next_eur  = expm1(pred_log_next)
```

> **Requiere:** `models/mlp_residual.keras`, `models/residual_dl_preprocessor.joblib` y `data/processed/player_market_value_dataset.parquet`.

---

## 11. Limitaciones

1. **Transfermarkt ≠ precios reales de traspaso.** Las valoraciones son estimaciones y están sujetas a sesgo mediático y al efecto recencia.
2. **Sesgo geográfico.** El Top 5 europeo está sobre-representado; otras regiones quedan dispersas.
3. **Datos de rendimiento incompletos.** Jugadores de divisiones inferiores o lesionados tienen registros escasos; rellenar con ceros puede infravalorar talentos jóvenes.
4. **Variables de club como *snapshots*.** `club_squad_size`, `club_average_age`, etc., no son medias móviles.
5. **Sin estadísticas avanzadas.** No hay xG, xA, métricas ajustadas por posesión ni datos de presión.
6. **Sin ratings externos.** No se integran calificaciones tipo EA FC / SoFIFA.
7. **Sin embeddings.** La codificación categórica es OneHot; embeddings serían más compactos e informativos.

---

## 12. Trabajo futuro

1. Integrar ratings EA FC / SoFIFA – mejora esperada de +2 / +3 % en R².
2. Añadir estadísticas avanzadas de FBref / StatsBomb (xG, xA, *progressive passes*).
3. Sustituir OneHot por **embeddings** aprendidos para posición, club y competición.
4. Modelos **por posición** o por liga (segmentación).
5. *Ranking* de jugadores infravalorados (`pred_value − actual_value`) para scouting.
6. Modelo directo sobre el cambio porcentual para mayor interpretabilidad.
7. Intervalos de incertidumbre con *Bayesian Dropout* sobre la MLP residual.

---

## 13. Stack técnico

| Capa | Herramientas |
|---|---|
| Lenguaje | Python 3.10+ |
| Datos | pandas, numpy, pyarrow (Parquet) |
| ML clásico | scikit-learn (Ridge, HistGradientBoosting, OneHotEncoder, StandardScaler) |
| Deep Learning | TensorFlow / Keras (Sequential MLP, BN, Dropout, L2, Huber, Adam) |
| Visualización | matplotlib, seaborn |
| Demo | Streamlit |
| Persistencia | joblib (preprocesadores), formato `.keras` (modelos) |
| Notebooks | Jupyter |

---

## 14. Autores

- **Eduardo Turriza**
- **Alejandro Jiménez**

Trabajo desarrollado como **proyecto final** de la asignatura de Inteligencia Artificial.

---

## 15. Licencia y datos

- El código de este repositorio se distribuye con fines **académicos y educativos**.
- Los datos originales pertenecen a [Transfermarkt](https://www.transfermarkt.com/) y se usan únicamente con fines de investigación, conforme a la licencia del [dataset público de Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores).
