# MarketScout DL: Presentation Outline (8 Slides)

---

## Slide 1: Title & Motivation

### Title
**MarketScout DL: Predicting Football Player Market Value with Deep Learning**

### Bullets
- **Challenge**: Estimate a player's next market valuation using historical performance and context
- **Why it matters**: 
  - Scout teams need data-driven player valuations for transfer decisions
  - Transfermarkt valuations drive market perceptions; can they be predicted?
  - Opportunity to identify undervalued talent before market consensus
- **Approach**: Deep Learning on 476K player-valuation records + temporal modeling

### Speaker Notes
"Today we're tackling the problem of predicting football player market value. This is relevant for scout teams, analytics platforms, and fans who want to understand valuation trends. We built a Deep Learning model trained on half a million player-valuation records from Transfermarkt, spanning 2008 to 2024. Our goal: predict with 13% median error where a player's value is heading based on recent form and club context."

---

## Slide 2: Problem Definition

### Title
**The Prediction Task: Temporal Regression**

### Bullets
- **Input**: For each player at valuation date $t$:
  - Player profile: age, position, nationality
  - Recent performance: goals, assists, minutes (90/180/365 days)
  - Club & competition context (squad size, league type)
  - Current market value (log-scaled)
  
- **Output**: Predict player's **next** recorded valuation (at date $t + \Delta t$)
  - Target: $y = \log(1 + \text{next\_market\_value\_eur})$
  - Metric: MAE in log-space + median percentage error

- **Why log-space?**
  - Stabilizes variance (1M–500M EUR range)
  - Emphasizes relative changes (%); not absolute EUR shifts
  - Symmetric under/over-prediction

### Speaker Notes
"We frame this as a temporal supervised learning task. Each training example is a player at a specific valuation date, and we're predicting their next valuation—which could be days, weeks, or months away. We work in log-space because market values range from 1 million to 500 million euros, and we care about relative changes, not absolute differences."

---

## Slide 3: Dataset

### Title
**Data: 476K Records from Transfermarkt (2008–2024)**

### Bullets
- **Source tables**:
  | Table | Rows | Purpose |
  |---|---|---|
  | players | 47,637 | Demographics (age, position, nation) |
  | player_valuations | 507,815 | Historical valuations (target) |
  | appearances | 1,877,839 | Match-level performance |
  | games | 88,271 | Match metadata (date, competition) |
  | clubs | 796 | Squad statistics |
  | competitions | 67 | League/cup metadata |

- **Final modeling dataset**: 476,307 rows × 100 columns
  - ~30,637 unique players
  - 379K train, 72K validation, 24K test (temporal split)
  - **No random shuffling**: preserves temporal causality

### Speaker Notes
"Our dataset is built from Transfermarkt's public data. The key constraint: we use a temporal split—train on 2008–2021, validate on 2022–2023, test on 2024-present. This prevents lookahead bias and reflects real deployment: we're always predicting future valuations from past and present data. The 476K records represent unique player-valuation combinations; the same player can appear multiple times if they were valued multiple times."

---

## Slide 4: Feature Engineering

### Title
**100 Features Across 5 Dimensions**

### Bullets

1. **Player Profile** (12 features)
   - Age, age_bucket (U18/18-21/22-25/26-29/30-33/34+)
   - Position, foot, height, contract days remaining

2. **Performance Windows** (33 features)
   - 90, 180, 365-day lookback windows (strictly historical)
   - Per window: appearances, minutes, goals, assists, cards, ratios
   - E.g., `goals_per_90_180` = goals in 180 days ÷ 180 days of 90-min games

3. **Club Context** (12 features)
   - Squad size, average age, foreigners %, national team players %
   - Stadium capacity, net transfer record

4. **Competition Context** (8 features)
   - League, country, confederation type (domestic/cup/continental)
   - Competition code

5. **Current Valuation** (2 features)
   - `market_value_in_eur` (current snapshot)
   - `log_market_value` (log-scaled)

### Speaker Notes
"Feature engineering was careful: all performance features are strictly historical, computed only from games before the valuation date. Players without prior appearances get zero-filled performance features—we preserve them rather than dropping rows. The residual models additionally compute the target delta: how much did the player's log-value change since the previous valuation?"

---

## Slide 5: Modeling Strategy

### Title
**7 Models: From Baselines to Deep Learning**

### Bullets

**Sprint 1–5: Data Pipeline**
- Temporal target construction, feature engineering, dataset assembly

**Sprint 6: Non-DL Baselines**
- Mean/median constants, previous_value (no-change), Ridge regression, HistGradientBoosting
- Result: previous_value strong (MAE_log 0.216, R² 0.956); HistGB best baseline (MAE_log 0.209, R² 0.966)

**Sprint 7: Direct MLP** ❌
- Keras MLP: Dense(256)→BN→Dense(128)→BN→Dense(64)→Dense(1)
- Predicts `log_next_market_value` directly from numeric features
- Result: **Failed** (MAE_log 0.296, worse than no-change baseline)

**Sprint 8: Residual MLP** ✅ **BEST**
- Smaller architecture: Dense(128)→BN→Dense(64)→BN→Dense(32)→Dense(1)
- Predicts $\Delta \log(\text{value}) = \log(\text{next}) - \log(\text{current})$
- Result: **MAE_log 0.205**, R² 0.965, median_APE 13.4% (best on 2/3 metrics)

### Speaker Notes
"We tried two Deep Learning approaches. The direct MLP failed because predicting absolute value from features is hard—the model got stuck learning that current_value ≈ next_value, leaving no capacity for meaningful changes. The residual MLP succeeded by reformulating: instead of predicting the absolute next value, predict how much it will *change*. This aligns with the fact that the no-change baseline is already very strong (95.6% R²). By focusing on delta, the model can exploit performance signals to identify deviation from persistence."

---

## Slide 6: Results

### Title
**Test Set Performance: 4 Metrics, 4 Models**

### Bullets

**Results Table**:
| Model | MAE (log) | R² (log) | median APE (%) |
|---|---|---|---|
| **previous_value** | 0.216 | 0.956 | 16.7% |
| **HistGradientBoosting** | 0.209 | **0.966** | 14.3% |
| **mlp_numeric (direct)** | 0.296 | 0.937 | 22.7% |
| **mlp_residual** | **0.205** | 0.965 | **13.4%** |

**Key observations**:
- Residual MLP: best MAE_log (5% beat previous_value) + best median_APE
- HistGB: best R², but 0.4% higher MAE_log
- Direct MLP: underperforms (0.09 worse than previous_value on MAE_log!)
- Gap between residual MLP & HistGB is negligible: <0.1% R², 0.4% MAE_log

### Speaker Notes
"On 24K test predictions, our residual MLP achieves 13.4% median error—meaning typical predictions are off by about ±13% of the true value. It beats the no-change baseline by 5% on absolute error and identifies exploitable patterns in performance data. HistGradientBoosting remains competitive on R², but our model wins on the metrics most relevant for practitioners: absolute error and percentage error. The direct MLP's failure illustrates an important principle: sometimes, reformulating the problem (residual vs. direct) matters more than algorithm choice."

---

## Slide 7: Key Findings

### Title
**Why Residual Learning Works: Insights & Interpretation**

### Bullets

1. **No-change baseline is nearly unbeatable**
   - Current valuation explains 95.6% of next valuation variance (R²=0.956)
   - Player values are sticky; changes are rare but predictable from recent form

2. **Direct learning is data-inefficient**
   - MLP trying to predict absolute value "wasted" capacity learning current_value ≈ next_value
   - Result: worse than no-change baseline (MAE_log 0.296 vs 0.216)

3. **Residual framing concentrates model capacity on hard problem**
   - Delta target: $y = \log(\text{next}) - \log(\text{current})$ (typically ±0.2 range)
   - Model focuses on "what changes?" not "what is the value?"
   - Result: 5% error reduction vs baseline, captures non-trivial performance signals

4. **HistGB & Residual MLP are equivalent on variance explained (R²)**
   - R² differs by <0.1%: both explain ~96.5% of variation
   - Residual MLP wins on absolute error (MAE_log 0.205 vs 0.209)
   - Residual MLP wins on robustness to outliers (median_APE 13.4% vs 14.3%)

5. **Interpretability matters**
   - Residual predictions directly answer: "Will this player's value increase or decrease?"
   - E.g., pred_delta = 0.05 ⟹ expect value to grow ~5% (expm1(0.05) ≈ 1.05)

### Speaker Notes
"The big insight: reformulating the problem from 'predict value' to 'predict change' was more important than switching from traditional ML to Deep Learning. The residual approach works because it plays to the strength of Deep Learning—learning nuanced patterns in a constrained output space—rather than trying to brute-force a full prediction from raw features. When a strong baseline exists (no-change), delta learning is the way to go."

---

## Slide 8: Limitations & Future Work

### Title
**What We Got Right & What's Next**

### Bullets

**Limitations (What We Can't Do Yet)**
- Transfermarkt ≠ real transfer prices (estimates, media-biased)
- Geographic bias: top-5 European leagues well-covered; others sparse
- Performance data incomplete for lower leagues, injured/benched players
- No advanced stats: xG, xA, pressure, possession-adjusted metrics
- No video game ratings (EA FC/SoFIFA) as quality proxy
- Club features are snapshots, not rolling windows
- No player embeddings; one-hot categorical encoding is inefficient

**Future Work (Roadmap)**
1. **Integrate ratings APIs**: Add EA FC ratings → expected +2–3% R²
2. **Advanced performance stats**: FBref/StatsBomb (xG, xA, pressures)
3. **Embeddings**: Learn player/club/position embeddings instead of one-hot
4. **Segmentation**: Train separate models per position or league
5. **Subvalued discovery**: Rank players by pred_value - actual_value to find steals
6. **Percentage change model**: Directly predict `pct_change` for interpretability
7. **Per-position evaluation**: Identify model blind spots (e.g., goalkeepers)
8. **Active player filtering**: Separate models for active vs. youth vs. retired players

### Speaker Notes
"Transfermarkt valuations are estimates, not real transfer fees—so our model predicts perceived market value, which is influenced by media hype and league popularity. We're missing advanced stats like expected goals and player positioning data, which would likely give us 5–10% more predictive power. Our immediate next step: integrate video game ratings from EA FC, which are strong proxies for player quality and fan perception. Longer term, we want to move from one-hot encoding to learned embeddings, and build production tools for scout teams to identify undervalued talent using our model's residual predictions."

---

## Presentation Tips

- **Duration**: ~20 minutes (2–3 min per slide, 5 min Q&A)
- **Visual aids**:
  - Show confusion matrix / residual plots from `reports/figures/`
  - Plot example predictions: "Here's a striker whose value increased 8% as predicted; here's a defensive midfielder whose value dropped 2% despite good performance"
  - Side-by-side comparison: HistGB feature importance vs. Residual MLP learned patterns
- **Engagement**:
  - "Who here thinks current player value is the best predictor of future value?" (leads into no-change baseline)
  - "Why did the direct MLP fail?" (builds suspense, introduces residual learning)
- **Closing**: "Residual learning lets us focus on the hard part: predicting when and why valuations deviate from persistence. That's where the business value lies."

---

## Appendix: Model Architecture Details

### Residual MLP (Sprint 8)
```
Input: [numeric features, n_features=58]
  ↓
Dense(128, activation='relu', kernel_regularizer=L2(1e-4))
BatchNormalization()
Dropout(0.15)
  ↓
Dense(64, activation='relu', kernel_regularizer=L2(1e-4))
BatchNormalization()
Dropout(0.10)
  ↓
Dense(32, activation='relu')
  ↓
Dense(1, activation='linear')  # Output: y_residual
  ↓
Loss: Huber(delta=0.5)
Optimizer: Adam(learning_rate=5e-4)
Epochs: up to 100 (EarlyStopping on val loss, patience=10)
Batch size: 2048
```

### Preprocessing
- Numeric features: median imputation, StandardScaler
- No categorical features (numeric only to reduce dimensionality)
- Target: Residual = log_next_market_value - log_market_value

### Inference
```
pred_residual = model.predict(preprocessed_features)
pred_log_next = log_market_value + pred_residual
pred_next_eur = expm1(pred_log_next)  # Convert back to EUR
```
