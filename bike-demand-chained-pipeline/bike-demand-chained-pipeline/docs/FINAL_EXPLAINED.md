# The whole project, explained from first principles

This document teaches everything in the project: every idea, every choice, every number, in the order the pipeline runs. It assumes Lectures 1–3 of ICS 504 (linear regression and gradient descent; polynomial regression, bias–variance, Ridge/Lasso/Elastic Net; logistic regression) and Practice Assignments 1–3, and nothing more.

**How to read the tags**
- **(Lecture n / Practice n)**: an idea from the course. I do not have the slides, so these tags follow the topic list of the course (Lecture 1: linear regression and gradient descent; Lecture 2: polynomial regression, bias–variance, regularization; Lecture 3: logistic regression) and the exercises named in our discussions (Practice 2-1 sums, Practice 2-6 Ridge, Practice 3 odds ratios).
- **[From the brief]**: something the project brief prescribes.
- **[Beyond the course]**: something we used that the lectures do not cover. These are the ideas to study most carefully before the evaluation.

**Where the numbers come from.** Every number is from the pipeline as it was actually run: the executed notebook `notebooks/final_submission.ipynb` and the artifacts in `experiments/`. Cell references like `In [5]` are the execution numbers in the saved notebook. A few small worked examples (such as a single scaler value) were computed with the same pipeline code from the saved artifacts; they describe the same fitted models.

---

## 1. The problem, the data and every column

### The operator's problem
A bike-sharing operator must decide, every hour, how many bikes to place at its stations. Too few bikes means empty stations and lost customers; too many means paying trucks to move idle bikes. **Under-supplying the busy hours is the expensive mistake.** That is why, besides the brief's R² and RMSE, we checked every model on the busiest hours (the "operator checks" below).

### The data
- `train.csv`: **10,886 labeled hours** over **456 days**, 2011–2012. Only **days 1–19 of each month** are included.
- `test.csv`: **574 hours**, all on **day 20** of a month, in both years. The test is therefore not "the future": it is unseen days *inside* the two years.
- **58 hours are missing**: 456 days × 24 = 10,944 would be complete. No hour has 0 rentals, so missing hours are simply absent rows, not zeros.
- Demand (`cnt`): mean **191.6**, median **145**, maximum **977**. The mean is above the median because a few very busy hours pull it up; the distribution is skewed.

### Every column
| Column | Meaning | What we did with it |
|---|---|---|
| `instant` | row number (1 to 17,117 across both files) | Identifier. Not a feature; kept to align the submission file. |
| `dteday` | the date | Its year, month and weekday are *exactly* `yr`, `mnth`, `weekday` (checked: 0 mismatches). Used through those columns. |
| `season` | 1 winter … 4 autumn | In this file season is an exact function of month (Jan–Mar = 1, …). Used only inside interactions. |
| `yr` | 0 = 2011, 1 = 2012 | Growth: training-hour mean demand **143.7 in 2011 vs 237.8 in 2012**. |
| `mnth` | 1…12 | One-hot (11 dummies). |
| `hr` | 0…23 | One-hot (23 dummies). The single most important column. |
| `holiday` | 1 on public holidays | 0/1 number. |
| `weekday` | 0 = Sunday … 6 = Saturday | One-hot (6 dummies). |
| `workingday` | 1 if neither weekend nor holiday | Main effect left out (exact combination of weekday and holiday); used inside interactions. |
| `weathersit` | 1 clear … 4 heavy rain/snow | One-hot (2 dummies); the single level-4 row was merged into 3. |
| `temp` | (t + 8)/47, t in °C | Number. Example: 0.80 = 29.6 °C. |
| `atemp` | "feels-like", (t + 16)/66 | Left out: correlation 0.984 with `temp` in our training rows. |
| `hum` | humidity / 100 | Number. |
| `windspeed` | wind / 67 | Number. |
| `cnt` | rentals in that hour | The target. Never a feature. |

**Say it at the evaluation:** "We predict hourly rentals for an operator who loses most when busy hours run out of bikes. So besides R² we always checked the busiest hours. The test is day 20 of each month, inside the period we train on."

---

## 2. The seed

### What it is [From the brief]
1. Write our three student IDs as text and **sort** them: `16001061`, `16001322`, `16008386`.
2. **Join them with underscores:** `16001061_16001322_16008386`.
3. Compute the **SHA-256 hash**. A hash is a fixed, scrambling function: the same text always gives the same 256-bit number, and a tiny change in the text gives a completely different number. Ours starts `686c269a146ca60b…`.
4. Read that hash as an integer and take it **mod 100,000** (the remainder after dividing by 100,000): **36954**.

The code first checks that the brief's example IDs give **41698** (`In [1]`). Sorting makes the order of names irrelevant.

### What a seed does to a shuffle
`train_test_split` shuffles the rows with a pseudo-random number generator before cutting 80/20. A pseudo-random generator is deterministic: start it from the same seed and it produces the same "random" sequence every time. With seed 36954, the first training rows are `instant` 3910, 11735, 325, 12621, 15243, …, and the first validation rows 8060, 10396, 3830, …. With another seed, a different 20% of hours would be validation.

### Why each team gets different weights and λ
A different split means slightly different training rows. Gradient descent then reaches slightly different weights, and every later choice (λ, degree, cutoff) is tuned on a different validation set. A copied answer would not match the team's own pipeline.

### Why the TAs can verify us
Everything after the seed is deterministic:
- the split is fixed;
- gradient descent starts from zero weights;
- the learning-rate rule is fixed (1.9/λ_max);
- the updates are full-batch (no randomness).

A TA can recompute 36954 from our roster, re-run Phase 1, and get the same weights that went into Phase 2.

### Collision odds
There are 100,000 possible seeds. The chance that *some* two teams share one is about 1 − exp(−N(N−1)/200,000), the "birthday problem": **0.4%** for 30 teams, **1.8%** for 60, **4.8%** for 100, **10.6%** for 150. The chance that one *specific* other team has our seed is 1 in 100,000.

**Say it:** "The seed is the SHA-256 of our sorted, underscore-joined IDs mod 100,000, so 36954. It fixes the shuffle, so our split and therefore our weights are reproducible; a TA can recompute it and re-run Phase 1."

---

## 3. Phase 1: Linear regression with gradient descent (Lecture 1 / Practice 1)

### 3.1 One-hot encoding (encoding categories) [Beyond the course in this form]
**What.** A categorical column with k levels becomes k − 1 columns of 0/1 ("dummies"), one per level except a *reference* level. Hour 08:00 becomes `hr=8` = 1 and all other hour dummies 0; hour 00:00 (the reference) is all zeros.

**Why hour cannot be a number.** A linear model gives each feature one weight, so `hr` as the number 0–23 would force demand to change by the same amount every hour: a straight line across the day. Real demand has two working-day peaks (08:00 and 17:00–18:00) and a weekend midday hump. With 23 dummies, each hour gets its own level.

**Why drop one level.** The intercept already represents "the baseline". If all 24 hour dummies were present, their sum would always be 1, the same as the intercept column. Then the weights would not be unique (exact collinearity).

**Our choice.**
- One-hot: hr (23), weekday (6), weathersit (2), month (11).
- Numbers: yr, holiday, temp, hum, windspeed.
- Total **47 features + intercept**.

### 3.2 Why some columns were left out
- **workingday:** in our training rows, workingday = (Monday–Friday) − holiday, exactly. With the weekday dummies, holiday and the intercept present, it is a fixed linear combination of them, so its weight would not be unique. We kept weekday + holiday instead because they say *more*: Saturday vs Sunday, Monday vs Friday.
- **atemp:** correlation 0.984 with temp. Almost the same information twice; its verdict was tested properly in Phase 4.
- **season vs month:** season is an exact function of month here, so only one can be used. Month gave validation R² **0.6823** against **0.6805** for season (`In [3]`).
- **dteday's leftover information:** the elapsed-days trend added only **0.0005** R² but raised the condition number from **56 to 8,084**, so gradient descent needed **36,407** updates instead of **512**. We left it out. (The condition number is explained in 3.5.)
- **weathersit = 4:** a single labeled hour, and it fell in validation, so no weight could be learned for it. We mapped it to 3, the nearest severe level.

### 3.3 Standardization (feature scaling, Lecture 1)
**What.** Replace each column x by z = (x − mean)/SD, with the mean and SD computed on the training rows only.

**Example from our data.** temp has training mean **0.4934** and SD **0.1906**. An hour at temp 0.80 (29.6 °C) becomes z = (0.80 − 0.4934)/0.1906 = **1.609**.

**Why.** All columns end up on the same scale, so one learning rate suits every weight. A side effect: because every column has mean 0, the intercept weight equals the mean demand. Ours is **190.97**, exactly the mean training `cnt`.

**Never on validation or test.** The validation and test rows are transformed with the *training* mean and SD. Fitting a scaler on validation would leak information.

### 3.4 Cost function and gradient (Lecture 1)
- **Cost function.** J(w) = (1/2n)·Σ(prediction − actual)² = ‖Xw − y‖²/(2n). This is RSS divided by 2n.
- **Gradient.** ∂J/∂w = Xᵀ(Xw − y)/n.

  Each entry j is the average of (error × feature j). For example, if the model under-predicts at 08:00 on working days, the `hr=8` entry is negative, and the update w ← w − η·gradient raises that weight.
- **Batch gradient descent.** Every update uses all 8,708 training rows.

### 3.5 The bowl picture and why the step must be below 2/λ_max [Beyond the course: eigenvalues]
- For linear regression, J is a **bowl** (a quadratic). Its steepness differs by direction: in some directions the walls are steep, in others the bowl is long and flat.
- The steepest curvature is **λ_max**, the largest eigenvalue of XᵀX/n. Ours: **1.9894**. The flattest is λ_min = **0.0356**.
- The ratio λ_max/λ_min is the **condition number κ**: **55.8** for Phase 1. It measures how "stretched" the bowl is.

**The one-weight example.** Imagine a single weight with cost J(w) = (a/2)(w − w*)². Its gradient is a(w − w*), so one update gives

  w_new − w* = (1 − η·a)(w − w*).

The error is multiplied by (1 − ηa) at every update. It shrinks only if |1 − ηa| < 1, i.e. **0 < η < 2/a**. In many dimensions the steepest direction has a = λ_max, so we need **η < 2/λ_max = 1.0053**.

| Learning rate | Factor on steepest direction (1 − η·λ_max) | What happened (`In [4]`) |
|---|---|---|
| 0.005/λ_max | +0.995 | far too slow: hit the **50,000** cap, max\|gradient\| still 0.0104 |
| 1.0/λ_max | 0.000 | converged in 512 updates |
| **1.9/λ_max = 0.955** | **−0.900** (flips sign but shrinks) | **converged in 267 updates** (chosen) |
| 2.1/λ_max = 1.056 | **−1.100** (flips sign and grows) | **diverged after 84 updates** |

At 2.1 the error grows by 1.1 per update, so the cost grows by 1.1² = 1.21 per update. It passes our "exploded" limit (10⁶ × the starting cost) after about 73 updates, plus a short start-up: hence 84.

Why does 1.9 converge fastest? The *flat* directions shrink by 1 − η·λ_min. With η = 1.9/λ_max that factor is 0.966 per update, with η = 1/λ_max it is 0.982. The larger safe step finishes the flat directions sooner.

### 3.6 Convergence evidence
- **Stopping rule.** Stop when the largest |gradient entry| < **10⁻⁴**. Also stop if the cost explodes (diverged) or after 50,000 updates (cap).
- **Why not "the loss curve is flat"?** The 0.005 run *looked* flat but still had gradient 0.0104: it was crawling, not finished. At the true minimum the gradient is exactly zero.
- **Independent check.** Solving the least-squares problem exactly (the normal equation, via `numpy.linalg.lstsq`) gives predictions within **0.0074 bikes** of ours (`In [5]`).

### 3.7 Results (`In [5]`, `In [6]`)
- Training R² **0.6970**, validation R² **0.6823** (RMSE **103.6** bikes). The small gap (0.015) means **underfitting** (high bias), not overfitting.
- **Operator checks [Beyond the course]:**
  - Hours above **452** bikes (the training 90th percentile) were under-predicted by **176.6** bikes on average.
  - Even the hours the *model itself* predicted busiest came in **95.9** bikes too low on average: an additive model cannot reach the peaks.
  - **213** validation hours were predicted below zero.
  - The worst misses: weekend 11:00–15:00 (121–154 bikes short) and working-day 08:00 (118 short).

**Handed to Phase 2:** the 48 weights, learning rate 0.955, 267 updates, the feature order and the fitted scaler (`experiments/phase1/phase1_artifact.json`).

**Say it:** "Phase 1 is batch gradient descent on J = ‖Xw − y‖²/2n with gradient Xᵀ(Xw − y)/n. The step must be below 2/λ_max because each update multiplies the error along the steepest direction by 1 − ηλ_max; 2.1/λ_max diverged and 1.9/λ_max converged in 267 updates. R² 0.682: it underfits, because hours have different shapes on working days and weekends."

---

## 4. The bonus: asymmetric cost [Beyond the course]

### The loss
Under-predicting costs 3× as much as over-predicting by the same amount. We kept the *squared* error and weighted it:

  ℓ(r) = ½·c·r², with r = prediction − actual, and c = 3 if r < 0 (under-prediction), else 1.

### Its gradient
- On each side of 0 the derivative is c·r: 3r on the left, r on the right. At 0 both sides are 0, so the slope is continuous and the loss is smooth enough for gradient descent.
- Gradient: **Xᵀ(c ⊙ r)/n** (⊙ = multiply row by row).
- Equivalently, MSE gradient + 2·Xᵀmin(r, 0)/n: the ordinary gradient plus an extra push from under-predicted rows only.
- The minimizer of this loss is the **0.75 expectile**, an "asymmetric mean" that leans upward, not the mean.

**Why not the absolute ("pinball") version?** A cost linear in bikes (like the newsvendor problem) would target the 75th percentile. But it has no gradient at r = 0, and its slope never shrinks, so fixed-step gradient descent with our gradient stopping rule could not converge on it.

### Why the step had to shrink by about 3×
The curvature is now up to 3·λ_max (the weight 3 multiplies the bowl's steepness). So the safe step is 2/(3λ_max) = 0.335 instead of 1.005.
- Phase 1's rate **0.955** diverged within **12** updates.
- 0.955/3 = **0.318** converged in **388** updates, and its loss never increased (`In [8]`).

### How predictions shifted (`In [9]`–`In [11]`)
- Mean prediction rose by **42.6** bikes/hour; only 0.2% of hours went down.
- R² fell **0.682 → 0.606** and RMSE rose 103.6 → 115.3. That is expected: they score the MSE objective, which we no longer minimize.
- The asymmetric cost fell **11,145 → 9,343** (−16.2%).
- For the operator: bikes short per hour **37.2 → 21.2**, idle bikes **39.6 → 66.1**. The top-decile shortfall fell 176.6 → 112.1, but 88% of those busy hours were still under-predicted.
- **Where it went: mostly a broad lift.** The intercept rose 36.9 bikes for every hour.
- **Why:** Phase 1 underfits. 44.4% of its residual variance lay between hour × day-type cells, and half the validation hours (50.5%) sat in cells it already over-predicted, yet they received 48.6% of the upward shift. Example: weekend 08:00 was predicted at 383 against an actual mean of 106, and was raised by another 98.

### The follow-up on Phase 2's features (`In [19]`–`In [20]`)
The same loss on the better 242-feature design:

| | Phase 1 features | Phase 2 features |
|---|---:|---:|
| average upward shift | +42.6 | **+18.9** |
| R² cost of the hedge | −0.076 | **−0.016** |
| top-decile shortfall (by actual demand) | 176.6 → 112.1 | 58.9 → **34.1** |

The largest shifts were now at weekend afternoons and working-day 17:00–18:00, where demand genuinely varies.

**Lesson.** The asymmetric loss moves errors toward surplus but cannot create structure the features lack. Once underfitting is removed, the buffer is cheap and well aimed.

**Say it:** "We weighted under-predictions 3× in a squared loss, so the gradient is Xᵀ(c ⊙ r)/n. The curvature triples, so the step had to shrink 3×. Predictions moved up 42.6 bikes, R² dropped as expected, and the asymmetric cost fell 16%. On Phase 2's features the same hedge cost only 0.016 R²."

---

## 5. Phase 2: Polynomial regression (Lecture 2)

### 5.1 The residual GROUP BY diagnosis [Beyond the course]
**Residual** = actual − predicted. If a model is right on average for a group of hours, that group's mean residual is near 0.

**Tiny example from our data** (Phase 1 training residuals, `In [13]`):

| hour | working day? | mean residual | training rows |
|---|---|---:|---:|
| 08:00 | no (weekend/holiday) | **−252.4** | 109 |
| 08:00 | yes | **+107.5** | 256 |
| 08:00 | both together | 0.0 | 365 |

Phase 1 has an `hr=8` dummy, so on average over all 08:00 hours it is right (0.0). But it over-predicts weekend mornings by 252 bikes and under-predicts working-day mornings by 108. One hour level cannot fit two day types. That is exactly what an **interaction term** hr=8 × workingday fixes.

**The measures we used:**
- **Between-group share:** of all the variance of the residuals, how much is "explained" by the group means.
  - Group by hour × workingday: **0.444**. Systematic error that one interaction group could remove.
- **Adjusted share:** the same, corrected for the number of groups (like adjusted R²). Many small groups can explain a bit of noise by chance; this correction removes that advantage.
  - Example: hour × weekday has 168 groups and share 0.450, but adjusted only **0.439**; hour × workingday with 48 groups is adjusted **0.441**. So workingday is the better choice.
- **Sparse cells:** groups with fewer than 30 training rows. A mean of 30 rows is still uncertain by about ±18 bikes.
  - hour × month has **105** sparse cells; hour × holiday has 24 groups of only **7–10** rows.

**Ranked table (Step A, training rows only):**

| combination | groups | share | adjusted | sparse groups |
|---|---:|---:|---:|---:|
| hour × weekday | 168 | 0.450 | 0.439 | 0 |
| **hour × workingday** | 48 | 0.444 | **0.441** | 0 |
| hour × month | 288 | 0.130 | 0.100 | 105 |
| temp quintile × hour | 120 | 0.112 | 0.100 | 4 |
| **hour × yr** | 48 | 0.107 | 0.102 | 0 |
| hour × season | 96 | 0.094 | 0.084 | 0 |
| temp × hum quintiles | 25 | 0.039 | 0.036 | 0 |
| hour × weathersit | 72 | 0.029 | 0.021 | 13 |
| windspeed × hour | 120 | 0.013 | −0.001 | 1 |
| hour × holiday | 48 | 0.008 | 0.002 | 24 |

### 5.2 Systematic vs random error (bias vs irreducible error, Lecture 2)
**The bathroom-scale idea.**
- A scale that always reads **+2 kg too high** has *systematic* error, called **bias**. You can fix it, because it is predictable.
- A scale that jumps ±0.5 kg at random has *random* error, called **noise** or **irreducible error**. No correction can predict it.

Group means are systematic: if weekend 08:00 is always 252 bikes too high, a term for "weekend 08:00" can remove it. The scatter of individual hours *around* their group mean is partly random (a sudden rain shower, an event) and partly other missing structure.

### 5.3 The interaction groups and why (re-ranking, `In [14]`)
After the obvious winner, we **re-ranked** the remaining candidates on what was left. Several candidates overlap: month, season and temperature all follow the yearly cycle. Removing hour × workingday and hour × yr first left **44.9%** of Phase 1's residual variance.

| Group | Degree | Columns | Why |
|---|---|---:|---|
| hour × workingday | 2 | 23 | largest systematic error (weekday vs weekend daily shape) |
| hour × yr | 2 | 23 | 2012 peaks grew more than 2011 nights (demand scales up, it does not just shift) |
| hour × season | 2 | 69 | beat hour × temp: after season, temp adds only 0.006; after temp, season still adds 0.028 |
| hour-block × weathersit | 2 | 8 | five day parts × weather. Nearly as good as full hour × weathersit (72 columns, 13 sparse) with only 15 cells |
| temp × hum | 2 | 1 | cheap (one column), adjusted share 0.032 |
| hour × workingday × yr | 3 | 24 | weekday peaks grew more than other hours |
| hour × workingday × temp | 3 | 47 | temperature affects weekend leisure hours differently; beat the season version (0.0246 vs 0.0229, and no sparse cells) |

**Rejected:** hour × weekday (worse adjusted share), hour × month (105 sparse cells), hour × holiday (7–10 rows per cell), windspeed × hour (adjusted share below 0), temp² (0.003).

### 5.4 Degree as interaction order (Lecture 2: polynomial features)
- In the lecture, "degree" means powers: x, x², x³. Here most columns are 0/1 dummies, and powers of a 0/1 column do nothing (1² = 1, 0² = 0).
- So **degree = how many columns are multiplied**:
  - degree 1 = Phase 1;
  - degree 2 = pairs (hour × workingday);
  - degree 3 = triples (hour × workingday × temp).
- Two levels of the same column are never multiplied, because their product is always 0 (an hour cannot be both 08:00 and 09:00).
- **Why not expand everything?** All pairs of 47 columns would give over 1,000 mostly useless or always-zero columns. The brief says "expanding every numeric column is neither required nor advisable". Explicit lists chosen from the residuals are the controlled way.

### 5.5 Centering [Beyond the course]
**What.** Before multiplying, every factor except the hour dummy is shifted by its training mean. Instead of hr=8 × workingday we use hr=8 × (workingday − **0.681**).

**Why it lowers the condition number.** The raw product hr=8 × workingday is 1 on working-day 08:00 hours and 0 elsewhere. It is almost a copy of hr=8 itself: correlation **0.832**. Two nearly identical columns make the bowl very stretched along the direction "raise one, lower the other". After centering, the correlation is **0.044**.
- Full design: condition number **21,616 → 1,713**. Gradient descent needs fewer updates, roughly in proportion.

**Why it does not change predictions.** hr=8 × (workingday − 0.681) = hr=8 × workingday − 0.681 × hr=8. Since hr=8 is already in the model, its weight simply absorbs the −0.681 × hr=8 part. The set of predictions the model can make is identical.

### 5.6 Warm start [From the brief; method Beyond the course]
- The brief requires Phase 2 to start from Phase 1's weights.
- Each Phase 1 weight goes into its own column; every new interaction weight starts at **0**.
- Because new columns are centered and standardized (mean 0) and start at weight 0, the starting model predicts **exactly** what Phase 1 predicted. We assert this before training: largest difference **4.5·10⁻¹³** bikes (`In [15]`).

### 5.7 Results (`In [16]`–`In [18]`)
| added group | features | train R² | validation R² | top-predicted-decile shortfall |
|---|---:|---:|---:|---:|
| (Phase 1) | 47 | 0.6970 | 0.6823 | 134.9 |
| hour × workingday | 70 | 0.8316 | **0.8310** | 88.0 |
| hour × yr | 93 | 0.8641 | 0.8675 | 77.4 |
| hour × season | 162 | 0.8933 | 0.8937 | 56.5 |
| hour-block × weathersit | 170 | 0.8979 | 0.9004 | 54.1 |
| temp × hum | 171 | 0.8993 | 0.9009 | 53.8 |
| hour × workingday × yr | 195 | 0.9078 | 0.9105 | 50.4 |
| hour × workingday × temp | 242 | 0.9178 | **0.9225** | **43.3** |

- **Degree 3 chosen** (highest validation R²): RMSE **51.1**, **3,884** updates, within 0.06 bikes of the exact solution.
- **Validation R² above training R²?** Training and validation RMSE are almost equal (51.74 vs 51.14). The validation target varies 3.6% more, which raises its R², and R² on random validation-sized subsets of training rows ranges **0.909–0.926**. The gap is noise.
- hour × workingday alone gave **+0.149**: the biggest single step in the project.

**Say it:** "We grouped Phase 1's training residuals by candidate interactions. Hour × working day held 44% of the error as systematic group means, for example weekend 08:00 over-predicted by 252 bikes. Degree here means how many columns are multiplied. Centering kept the products from copying the hour dummies, which cut the condition number from 21,616 to 1,713 without changing what the model can predict. Degree 3 reached R² 0.9225."

---

## 6. The precomputed table H = XᵀX/n

### Why it is identical math
J = ‖Xw − y‖²/(2n) expands to

  J = ½·wᵀ(XᵀX/n)w − wᵀ(Xᵀy/n) + yᵀy/(2n).

Define **H = XᵀX/n** (a p × p table of averages of products of features) and **b = Xᵀy/n** (averages of feature × target). Then

  **gradient = Xᵀ(Xw − y)/n = H·w − b.**

This is algebra, not an approximation. H and b are the **same sums** you computed in **Practice 2-1** for the normal equations (Σx², Σxy, …), divided by n.

### Why it is faster
- The explicit gradient touches all n = 8,708 rows × p columns every update.
- H·w touches only p × p numbers: 565 × 565 for our largest model.
- We compute H and b **once per fit**, then every update is cheap.

In our measurement the degree-3 fit took **0.4 s instead of 26 s** (about 65×), and the full notebook went from **1,422 s to 337 s**. Every fit asserts at its start and end that H·w − b equals Xᵀ(Xw − y)/n, and all weights matched the explicit version within 6·10⁻¹².

### Why the bonus and logistic regression cannot use it
H·w − b works because the cost is quadratic in w with *fixed* weights on each row.
- **Bonus:** each row's weight c (1 or 3) depends on the sign of its current residual, which changes as w changes. There is no single fixed H.
- **Logistic regression:** the gradient Xᵀ(p − y)/m contains p = σ(Xw), which is nonlinear in w.

Both must recompute the explicit gradient at each update.

### "Solving directly" vs gradient descent
- **Directly:** solve H·w = b (the normal equation, Practice 2-1). One linear solve, no learning rate, no iterations.
- **Gradient descent:** repeat w ← w − η(Hw − b).
- The brief requires our own gradient descent. We used the direct solve only as an *independent check* (`lstsq`).

### Why κ² matters for direct solves but not for gradient descent [Beyond the course]
- **κ(X)** is the condition number of the data matrix X itself.
- Forming H = XᵀX **squares** it: **κ(H) = κ(X)²**.
  - Our largest target design has κ(H) = **5,062**, so κ(X) = √5,062 ≈ **71**.
  - Phase 2's degree 3 has κ(H) = 1,713.
- **For a direct solve, κ is about accuracy.** A solve on H can lose about log₁₀(κ(H)) ≈ 3.7 of the ~16 decimal digits a computer keeps. That is why careful solvers like `lstsq` work on X itself (κ(X), not κ(X)²). With our numbers both are harmless: ~12 correct digits remain.
- **For gradient descent, κ is about speed.** The number of updates grows roughly with κ(H), because the flat directions shrink by only 1 − η·λ_min per update. Rounding does not accumulate into a wrong answer: we stop when the gradient is truly small and confirm against the exact solution.

  So a bad κ costs us *time*, not *correctness*, which is why centering (5.5) mattered for speed.

**Say it:** "XᵀX/n and Xᵀy/n are the Practice 2-1 sums. The gradient Xᵀ(Xw − y)/n equals Hw − b exactly, so we precompute them once per fit and each update costs p² instead of n·p. That was about 65× faster for degree 3. The bonus and logistic losses are not quadratic, so they keep the explicit gradient."

---

## 7. Phase 3: Bias–variance trade-off (Lecture 2)

### 7.1 Bias, variance, irreducible error
Expected test error = **bias²** (systematically wrong, too simple) + **variance** (too sensitive to which rows we trained on, too complex) + **irreducible error** (noise no model can predict).
- **Underfit:** high bias. Train and validation errors are both high and close together.
- **Overfit:** high variance. Train error is low, validation error clearly higher.

### 7.2 The complexity sweep (`In [22]`)
We followed Phase 2's order, then went **past** it to find the high-variance side:
- the rejected groups: hour × holiday, hour × weekday, hour × month, full hour × weathersit;
- then a degree-4 group: hour × workingday × yr × season.

**Exact duplicates [Beyond the course].** Some new columns are exact combinations of earlier ones. For example, hour × month contains hour × season, because a season is a set of months. We detected these with a QR factorization: it measures, for each column in order, the part the earlier columns cannot express. Duplicates were dropped (23 + 69 + 8 = 100 in total).

| step | features | train R² | validation R² | gap |
|---|---:|---:|---:|---:|
| degree 3 (end of Phase 2) | 242 | 0.9178 | 0.9225 | −0.0047 |
| + hour × holiday | 265 | 0.9204 | 0.9223 | −0.0018 |
| + hour × weekday | 380 | 0.9264 | 0.9249 | +0.0014 |
| + hour × month | 564 | 0.9317 | 0.9262 | +0.0054 |
| + full hour × weathersit | 602 | 0.9333 | 0.9272 | +0.0061 |
| + degree 4 | 818 | 0.9387 | **0.9274** | **+0.0112** |

- Training R² keeps rising: a larger linear model can never fit its own training data worse.
- Validation R² rises only a little, then flattens: **a plateau**. The gap opens from −0.005 to +0.011.
- Every fit converged; the largest took **11,761** updates.
- A fit that hits the cap would be flagged "not converged" and never called overfitted. None did.

### 7.3 Learning curves (`In [23]`) [Beyond the course in this form]
Train and validation RMSE as the training set grows (25%, 50%, 75%, 100%), with the validation set fixed:

| training rows | degree-3 design (train / validation) | 818-feature design (train / validation) |
|---:|---|---|
| 2,177 | 46.2 / 54.2 | 34.9 / **60.9** |
| 4,354 | 49.8 / 52.1 | 41.4 / 53.0 |
| 6,531 | 51.9 / 51.7 | 44.4 / 51.0 |
| 8,708 | 51.7 / 51.1 | 44.7 / 49.5 |

- **Degree 3:** the curves meet near 51. More data would barely help: **bias-limited**.
- **818 features:** a gap of 26 bikes at 25% of the data, still 4.8 with all of it, and closing: **variance-limited**. More data would help it.

### 7.4 The chronological split (`In [24]`) [From the brief]
- **How.** Sort the labeled days; the last 20% of *days* (91 days, 2012-08-05 to 2012-12-19, 2,183 hours) are held out, and the first 365 days (8,703 hours) are used for training.
  - The cut is between whole days, so no day has hours on both sides.
  - Everything is refitted on the earlier days: encoder, scaler, centering means, Phase 1, every sweep step.
- **Why it scores lower.** The random split puts hours of the *same day* in train and validation: same weather, same events, same growth level. That is **interpolation**. The chronological holdout is *after* all training data, in the busiest stretch of 2012. That is **extrapolation in time**.
  - Degree 3: R² **0.9094**, RMSE **65.5** (vs 0.9225 and 51.1 random).
  - In the hours it predicts busiest, it is **47.8** bikes too low on average (vs 11.8): it cannot know how much demand will have grown.
- **The test lies in between.** The test (day 20 of each month) is unseen days *inside* the training period, neither of the two settings.

### 7.5 The pre-registered rule and why it chose 564 features
- **Rule, written before the run.** Among converged fits, take the **fewest features** whose validation R² is within **0.002** of the best.
- **Applied.** The best is 0.9274 (818 features), so the cutoff is 0.9254. The smallest design above it is the one ending at hour × month: **564 features, 0.9262**. Degree 3 (0.9225) is outside.
  - Excluding the noisy degree-4 model changes nothing: the best would be 0.9272, and hour × month still qualifies.
- **Why "within 0.002, fewest features".** Differences of a few thousandths on one validation set are within noise. Among practically equal models, the simpler one is safer.

### 7.6 What our predictions got right and wrong
- **Right:** a plateau on the seeded split; the gap grows to about 0.01; everything converges; degree 3 bias-limited and degree 4 variance-limited; larger chronological shortfalls.
- **Wrong (5b):** we expected the extensions to hurt *more* on the chronological split. In fact hour × holiday, weekday, month and weathersit **helped more** there (+0.0149 R²: 0.9094 → 0.9243, vs +0.0047 on the random split). Only degree 4 hurt: **−0.0165** R² and **+11.2** bikes of peak shortfall.
- **Why degree 4 failed there.** The earlier days contain **zero 2012-autumn hours** and only 552 hours of 2012 summer (July and early August).
  - The 2012 × autumn columns are constant, so they are dropped (770 instead of 818 features).
  - The 2012 × summer terms are fitted on two months and then applied to late August and September.

**The lesson: combinations seen in training transfer; unseen ones do not.** hour × month works chronologically because August–December *of 2011* is in the training days, so each month's daily shape is known. Year × season × hour × working-day cells for 2012 autumn have never been seen, so their terms extrapolate.

**Handed to Phase 4:** the 564-feature design with lineage, scaler, centering means and converged weights; the diagnosis; and the noise candidate **temp × hum**.

**Say it:** "Training R² always rises with complexity; validation plateaued at about 0.927, and the gap opened only at degree 4. Learning curves show degree 3 is bias-limited and the 818-feature model variance-limited. The chronological split is lower (0.909) because it extrapolates into later, busier days. Our pre-registered rule, fewest features within 0.002 of the best, picked 564 features."

---

## 8. Phase 4: Regularization (Lecture 2)

### 8.1 The objective
One formula for all three methods, with the intercept never penalized:

  J(w) = ‖Xw − y‖²/(2n) + λ·[ρ·Σ|w_j| + (1 − ρ)·Σw_j²/2]

- **Ridge (L2), ρ = 0:** penalizes squared weights and shrinks all weights smoothly.
- **Lasso (L1), ρ = 1:** penalizes absolute weights, and can set weights to **exactly 0** (feature selection).
- **Elastic Net, 0 < ρ < 1:** a mix of the two.
- **λ** is the strength: λ = 0 means no penalty, a large λ means everything shrinks to 0.

### 8.2 Ridge = λ on the table's diagonal (Practice 2-6)
The Ridge gradient is H·w − b + λ·w (intercept excluded) = **(H + λI)·w − b**. Ridge simply **adds λ to the diagonal** of the table H, exactly as in Practice 2-6. Our standardized columns have H-diagonal entries of 1, so the chosen λ = **0.00045** is a 0.045% nudge.

### 8.3 Lasso and soft-thresholding [Beyond the course: the algorithm]
|w| has no slope at 0, so plain gradient descent cannot make a weight exactly 0. Our update has two steps:
1. **A normal gradient step** on the squared-error part (plus any L2 part): step = w − η·gradient.
2. **Soft-thresholding:** pull every penalized weight toward 0 by **η·λ·ρ**. If it would cross 0, set it to **exactly 0**.

   For example, with a threshold of 0.0377: a weight of 0.05 becomes 0.0123, a weight of −0.02 becomes 0, and a weight of 0.5 becomes 0.4623.

For our selected Lasso, η = 1/L = 0.223 and λ = 0.169, so each update pulls weights toward 0 by **0.0377**. A weight survives only if the data's gradient keeps pushing it back out.

**Learning rate.**
- Ridge keeps **1.9/L**, with L = λ_max(H) + λ(1 − ρ).
- Lasso and Elastic Net use **1/L**, because soft-thresholding steps of at most 1/L are guaranteed never to increase the objective.
- Every fit recorded that the objective never increased.

**Stopping.** The largest |w − w_next|/η < 10⁻⁴: the size of the move still being made. Without L1 this is exactly the gradient.

**Checks (tests).** Ridge matches its closed-form solution (H + λI)⁻¹b. Lasso satisfies the optimality conditions:
- the gradient = −λ·sign(w) on nonzero weights;
- |gradient| ≤ λ on zeroed weights.

### 8.4 The λ path and the grid extension [Beyond the course]
- **Path.** For each method, 30 log-spaced λ values from strong to weak, each fit starting from the previous fit's weights (warm start). That makes the whole path cheap.
  - Lasso and Elastic Net start at λ_max, the smallest λ at which every weight is exactly 0. For Lasso that is 70.6.
  - Ridge starts at 1,000.
- **Grid extension.** In the first official run all four L1-type paths reached their best validation R² at their **last** (smallest) λ: the grid stopped too early. Elastic Net 0.1 ended at λ = 0.07, still 0.012 below its plateau.
  - We extended every L1-type path down to 10⁻⁵ at the same spacing and re-ran.
  - The first 30 points reproduced exactly; Ridge, Lasso and every verdict were unchanged; only Elastic Net's choice moved (ρ 0.5 → 0.1).
  - This is recorded openly in the notebook.

### 8.5 Selection rules
- The brief fixes one validation set, so there is **no k-fold** cross-validation.
- **Ridge:** the highest validation R².
- **Lasso and Elastic Net:** the **largest λ within 0.002** of that path's best. Among near-ties, prefer the sparser, more stable model.
- **Elastic Net ratio:** the ratio whose selected fit scores best.

### 8.6 Results (`In [27]`, `In [28]`): "no variance to buy, only bias paid"
| method | λ | ρ | nonzero (of 564) | validation R² | RMSE | top-decile shortfall |
|---|---:|---:|---:|---:|---:|---:|
| unpenalized (Phase 3) | 0 | – | 564 | 0.926228 | 49.9 | 53.8 |
| **Ridge** | 0.00045 | 0 | 564 | **0.926237** | 49.9 | 54.0 |
| Lasso | 0.169 | 1 | **466** | 0.9248 | 50.4 | 56.9 |
| Elastic Net | 0.0144 | 0.1 | 564 | 0.9246 | 50.5 | 57.6 |

- Phase 3 showed almost **no variance** left at 564 features (gap 0.005, bias-limited learning curve).
- A penalty trades a little bias for less variance. With no variance to buy, it could only **add bias**: the more the penalty bit, the lower R² went, and the busiest hours got worse (shrinkage pulls predictions toward the average).
- Ridge won only by choosing an almost-zero λ.

### 8.7 Every column verdict (`In [29]`)
**How each column was judged.**
- **Grouped ablation [Beyond the course].** Remove the column **and every term built from it**, rebuild the design, refit with Ridge's settings, and measure ΔR². The design is rebuilt rather than columns deleted, so terms that the removed column had made redundant come back.
- **Add-back,** for columns not in the model: add the column and refit.
- **Alone R²:** a model with only that column, in its full encoding (hr as 23 dummies). Does it carry demand information at all?
- **Lasso survival:** how many of its terms Lasso keeps.

**Rule (pre-registered).**
- **useful** if removing it costs ≥ 0.002 R²;
- else **redundant** if alone R² ≥ 0.01 (its information is carried by other columns);
- else **uninformative**.

| column | evidence | verdict |
|---|---|---|
| **hr** | ablation **−0.568** R², +240 bikes peak shortfall; alone R² 0.491; Lasso keeps 443/537 | useful |
| **yr** | ablation **−0.110** R², +63 bikes; alone 0.067; 45/48 kept | useful |
| **weathersit** | ablation −0.0175; alone 0.016; 9/10 kept | useful |
| **workingday** | ablation **−0.0162**, +12.5 bikes; but alone only **0.0001** (its marginal correlation with demand is near 0, yet its interaction with hour is the biggest gain of the project) | useful |
| **temp** | ablation −0.0143; alone 0.165; 42/49 kept | useful |
| **mnth** | ablation −0.0055; alone 0.070; 149/195 kept | useful |
| **weekday** | ablation −0.0052; alone −0.001 (on its own slightly worse than predicting the average) | useful |
| **season** | ablation **0.0000**: hour × month carries season's information exactly; main effect an exact combination of month; alone 0.053 | redundant |
| **hum** | ablation −0.0006; alone 0.097 | redundant |
| **windspeed** | ablation −0.0006; alone **0.0146**, just above the 0.01 threshold | redundant (we expected "uninformative") |
| **atemp** | add-back +0.0003; alone 0.162 (a near-copy of temp) | redundant |
| **instant** | add-back +0.0003; alone 0.094 (rises with time, like yr) | redundant |
| **dteday** | add-back of elapsed days + day of month +0.0002; alone 0.094 | redundant |
| **holiday** | ablation −0.0000; alone −0.0003; **Lasso keeps 18 of 24 terms** | uninformative |

**The holiday lesson.**
- Lasso kept 18 of holiday's 24 terms, yet removing all of them costs nothing.
- At a small λ, Lasso keeps terms that slightly improve the *training* fit, even if they do not generalize.
- **A surviving Lasso coefficient is not evidence of usefulness; a zero is evidence of non-usefulness.** The proof of usefulness is what happens on validation when you remove it.

**Handed to Phase 5:** the **466** Lasso survivors (from 11 original columns) with lineage and the fitted scaler. Ridge is the recommended regression model.

**Say it:** "Ridge adds λ to the diagonal of XᵀX/n. Lasso adds a soft-threshold after each gradient step, which sets small weights exactly to zero. Since Phase 3 showed no variance left, regularization could only add bias: Ridge won with an almost-zero λ, and Lasso kept 466 of 564 terms at slightly lower R². Columns were judged by removing them with every term built from them; holiday shows that a surviving Lasso weight doesn't prove usefulness."

---

## 9. Phase 5: Logistic regression (Lecture 3 / Practice 3)

### 9.1 Discretizing cnt: what is "high demand"?
We turn the number `cnt` into a yes/no label. Three options:

| definition | positive | 2011 | 2012 | hours of the day never positive | hourly positive rate |
|---|---:|---:|---:|---:|---|
| A: one global cutoff (75th percentile of all hours) | 25.0% | 15.6% | 34.3% | **8 of 24** | 0% – 75% |
| B: per hour × working day | 24.5% | **5.6%** | **43.1%** | 0 | 21% – 25% |
| **B + year: per hour × working day × year** (chosen) | 24.4% | 24.3% | 24.6% | 0 | 21% – 25% |

- **A teaches the clock.** Night hours are *never* above a global cutoff and commute peaks often are. A classifier could score well just by reading the hour. Hours with only one class also cause *separation* (logistic weights grow without limit).
- **B teaches "is it 2012?".** Demand grew, so most 2012 hours exceed 2011-based cutoffs.
- **B + year.** Each hour × day-type × year group (96 groups) has its own cutoff: the 75th percentile of its *training* hours.
  - Working-day 08:00 in 2012: cutoff **692** bikes. Weekend 08:00 in 2012: **170**. Working-day 03:00 in 2011: **6**.
  - An hour is "high" if it exceeds its group's cutoff: **unusually busy for its slot**.
  - **2,127 of 8,708** training hours are positive (24.4%), 25.2% in validation.
  - Some groups sit below 25% (min 19.1%) because night counts are small integers with many ties, and a tie is not "above".

### 9.2 Sigmoid, log odds, maximum likelihood (Lecture 3)
- **Sigmoid:** σ(z) = 1/(1 + e^(−z)) turns any number into a probability between 0 and 1.
  - Our intercept is −2.754, so an "average" hour has p = σ(−2.754) = **0.06**.
- **Log odds (logit):** z = log(p/(1 − p)) = Xw. Logistic regression is a *linear model for the log odds*.
- **Maximum likelihood:** choose w to make the observed labels most probable. Multiplying the probabilities and taking −log gives the **log-loss** (cross-entropy):

  J(w) = −(1/m)·Σ[y·log p + (1 − y)·log(1 − p)]

  For a truly high hour (y = 1), predicting p = 0.9 costs 0.105, p = 0.5 costs 0.693, and p = 0.1 costs 2.303. Confident mistakes are punished heavily.
- **Gradient:** Xᵀ(p − y)/m. It has the same shape as linear regression's, with p in place of Xw.

### 9.3 The curvature/4 step [Beyond the course]
- The log-loss curvature is Xᵀ·diag(p(1 − p))·X/m.
- Since p(1 − p) is at most 1/4 (at p = 0.5), the curvature is at most **L = λ_max(XᵀX/m)/4 = 1.0424**.
- Any step below 2/L is safe; we kept our rule **1.9/L = 1.823**.
- Converged in **1,229** updates; the loss never increased; final training log-loss **0.333**.
- No precomputed table here: p changes every step.

### 9.4 Features
Exactly Phase 4's **466** Lasso survivors, with the same fitted scaler. Neither `cnt` nor the cutoffs enter X.

### 9.5 The 0.37 cutoff and why it is below 0.5 (`In [33]`)
We searched cutoffs 0.05–0.95 for the best validation **F-measure (F1)**, the harmonic mean of precision and recall.

| cutoff | TP | FP | FN | TN | accuracy | precision | recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.37** | 388 | 230 | 161 | 1,399 | 0.820 | 0.628 | **0.707** | **0.665** |
| 0.50 | 309 | 137 | 240 | 1,492 | **0.827** | 0.693 | 0.563 | 0.621 |

- Only 25% of hours are positive, so the model's probabilities for real busy hours are often below 0.5. Lowering the cutoff catches more of them (recall 0.563 → 0.707) at some cost in precision.
- **ROC-AUC [Beyond the course; required by the brief] = 0.871.** The probability that a random busy hour gets a higher score than a random normal hour. It does not depend on the cutoff.
- Tuning and scoring the cutoff on the same validation set is slightly optimistic.

### 9.6 Why accuracy misleads with 25% positives
- Predicting "not high" for every hour already scores **0.748** accuracy.
- Going from 0.37 to 0.50, accuracy *rose* (0.820 → 0.827) while F1 *fell* (0.665 → 0.621). The higher cutoff turned **79** caught busy hours into misses (388 → 309 true positives) and avoided **93** false alarms (230 → 137). Accuracy counts both errors the same.
- But a missed busy hour is the operator's costly mistake. With imbalanced classes, accuracy rewards ignoring the rare class.

### 9.7 Odds ratios (Practice 3)
- e^β is the factor by which the **odds** of "high" change when a feature rises by one standard deviation, others fixed.
- **Temperature:** β = 1.295, so **×3.65** per SD. One SD of temp is 0.1906 normalized = about 9.0 °C. Per +4.7 °C (0.1 normalized) the odds roughly **double (×1.97)**. (Temperature also appears in interaction terms, so "others fixed" is a simplification.)
- **Months:** September ×2.87, October ×2.62, May–August ×2.3–2.5, February ×0.50.
- **Weekdays:** Monday–Wednesday ×0.49–0.60 against Sunday.
- **Hour dummies matter little** (only 20:00 in the top 15, ×0.63): the label already removed each hour's level.

### 9.8 The seasonal leak and the 0.809 / 0.747 diagnostics (`In [36]`)
- The label removes hour, day type and year, but **not the season**. Only **0.4%** of January training hours are "high", against **48.8%** of June.
- **Calendar-only ranking:** score each hour by its month × weekday's training positive rate, with no model at all. That alone gets ROC-AUC **0.809**.
- **Within a month** (where the calendar cannot help), the classifier ranks busy hours at **0.747** on average; within a season **0.641–0.862**. That is its genuine skill from weather and temperature.
- Why we kept the label: in a side check run on the training rows (not printed in the notebook), adding season to the groups would make 193 of 384 groups smaller than 20 rows (noisy cutoffs), and March would still be 49.9% positive against January's 6.0%. We report the leak openly instead.

**Say it:** "High demand means above the training 75th percentile of the same hour, day type and year, so the label is not just the clock or 2012. Logistic regression is a linear model of the log odds, fitted by maximum likelihood, i.e. minimizing log-loss with gradient Xᵀ(p − y)/m. The best-F1 cutoff is 0.37 because only a quarter of hours are positive; accuracy is misleading there since always-0 already scores 0.748. ROC-AUC 0.871, but 0.809 of that comes from the calendar; within a month it's 0.747."

---

## 10. Final submission (`In [39]`)
- **Model.** Phase 4's **Ridge** (λ = 0.00045) on the 564-feature design: the best regression model on validation. The brief wants the submission from the regression pipeline, not the classifier.
- **Refit on all 10,886 labeled rows.** Every choice (features, groups, λ, procedure) was frozen on the seeded split. The validation rows have done their job of choosing, so using them as extra training data changes no decision; it just gives the same model 25% more rows. The brief allows this if justified.
  - Encoder, scaler, centering means and weights are refitted on the combined rows (the same 92 duplicate columns are dropped).
  - Same Ridge path: converged in **2,053** updates; training R² 0.9317, RMSE 47.3.
- **Clipping.** Rentals cannot be negative. **35** predictions below 0 were set to 0. For a nonnegative target, clipping can only reduce the squared error.
- **Checks.** 574 rows, in the sample file's `instant` order, all finite, ≥ 0. Mean 191.8, max 870.9. File: `outputs/submission/sample_submission.csv`.

**Say it:** "All choices were frozen on the seeded split; then we refitted the Ridge model, scaler included, on all labeled rows and clipped negatives to 0."

---

## 11. Honest limitations
1. **Seasonal label leakage.** The Phase 5 label still carries the season; much of the 0.871 ROC-AUC is reproducible from the calendar alone (0.809).
2. **Growth extrapolation.** On later, busier days the models under-predict their own busiest hours by about 37–48 bikes. They cannot predict growth beyond the training period.
3. **One validation set for tuning and scoring.** λ, the Phase 3 target and the 0.37 cutoff were chosen and scored on the same seeded validation set, so those scores are slightly optimistic.
4. **Phase 4 grid revision.** The L1 paths were extended after the first official run; only the Elastic Net choice changed. It is recorded.
5. **Reconstructed expectations.** Phase 1's and the bonus's Expectation cells were written after their runs, from earlier design notes, and are labelled as such.
6. **Negative and noisy night predictions.** 120 validation hours are still predicted below zero at degree 3, and night recall is the weakest (0.54–0.59).

---

## 12. Glossary
- **RSS:** residual sum of squares, Σ(actual − predicted)².
- **Cost function J:** what gradient descent minimizes. Here RSS/(2n), or the log-loss.
- **Gradient:** the direction of steepest increase of J. We step the opposite way.
- **Learning rate η:** the step size of gradient descent.
- **Batch gradient descent:** each update uses all training rows.
- **λ_max:** the largest eigenvalue of XᵀX/n, the steepest curvature of the cost bowl. [Beyond the course]
- **Condition number κ:** λ_max/λ_min, how stretched the bowl is. A large κ means slow gradient descent. [Beyond the course]
- **One-hot / dummy variables:** 0/1 columns for the levels of a category, minus one reference level.
- **Standardization:** (x − mean)/SD, fitted on training rows.
- **Interaction term:** a product of columns, e.g. hr=8 × workingday, so one feature's effect can depend on another.
- **Degree (here):** how many columns are multiplied in a term.
- **Centering:** subtracting the training mean before multiplying. [Beyond the course]
- **Warm start:** starting gradient descent from a previous model's weights.
- **Bias:** systematic error from a too-simple model.
- **Variance:** sensitivity to the particular training rows, from a too-complex model.
- **Irreducible error:** noise no model can predict.
- **Learning curve:** error against training-set size. [Beyond the course]
- **Chronological split:** train on earlier dates, validate on later ones.
- **Ridge / L2:** penalty λΣw²/2, which shrinks all weights.
- **Lasso / L1:** penalty λΣ|w|, which can set weights to exactly 0.
- **Elastic Net:** a mix of L1 and L2, controlled by ρ (the l1_ratio).
- **Soft-thresholding:** moving a weight toward 0 by a fixed amount and stopping at 0. [Beyond the course]
- **λ path:** fitting along decreasing λ values with warm starts. [Beyond the course]
- **Ablation:** removing a column and everything built from it, then refitting. [Beyond the course]
- **Sigmoid σ:** 1/(1 + e^(−z)).
- **Log odds:** log(p/(1 − p)).
- **Maximum likelihood:** choosing weights that make the observed labels most probable.
- **Log-loss:** −mean[y·log p + (1 − y)·log(1 − p)].
- **Precision:** of the hours flagged high, the share that are high.
- **Recall:** of the truly high hours, the share flagged.
- **F-measure (F1):** 2PR/(P + R).
- **ROC-AUC:** the probability that a random positive outranks a random negative. [Beyond the course]
- **Odds ratio:** e^β, the multiplicative change of the odds per unit (here per SD) of a feature.
- **Expectile:** the minimizer of an asymmetric squared loss, an "asymmetric mean". [Beyond the course]
- **Top-decile shortfall:** mean bikes missing in hours above the training 90th percentile of demand. [Beyond the course; our operator check]

---

## 13. The 25 questions a TA is most likely to ask

1. **How is your seed computed?**
   SHA-256 of the sorted, underscore-joined IDs `16001061_16001322_16008386`, as an integer, mod 100,000 = **36954**. The brief's example (41698) is checked first. → `In [1]`

2. **What did you fit on validation or test?**
   Nothing. Encoders, scalers, centering means, interaction choices and label cutoffs all come from training rows; validation is only for choosing and scoring; test is only predicted. → `src/features.py`, `In [39]`

3. **Why is hour one-hot and not a number?**
   A number forces a straight line across the day; demand has two weekday peaks and a weekend hump, so each hour needs its own level. → markdown above `In [2]`

4. **Why did you drop workingday and atemp in Phase 1?**
   workingday = (Mon–Fri) − holiday exactly, so it is collinear with the weekday dummies + holiday. atemp has correlation 0.984 with temp. → `In [2]`

5. **Write the cost and the gradient.**
   J = ‖Xw − y‖²/(2n); gradient Xᵀ(Xw − y)/n. → `src/phase1_gradient_descent.py`

6. **Why must the learning rate be below 2/λ_max?**
   Each update multiplies the error along the steepest direction by 1 − ηλ_max; that shrinks only if η < 2/λ_max = 1.0053. 2.1/λ_max diverged after 84 updates. → `In [4]`

7. **How do you know it converged?**
   max|gradient| < 10⁻⁴, and the predictions are within 0.0074 bikes of the exact least-squares solution. A flat loss curve alone is not proof: the 0.005 run looked flat but its gradient was 0.0104. → `In [4]`, `In [5]`

8. **What is the precomputed table?**
   H = XᵀX/n and b = Xᵀy/n (the Practice 2-1 sums). The gradient equals Hw − b exactly; each update costs p² instead of n·p; every fit asserts both forms agree. → `src/phase1_gradient_descent.py`

9. **What is the bonus's loss and gradient?**
   ½·c·r² with c = 3 for under-predictions; gradient Xᵀ(c ⊙ r)/n. The step had to shrink 3× (0.955 diverged, 0.318 converged). → bonus section, `In [8]`

10. **Why did R² drop in the bonus, and is that bad?**
    R² scores MSE, which we no longer minimize. The asymmetric cost fell 16%, which is the actual goal. → `In [9]`

11. **Why hour × workingday first?**
    44% of Phase 1's residual variance was between hour × working-day groups (e.g. weekend 08:00 over-predicted by 252). Alone it gave +0.149 R². → `In [13]`, `In [16]`

12. **What does degree mean here?**
    How many columns are multiplied. Powers of 0/1 dummies do nothing, so degree = interaction order. → markdown above `In [15]`

13. **What is centering and why?**
    Subtract each non-hour factor's training mean before multiplying. Same predictions possible, but the products no longer copy the hour dummies; the condition number fell 21,616 → 1,713. → `In [15]`

14. **How did Phase 2 use Phase 1's weights?**
    Warm start: Phase 1 weights in their own columns, new weights 0; we assert identical starting predictions. → `In [16]`

15. **Validation R² above training R², isn't that suspicious?**
    No. RMSE is nearly equal (51.7 vs 51.1); the gap is within the 0.909–0.926 range of random subsets. → `In [18]`

16. **How did you tell bias from variance?**
    Train vs validation as complexity grows, plus learning curves. Degree 3 has a gap of about 0 and is flat (bias-limited); 818 features has a 26-bike gap at 25% data, closing (variance-limited). → `In [22]`, `In [23]`

17. **What was your Phase 3 rule, and why 564 features?**
    The fewest features within 0.002 of the best validation R² (0.9274), so hour × month at 0.9262. Fixed before the run. → `In [25]`

18. **Why is the chronological score lower, and which do you trust?**
    It extrapolates to later, busier days (0.9094 vs 0.9225). Trust it for forecasting; the test (day 20, inside the period) lies between the two. → `In [24]`

19. **Why did degree 4 fail chronologically?**
    No 2012-autumn hours in the earlier days, so those terms cannot be learned. Combinations seen in training transfer; unseen ones don't. → `In [24]`

20. **How does your Lasso create exact zeros?**
    Soft-thresholding: after each gradient step, move every penalized weight toward 0 by η·λ, and set it to 0 if it crosses. → `src/phase4_regularization.py`

21. **Why did regularization not improve R²?**
    Phase 3 showed no variance left, so the penalty only added bias. Ridge won with λ = 0.00045 (equal to unpenalized); Lasso kept 466 terms at R² 0.9248. → `In [28]`

22. **How did you decide a column is useful, redundant or uninformative?**
    Remove it with all derived terms and refit (≥ 0.002 R² lost = useful); else alone R² ≥ 0.01 = redundant; else uninformative; plus Lasso survival. Holiday shows a surviving Lasso weight is not proof. → `In [29]`

23. **Why this label, and not one global cutoff?**
    A global cutoff learns the clock (8 hours never high); without year it learns "is it 2012" (5.6% vs 43.1%). Per hour × day type × year, every slot has about 25% high hours. → `In [31]`

24. **Why is the cutoff 0.37 and not 0.5?**
    Best validation F1 (0.665 vs 0.621). With 25% positives, accuracy is misleading: always-0 scores 0.748, and 0.5 has higher accuracy but misses 79 more busy hours. → `In [33]`

25. **Is ROC-AUC 0.871 the model's real skill?**
    Partly. The label still carries season: a calendar-only ranking gets 0.809. The classifier's own skill is the within-month ranking, 0.747. → `In [36]`
