# Evaluation defense notes

Short answers to the questions a TA is most likely to ask, phase by phase. Cell references (`In [n]`) are the execution numbers in the saved `notebooks/final_submission.ipynb`. Numbers are from that run.

---

## Setup and data

**How did you get the seed, and what does it control?**
Sort our three student IDs as strings (`16001061`, `16001322`, `16008386`), join them with `_`, take SHA-256, read it as an integer and take it mod 100,000. That gives **36954**. The code first checks the brief's example (41698). The seed goes into the one `train_test_split(..., test_size=0.20, random_state=36954)`: 8,708 training and 2,178 validation rows. *Where:* `In [1]`, `src/data.py`.

**What did you fit on validation or test data?**
Nothing. Every encoder level, scaler mean/SD, centering mean, interaction choice and label cutoff is fitted on the training portion. Validation is only for choosing and scoring. `test.csv` is only predicted, at the very end. *Where:* `src/features.py` (`fit` uses `train_df` only), `In [39]`.

**Why not use `dteday`, `workingday`, `season`, `atemp`, `instant` directly?**
The code checks these facts:
- `dteday`'s year, month and weekday equal `yr`, `mnth` and `weekday` (0 mismatches).
- `workingday` = (Mon–Fri) − holiday exactly, so it is a linear combination of the weekday dummies and holiday.
- `season` is an exact function of month in this file.
- `atemp` has correlation 0.984 with `temp`.
- `instant` is a row number.

*Where:* `In [2]`.

---

## Phase 1 — Gradient descent

**What exactly does your gradient descent compute?**
- Cost function: J(w) = ‖Xw − y‖²/(2n).
- Gradient: Xᵀ(Xw − y)/n.
- Update: w ← w − η·gradient.

This is batch gradient descent: every update uses all training rows. *Where:* `src/phase1_gradient_descent.py`, `loss_and_gradient`.

**Why must the learning rate be below 2/λmax?**
- The cost is a bowl (quadratic). Its steepest curvature is λmax, the largest eigenvalue of XᵀX/n.
- Along that direction, each update multiplies the error by (1 − η·λmax). That shrinks the error only if |1 − η·λmax| < 1, i.e. η < 2/λmax.
- Here λmax = 1.9894, so the limit is 1.0053.
- We saw it: 2.1/λmax diverged after 84 updates, 1.9/λmax converged in 267, and 0.005/λmax was too slow (it hit the 50,000 cap).

*Where:* `In [4]`, plot below it.

**What is the "precomputed table" in the loop?**
- H = XᵀX/n and b = Xᵀy/n are the sums of products of features with each other and with y: the same table of sums as in Practice 2-1.
- The cost is quadratic, so the gradient equals H·w − b. We compute H and b once per fit; each update then costs p² operations instead of n·p.
- The readable form Xᵀ(Xw − y)/n is kept, and every fit asserts at its start and end that both forms give the same gradient and loss.

*Where:* `_objective` and `_assert_matches_readable_form` in `src/phase1_gradient_descent.py`.

**How do you know it converged rather than stalled?**
- The stopping rule is max |gradient| < 10⁻⁴. A flat loss curve is not proof: the 0.005/λmax run looked flat but still had max|gradient| 0.0104.
- Independent check: the predictions are within 0.0074 bikes of the exact least-squares solution.

*Where:* `In [4]`, `In [5]`.

**Why one-hot hours instead of hour as a number? Why scale?**
- Demand has two weekday peaks and a weekend midday hump. One number 0–23 can only draw a straight line across the day; 23 dummies give each hour its own level.
- Standardizing puts all columns on the same scale, so one learning rate suits every weight.

*Where:* markdown above `In [2]`.

---

## Phase 1 bonus — asymmetric cost

**Which loss, and what is its gradient?**
- Loss: ℓ(r) = ½·c·r², with r = prediction − actual and c = 3 when r < 0 (under-prediction), else 1.
- Its derivative c·r is continuous at 0 (both sides are 0), so the gradient is Xᵀ(c ⊙ r)/n.
- That equals the MSE gradient plus 2·Xᵀmin(r, 0)/n: an extra push only from under-predicted rows.

*Where:* markdown at the top of the bonus section, `src/bonus_asymmetric.py`.

**Why a smaller learning rate?**
- The curvature can be up to 3× larger, so the safe step shrinks by 3.
- Phase 1's step (0.955) diverged within 12 updates; 0.955/3 = 0.318 converged in 388.

*Where:* `In [8]`.

**Why not the pinball (absolute) loss?**
It has no gradient at r = 0, and its slope never shrinks toward zero, so fixed-step gradient descent with our gradient-based stopping rule cannot converge on it. The squared version changes only the asymmetry compared with MSE.

**What happened?**
- Asymmetric cost −16.2%, R² 0.682 → 0.606.
- Most of the lift was spread over all hours, because Phase 1 underfits. On Phase 2's features the buffer was cheaper (−0.016 R²) and better aimed.

*Where:* `In [9]`–`In [11]`, `In [19]`–`In [20]`.

---

## Phase 2 — Polynomial regression

**Why hour × workingday first?**
44% of Phase 1's residual variance lay between hour × working-day groups. Phase 1 over-predicted weekend 08:00 by about 250 bikes and under-predicted the weekend midday. One group then gave +0.149 R² (0.682 → 0.831). *Where:* `In [13]` (diagnosis and plot), `In [16]`.

**What does "degree" mean here, and why not expand everything?**
- Degree = interaction order. Degree 2 = products of two columns (hour × workingday, …); degree 3 = three (hour × workingday × temp).
- Blind expansion creates useless terms: two dummies of one column multiply to 0, and a squared 0/1 column is the same column.
- So we used explicit term lists chosen from the residuals.

*Where:* `group_terms` in `src/phase2_polynomial.py`, markdown above `In [15]`.

**What is "centering", and why?**
- Before multiplying, every factor except the hour dummy is shifted by its training mean, e.g. `hr=8 × (workingday − mean)`.
- The model can express exactly the same predictions, but the products are much less correlated with the hour dummies.
- The condition number falls from 21,616 to 1,713, so gradient descent needs far fewer updates.

*Where:* `In [15]`.

**How did you use Phase 1's weights?**
- Warm start: the Phase 1 weights go into their own columns, and every new column starts at 0.
- Before training we assert that this starting model predicts exactly what Phase 1 predicts.

*Where:* `fit_and_score` in `src/phase2_polynomial.py`, `In [16]`.

**Validation R² is above training R² — isn't that suspicious?**
No. RMSE is almost equal (51.7 vs 51.1). The validation target simply varies a bit more, and the gap is within the noise of random subsets (0.909–0.926). *Where:* `In [18]`.

---

## Phase 3 — Bias–variance trade-off

**How did you tell bias from variance?**
- *Bias:* train and validation errors are high and close together.
- *Variance:* train keeps improving while validation stalls, and the gap opens.
- Results:
  - Degree 3 had a gap of about 0 and a flat learning curve, so it is bias-limited.
  - The 818-feature design had a 26-bike gap at 25% of the data, shrinking to 4.8 with all of it, so it is variance-limited.

*Where:* `In [22]`, `In [23]`.

**What was the rule for the target, and why did it move to 564 features?**
- The rule, fixed before the run: among converged fits, the fewest features whose validation R² is within 0.002 of the best.
- Best = 0.9274 (818 features). The smallest design within 0.002 was the one ending at hour × month: 564 features, 0.9262. Degree 3 (0.9225) was outside.

*Where:* `In [25]`.

**Why does the chronological split score lower, and which estimate do you trust?**
- The random split shares days with training: interpolation. The chronological split predicts later, busier days: extrapolation.
- R² for degree 3: 0.9225 random vs 0.9094 chronological.
- For forecasting new periods, trust the chronological one. The test (day 20 of each month) lies in between.

*Where:* `In [24]`.

**Why did degree 4 fail on the chronological split?**
- The earlier days contain no 2012-autumn hours, so those terms cannot be learned: their columns are constant and are dropped.
- The 2012-summer terms are learned from July and early August only.
- Result: −0.0165 R².

*Where:* `In [24]`.

**What does "not converged — flagged" mean?**
A fit that hits the update cap is never called over-fitted; we show its exact least-squares score next to it instead. In the official run every fit converged. *Where:* plot in `In [22]`.

---

## Phase 4 — Regularization

**Write the objective.**
- J(w) = ‖Xw − y‖²/(2n) + λ·[ρ·Σ|wⱼ| + (1 − ρ)·Σwⱼ²/2], with the intercept not penalized.
- Ridge: ρ = 0. Lasso: ρ = 1. Elastic Net: in between.

*Where:* markdown above `In [26]`.

**How does your Lasso produce exact zeros (soft-thresholding)?**
- After a normal gradient step on the squared-error part, every weight is pulled toward 0 by η·λ·ρ. If it would cross 0, it is set to exactly 0.
- Ridge only shrinks (it adds λ to the diagonal of H), so it never reaches exactly 0.

*Where:* `PenalizedGD.fit` in `src/phase4_regularization.py`.

**How did you search λ, and why did you change the grid?**
- One log-spaced path per method, from strong to weak, each fit starting from the previous one.
- In the first run all L1-type paths peaked at their last grid point. We extended them down to 10⁻⁵ at the same spacing: the first 30 points reproduced exactly, and only Elastic Net's choice changed. The change is recorded in the notebook.

*Where:* `In [26]`, `In [28]`.

**Who won and why didn't regularization help?**
- Ridge (λ = 0.00045) matched the unpenalized R² (0.9262); Lasso (0.9248) and Elastic Net (0.9246) were slightly worse.
- Phase 3 showed almost no variance to remove, so the penalty only added bias. It also made the busiest-hour shortfall worse (53.8 → 56.9 bikes under Lasso).

*Where:* `In [28]`.

**How did you decide a column is useful, redundant or uninformative?**
- Remove the column and every term built from it, refit, and look at ΔR². ≤ −0.002 means useful.
- Otherwise: if the column alone (in its full encoding) reaches R² ≥ 0.01, it is redundant (others carry its information); if not, uninformative.
- Lasso survival is reported too. A surviving coefficient is not proof of usefulness (holiday kept 18 of 24 terms, yet removing them costs nothing); a zero would be evidence against.

*Where:* `In [29]`.

---

## Phase 5 — Logistic regression

**Why this label and not "cnt above one global cutoff"?**
- A global cutoff mostly learns the clock. 8 of 24 hours are never "high", so those hours contain only one class.
- Without year, it learns "is it 2012" (5.6% vs 43.1% positive).
- Per hour × working day × year, every kind of hour has about 25% "high": unusually busy *for its slot*. Cutoffs come from training only.

*Where:* `In [31]`.

**What is the model and its gradient?**
- p = σ(Xw) with σ(z) = 1/(1 + e^(−z)).
- Cost: log-loss −mean[y·log p + (1 − y)·log(1 − p)].
- Gradient: Xᵀ(p − y)/m.
- The curvature is at most λmax(XᵀX/m)/4 (because p(1 − p) ≤ 1/4), so any step below 2/L is safe; we used 1.9/L = 1.823. It converged in 1,229 updates.

*Where:* `In [32]`, `src/phase5_logistic.py`.

**Why a cutoff of 0.37 and not 0.5?**
- We chose the cutoff with the best validation F1: at 0.37, F1 0.665 (recall 0.707, precision 0.628). At 0.5, F1 0.621 (recall 0.563).
- Accuracy is higher at 0.5 (0.827 vs 0.820), but with only 25% positives, predicting "not high" is usually right: always-0 already scores 0.748.
- Accuracy therefore rewards missing busy hours, the operator's costly mistake.
- Tuning and scoring on the same validation set is slightly optimistic.

*Where:* `In [33]` (table, confusion matrices, plot).

**How do you read the coefficients?**
e^β is the factor by which the odds of "high" change when that (standardized) column rises by one standard deviation. Temperature: ×3.65 per standard deviation; the September dummy: ×2.87 per standard deviation (months are the next-largest effects, i.e. the season inside the label). *Where:* `In [34]`.

**Is ROC-AUC 0.871 really the model's skill?**
- Partly. The label still carries the season: 0.4% of January hours are "high" against 48.8% of June hours.
- A calendar-only ranking (month × weekday) already gets 0.809.
- The classifier's own skill is the within-month ranking: 0.747 on average.

*Where:* `In [36]`.

---

## Final model and submission

**Which model produced `sample_submission.csv`, and why refit?**
- Phase 4's Ridge (λ = 0.00045) on the 564-feature design.
- All choices were frozen on the seeded split. We then refitted encoder, scaler, centering means and weights on all 10,886 labeled rows with the same Ridge path; the final fit converged in 2,053 updates.
- Negative predictions were clipped to 0 (35 rows). 574 rows, in the sample's `instant` order.

*Where:* `In [39]`; file `outputs/submission/sample_submission.csv`.

**Which split / number should we believe for unseen data?**
For future periods: the chronological R² (0.9224 for the target). For the test days (inside the training period): between the chronological and the seeded estimate (0.9262).
