# Predicted sample requirement, Salsa 2.0 at n=30, h=3

**No new training was performed for this report.** Every measured value is
extracted programmatically from previously completed runs.

Model under test: Salsa2-GatedUT, 4,131,200 parameters, T_e=2, T_d=2, Representation R, q=251, sigma=3, seed 0

## 1. Measured Salsa 2.0 results

| experiment | secret bits | samples | breakpoint | best acc_tau | best exact | min loss | samples/s | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n=12, h=2 | 6.04 | 100,032 | 40,448 | 0.349 | 0.0122 | 1.7297 | 55.57 | 65.0 min |
| n=20, h=2 | 7.57 | 200,000 | 161,792 | 0.234 | 0.0068 | 1.8233 | 43.84 | 127.3 min |
| n=30, h=3 | 11.99 | 100,032 | **none reached** | 0.208 | 0.0078 | 1.8426 | 29.39 | 69.9 min |

## 2. Extrapolation models

Each model is fitted through the two observed breakpoints. Two points fix a
two-parameter model exactly, so **every one of these fits is perfect and none
of them carries evidence about which is right**.

| model | form | fitted | n=30 estimate | accounts for h? |
|---|---|---|---:|:---:|
| **A** linear in n | `S(n) = S2 + slope * (n - n2)` | slope_samples_per_dimension=15168.0 | 313,000 | **no** |
| **B** exponential in n | `S(n) = S2 * exp(k * (n - n2))` | k_per_dimension=0.17329, multiplier_per_dimension=1.1892 | 915,000 | **no** |
| **C** exponential in log2(secret-space size) | `S = S2 * exp(c * (bits - bits2))` | c_per_bit=0.90877, multiplier_per_bit=2.4813 | 8,962,000 | yes |
| **D** observed breakpoint ratio, applied once more | `S(next) = S2 * (S2 / S1)` | observed_ratio=4.0 | 647,000 | **no** |
| **E** power law in n | `S(n) = S2 * (n / n2) ** p` | exponent_p=2.7138 | 486,000 | **no** |

Spread across models: **29x**.

## 3. Prediction band (extrapolated, NOT measured)

| | samples | training h | validation h | total CPU h | days |
|---|---:|---:|---:|---:|---:|
| optimistic | 915,000 | 8.65 | 2.26 | **10.91** | 0.45 |
| central | 2,864,000 | 27.07 | 7.08 | **34.15** | 1.42 |
| conservative | 8,962,000 | 84.7 | 22.16 | **106.85** | 4.45 |

Runtimes use the throughput measured in the completed Salsa 2.0 n=30 pilot
(29.392 samples/s training, 45.0s per validation).

## 4. External anchor (NOT Salsa 2.0 data)

Wenger, Chen, Charton, Lauter, SALSA (NeurIPS 2022), Table 2 reports 3,913,424 to 29,210,829 samples (2^21.9-2^24.8) for n=30, q=251, sparse binary secret.

*External reference only. Not Salsa 2.0 data. Not fitted to. Different model (~51M parameters), different training setup.*

## 5. Why this prediction is uncertain

1. Only TWO positive breakpoints exist. Every two-parameter model fits them exactly, so the fits carry no residual and no goodness-of-fit evidence; model choice, not data, drives the answer.
2. Both measured breakpoints have h=2, while the target has h=3. Models A, B, D and E cannot see that change and are biased LOW by construction.
3. Only one n=30 pilot exists, and it produced no breakpoint - a censored observation. It bounds the answer from below (>100,032) and nothing more.
4. Sample-efficiency scaling need not stay linear, polynomial or exponential across this range; a regime change between n=20 and n=30 would invalidate all five models equally.
5. Model capacity has not been varied. 4.13M may be adequate at n=12-20 and inadequate at n=30, in which case no sample budget succeeds and the whole extrapolation is moot.
6. The n=20 run had not converged when its budget ran out - its loss was still falling - so its breakpoint is the earliest detectable one, not the point of useful performance.
7. The five models span a factor of 29x at n=30. A prediction with that spread cannot be settled by one more training run.

This is a prediction. It is not measured performance.
