# Broad Jev-like cross-backbone validation

## Outcome

The result is **positive but not universal**. Correct Qwen behavior transfers without target labels across all three heterogeneous targets on both Choice and ordered Score ID decisions. Ordered Score also transfers under dataset shift on every target. Boolean/Noul-like behavior does **not** reliably beat the mismatched-teacher control, and Choice OOD is mixed on Gemma. This meets the stated positive-evidence threshold of multiple backbones and multiple decision families, but it does not support a claim that all decision types transfer.

The evidence is broad enough to justify prototyping a real shared Jev-like DecisionCore, provided Boolean behavior and calibration are treated as explicit open requirements, not as solved capabilities.

At the overall level, correct transfer beat both negative controls in every individual seed for every target on both ID and OOD splits.

## Protocol

- Source: `Qwen/Qwen3-0.6B`.
- Targets: SmolLM2-360M-Instruct, Gemma 3 270M Instruct, and TinyLlama-1.1B-Chat-v1.0.
- Data: 4,500 train, 2,000 ID, and 1,500 OOD decisions.
- Train/ID: ARC-Easy Choice, BoolQ Boolean, Yelp five-level ordered Score.
- OOD: OpenBookQA Choice, QNLI Boolean, Amazon Reviews Multi five-level Score.
- Five fixed seeds (0-4), ten epochs, one optimizer configuration, no per-target/task/seed tuning, and exact frozen-hidden-state caching.
- Correct and mismatched transfer saw answer-free records only; Qwen, every backbone, the source adapter, and the source head stayed frozen; only each target adapter was optimized. All six target conditions began from the same per-target adapter state.
- Values below are mean ± sample standard deviation over five seeds.

## Overall transfer and behavior matching

| Target | Split | Correct acc. | Untrained acc. | Mismatch acc. | Gain vs untrained | Gain vs mismatch | Teacher agreement | Probability corr. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gemma | ID | 0.5114 ± 0.0070 | 0.3442 ± 0.0499 | 0.3640 ± 0.0116 | +0.1672 ± 0.0536 | +0.1474 ± 0.0102 | 0.5562 ± 0.0350 | 0.6707 ± 0.0174 |
| gemma | OOD | 0.3763 ± 0.0143 | 0.3273 ± 0.0155 | 0.3101 ± 0.0124 | +0.0489 ± 0.0217 | +0.0661 ± 0.0122 | 0.4753 ± 0.0629 | 0.5948 ± 0.0203 |
| smollm | ID | 0.5386 ± 0.0114 | 0.3477 ± 0.0264 | 0.3629 ± 0.0047 | +0.1909 ± 0.0334 | +0.1757 ± 0.0141 | 0.5754 ± 0.0286 | 0.7210 ± 0.0162 |
| smollm | OOD | 0.3915 ± 0.0188 | 0.3127 ± 0.0189 | 0.3064 ± 0.0048 | +0.0788 ± 0.0186 | +0.0851 ± 0.0198 | 0.4728 ± 0.0631 | 0.6105 ± 0.0127 |
| tinyllama | ID | 0.5520 ± 0.0101 | 0.3391 ± 0.0495 | 0.3625 ± 0.0105 | +0.2129 ± 0.0588 | +0.1895 ± 0.0130 | 0.5927 ± 0.0300 | 0.7346 ± 0.0134 |
| tinyllama | OOD | 0.3835 ± 0.0145 | 0.3212 ± 0.0111 | 0.3123 ± 0.0072 | +0.0623 ± 0.0184 | +0.0712 ± 0.0149 | 0.4991 ± 0.0827 | 0.5990 ± 0.0503 |

## Decision-type transfer gains

### ID

| Target | Type | Correct acc. | Gain vs untrained | Gain vs mismatch |
|---|---|---:|---:|---:|
| gemma | boolean | 0.6289 ± 0.0195 | +0.0789 ± 0.1252 | -0.0074 ± 0.0144 |
| gemma | choice | 0.4312 ± 0.0070 | +0.1786 ± 0.0411 | +0.1933 ± 0.0445 |
| gemma | score | 0.4614 ± 0.0079 | +0.2430 ± 0.0238 | +0.2600 ± 0.0197 |
| smollm | boolean | 0.6371 ± 0.0052 | +0.0634 ± 0.0729 | +0.0017 ± 0.0064 |
| smollm | choice | 0.5326 ± 0.0117 | +0.2849 ± 0.0252 | +0.2863 ± 0.0289 |
| smollm | score | 0.4488 ± 0.0218 | +0.2397 ± 0.0255 | +0.2562 ± 0.0181 |
| tinyllama | boolean | 0.6360 ± 0.0027 | +0.0994 ± 0.1329 | +0.0069 ± 0.0133 |
| tinyllama | choice | 0.5305 ± 0.0064 | +0.2611 ± 0.0117 | +0.2828 ± 0.0259 |
| tinyllama | score | 0.4882 ± 0.0257 | +0.2841 ± 0.0441 | +0.2918 ± 0.0297 |

### OOD

| Target | Type | Correct acc. | Gain vs untrained | Gain vs mismatch |
|---|---|---:|---:|---:|
| gemma | boolean | 0.4824 ± 0.0075 | -0.0000 ± 0.0250 | +0.0040 ± 0.0111 |
| gemma | choice | 0.2712 ± 0.0076 | -0.0128 ± 0.0378 | +0.0188 ± 0.0386 |
| gemma | score | 0.3752 ± 0.0353 | +0.1596 ± 0.0332 | +0.1756 ± 0.0398 |
| smollm | boolean | 0.4976 ± 0.0350 | +0.0192 ± 0.0198 | +0.0224 ± 0.0345 |
| smollm | choice | 0.3468 ± 0.0180 | +0.0904 ± 0.0467 | +0.1024 ± 0.0139 |
| smollm | score | 0.3300 ± 0.0190 | +0.1268 ± 0.0218 | +0.1304 ± 0.0226 |
| tinyllama | boolean | 0.4752 ± 0.0041 | -0.0156 ± 0.0221 | -0.0076 ± 0.0215 |
| tinyllama | choice | 0.3188 ± 0.0171 | +0.0380 ± 0.0530 | +0.0672 ± 0.0287 |
| tinyllama | score | 0.3564 ± 0.0294 | +0.1644 ± 0.0415 | +0.1540 ± 0.0353 |

## Dataset-level negative-control test

Each split has one dataset per decision type; this table makes the required dataset-level comparison explicit.

| Target | Split | Dataset | Correct acc. | Gain vs untrained | Gain vs mismatch |
|---|---|---|---:|---:|---:|
| gemma | ID | arc_easy | 0.4312 ± 0.0070 | +0.1786 ± 0.0411 | +0.1933 ± 0.0445 |
| gemma | ID | boolq | 0.6289 ± 0.0195 | +0.0789 ± 0.1252 | -0.0074 ± 0.0144 |
| gemma | ID | yelp_review_full | 0.4614 ± 0.0079 | +0.2430 ± 0.0238 | +0.2600 ± 0.0197 |
| gemma | OOD | amazon_reviews_multi | 0.3752 ± 0.0353 | +0.1596 ± 0.0332 | +0.1756 ± 0.0398 |
| gemma | OOD | openbookqa | 0.2712 ± 0.0076 | -0.0128 ± 0.0378 | +0.0188 ± 0.0386 |
| gemma | OOD | qnli | 0.4824 ± 0.0075 | -0.0000 ± 0.0250 | +0.0040 ± 0.0111 |
| smollm | ID | arc_easy | 0.5326 ± 0.0117 | +0.2849 ± 0.0252 | +0.2863 ± 0.0289 |
| smollm | ID | boolq | 0.6371 ± 0.0052 | +0.0634 ± 0.0729 | +0.0017 ± 0.0064 |
| smollm | ID | yelp_review_full | 0.4488 ± 0.0218 | +0.2397 ± 0.0255 | +0.2562 ± 0.0181 |
| smollm | OOD | amazon_reviews_multi | 0.3300 ± 0.0190 | +0.1268 ± 0.0218 | +0.1304 ± 0.0226 |
| smollm | OOD | openbookqa | 0.3468 ± 0.0180 | +0.0904 ± 0.0467 | +0.1024 ± 0.0139 |
| smollm | OOD | qnli | 0.4976 ± 0.0350 | +0.0192 ± 0.0198 | +0.0224 ± 0.0345 |
| tinyllama | ID | arc_easy | 0.5305 ± 0.0064 | +0.2611 ± 0.0117 | +0.2828 ± 0.0259 |
| tinyllama | ID | boolq | 0.6360 ± 0.0027 | +0.0994 ± 0.1329 | +0.0069 ± 0.0133 |
| tinyllama | ID | yelp_review_full | 0.4882 ± 0.0257 | +0.2841 ± 0.0441 | +0.2918 ± 0.0297 |
| tinyllama | OOD | amazon_reviews_multi | 0.3564 ± 0.0294 | +0.1644 ± 0.0415 | +0.1540 ± 0.0353 |
| tinyllama | OOD | openbookqa | 0.3188 ± 0.0171 | +0.0380 ± 0.0530 | +0.0672 ± 0.0287 |
| tinyllama | OOD | qnli | 0.4752 ± 0.0041 | -0.0156 ± 0.0221 | -0.0076 ± 0.0215 |

## Macro accuracy across task families

| Target | Condition | ID macro acc. | OOD macro acc. |
|---|---|---:|---:|
| gemma | untrained adapter | 0.3403 ± 0.0481 | 0.3273 ± 0.0155 |
| gemma | correct teacher | 0.5072 ± 0.0069 | 0.3763 ± 0.0143 |
| gemma | mismatched teacher | 0.3585 ± 0.0130 | 0.3101 ± 0.0124 |
| gemma | native supervised | 0.5006 ± 0.0086 | 0.3760 ± 0.0103 |
| gemma | supervised DecPort | 0.5052 ± 0.0052 | 0.3823 ± 0.0064 |
| gemma | random-head supervised | 0.5024 ± 0.0073 | 0.3920 ± 0.0087 |
| smollm | untrained adapter | 0.3435 ± 0.0256 | 0.3127 ± 0.0189 |
| smollm | correct teacher | 0.5395 ± 0.0113 | 0.3915 ± 0.0188 |
| smollm | mismatched teacher | 0.3581 ± 0.0052 | 0.3064 ± 0.0048 |
| smollm | native supervised | 0.5408 ± 0.0050 | 0.3887 ± 0.0106 |
| smollm | supervised DecPort | 0.5390 ± 0.0062 | 0.3844 ± 0.0103 |
| smollm | random-head supervised | 0.5344 ± 0.0138 | 0.3853 ± 0.0126 |
| tinyllama | untrained adapter | 0.3367 ± 0.0468 | 0.3212 ± 0.0111 |
| tinyllama | correct teacher | 0.5516 ± 0.0095 | 0.3835 ± 0.0145 |
| tinyllama | mismatched teacher | 0.3578 ± 0.0106 | 0.3123 ± 0.0072 |
| tinyllama | native supervised | 0.5760 ± 0.0060 | 0.4028 ± 0.0059 |
| tinyllama | supervised DecPort | 0.5746 ± 0.0099 | 0.4037 ± 0.0088 |
| tinyllama | random-head supervised | 0.5766 ± 0.0092 | 0.3949 ± 0.0053 |

## Full overall metrics — ID

| Target | Condition | Accuracy | Macro-F1 | NLL | Brier | ECE | Teacher agreement | Probability corr. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gemma | untrained adapter | 0.3442 ± 0.0499 | 0.1456 ± 0.0285 | 1.2239 ± 0.0082 | 0.6793 ± 0.0072 | 0.0483 ± 0.0128 | 0.3355 ± 0.0952 | 0.5305 ± 0.0285 |
| gemma | correct teacher | 0.5114 ± 0.0070 | 0.2780 ± 0.0062 | 1.0735 ± 0.0133 | 0.6055 ± 0.0039 | 0.0483 ± 0.0134 | 0.5562 ± 0.0350 | 0.6707 ± 0.0174 |
| gemma | mismatched teacher | 0.3640 ± 0.0116 | 0.1368 ± 0.0268 | 1.2346 ± 0.0094 | 0.6791 ± 0.0046 | 0.0562 ± 0.0132 | 0.4061 ± 0.0388 | 0.5422 ± 0.0288 |
| gemma | native supervised | 0.5059 ± 0.0086 | 0.2601 ± 0.0112 | 1.0697 ± 0.0354 | 0.6050 ± 0.0130 | 0.0784 ± 0.0204 | 0.5159 ± 0.0231 | 0.6264 ± 0.0210 |
| gemma | supervised DecPort | 0.5109 ± 0.0052 | 0.2601 ± 0.0086 | 1.0475 ± 0.0208 | 0.5989 ± 0.0091 | 0.0718 ± 0.0250 | 0.5253 ± 0.0281 | 0.6306 ± 0.0242 |
| gemma | random-head supervised | 0.5072 ± 0.0069 | 0.2668 ± 0.0131 | 1.0540 ± 0.0074 | 0.6016 ± 0.0042 | 0.0492 ± 0.0058 | 0.5161 ± 0.0260 | 0.6308 ± 0.0194 |
| smollm | untrained adapter | 0.3477 ± 0.0264 | 0.1432 ± 0.0189 | 1.2249 ± 0.0035 | 0.6794 ± 0.0028 | 0.0404 ± 0.0156 | 0.3599 ± 0.0457 | 0.5311 ± 0.0247 |
| smollm | correct teacher | 0.5386 ± 0.0114 | 0.3654 ± 0.0093 | 1.0242 ± 0.0142 | 0.5701 ± 0.0039 | 0.0358 ± 0.0139 | 0.5754 ± 0.0286 | 0.7210 ± 0.0162 |
| smollm | mismatched teacher | 0.3629 ± 0.0047 | 0.1427 ± 0.0128 | 1.2464 ± 0.0130 | 0.6849 ± 0.0058 | 0.0686 ± 0.0184 | 0.4062 ± 0.0416 | 0.5367 ± 0.0278 |
| smollm | native supervised | 0.5395 ± 0.0056 | 0.3653 ± 0.0087 | 1.0361 ± 0.0169 | 0.5750 ± 0.0068 | 0.0466 ± 0.0127 | 0.5486 ± 0.0245 | 0.6770 ± 0.0210 |
| smollm | supervised DecPort | 0.5376 ± 0.0065 | 0.3632 ± 0.0082 | 1.0443 ± 0.0064 | 0.5798 ± 0.0024 | 0.0529 ± 0.0060 | 0.5406 ± 0.0200 | 0.6747 ± 0.0158 |
| smollm | random-head supervised | 0.5325 ± 0.0146 | 0.3655 ± 0.0072 | 1.0647 ± 0.0311 | 0.5880 ± 0.0142 | 0.0579 ± 0.0291 | 0.5513 ± 0.0291 | 0.6632 ± 0.0224 |
| tinyllama | untrained adapter | 0.3391 ± 0.0495 | 0.1541 ± 0.0071 | 1.2371 ± 0.0195 | 0.6859 ± 0.0188 | 0.0652 ± 0.0348 | 0.3376 ± 0.0558 | 0.5262 ± 0.0360 |
| tinyllama | correct teacher | 0.5520 ± 0.0101 | 0.3627 ± 0.0060 | 0.9957 ± 0.0261 | 0.5610 ± 0.0120 | 0.0390 ± 0.0128 | 0.5927 ± 0.0300 | 0.7346 ± 0.0134 |
| tinyllama | mismatched teacher | 0.3625 ± 0.0105 | 0.1419 ± 0.0170 | 1.2538 ± 0.0157 | 0.6895 ± 0.0083 | 0.0774 ± 0.0283 | 0.4053 ± 0.0515 | 0.5296 ± 0.0267 |
| tinyllama | native supervised | 0.5786 ± 0.0063 | 0.3594 ± 0.0112 | 0.9513 ± 0.0159 | 0.5431 ± 0.0072 | 0.0628 ± 0.0125 | 0.5714 ± 0.0296 | 0.6904 ± 0.0148 |
| tinyllama | supervised DecPort | 0.5770 ± 0.0112 | 0.3599 ± 0.0101 | 0.9580 ± 0.0218 | 0.5465 ± 0.0109 | 0.0634 ± 0.0167 | 0.5742 ± 0.0263 | 0.6894 ± 0.0163 |
| tinyllama | random-head supervised | 0.5786 ± 0.0105 | 0.3666 ± 0.0070 | 0.9609 ± 0.0364 | 0.5449 ± 0.0156 | 0.0534 ± 0.0128 | 0.5815 ± 0.0214 | 0.6909 ± 0.0154 |

## Full overall metrics — OOD

| Target | Condition | Accuracy | Macro-F1 | NLL | Brier | ECE | Teacher agreement | Probability corr. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gemma | untrained adapter | 0.3273 ± 0.0155 | 0.1683 ± 0.0209 | 1.2333 ± 0.0061 | 0.6859 ± 0.0038 | 0.0336 ± 0.0172 | 0.3175 ± 0.0863 | 0.6109 ± 0.0284 |
| gemma | correct teacher | 0.3763 ± 0.0143 | 0.1594 ± 0.0049 | 1.3799 ± 0.0893 | 0.7575 ± 0.0346 | 0.1855 ± 0.0500 | 0.4753 ± 0.0629 | 0.5948 ± 0.0203 |
| gemma | mismatched teacher | 0.3101 ± 0.0124 | 0.1461 ± 0.0248 | 1.2745 ± 0.0320 | 0.7111 ± 0.0184 | 0.0988 ± 0.0290 | 0.3992 ± 0.0491 | 0.6010 ± 0.0212 |
| gemma | native supervised | 0.3760 ± 0.0103 | 0.1707 ± 0.0029 | 1.4150 ± 0.0796 | 0.7635 ± 0.0192 | 0.2162 ± 0.0218 | 0.4404 ± 0.0523 | 0.5519 ± 0.0167 |
| gemma | supervised DecPort | 0.3823 ± 0.0064 | 0.1665 ± 0.0061 | 1.3585 ± 0.0533 | 0.7501 ± 0.0291 | 0.2070 ± 0.0349 | 0.4416 ± 0.0627 | 0.5647 ± 0.0291 |
| gemma | random-head supervised | 0.3920 ± 0.0087 | 0.1735 ± 0.0093 | 1.3235 ± 0.0426 | 0.7338 ± 0.0220 | 0.1753 ± 0.0272 | 0.4241 ± 0.0551 | 0.5655 ± 0.0234 |
| smollm | untrained adapter | 0.3127 ± 0.0189 | 0.1498 ± 0.0268 | 1.2337 ± 0.0041 | 0.6857 ± 0.0023 | 0.0326 ± 0.0188 | 0.3632 ± 0.0557 | 0.6143 ± 0.0342 |
| smollm | correct teacher | 0.3915 ± 0.0188 | 0.2129 ± 0.0119 | 1.3653 ± 0.0753 | 0.7363 ± 0.0377 | 0.1789 ± 0.0450 | 0.4728 ± 0.0631 | 0.6105 ± 0.0127 |
| smollm | mismatched teacher | 0.3064 ± 0.0048 | 0.1410 ± 0.0070 | 1.2886 ± 0.0253 | 0.7145 ± 0.0145 | 0.1017 ± 0.0306 | 0.3971 ± 0.0593 | 0.6033 ± 0.0318 |
| smollm | native supervised | 0.3887 ± 0.0106 | 0.2155 ± 0.0099 | 1.4669 ± 0.0604 | 0.7750 ± 0.0306 | 0.2254 ± 0.0321 | 0.4560 ± 0.0724 | 0.5569 ± 0.0266 |
| smollm | supervised DecPort | 0.3844 ± 0.0103 | 0.2132 ± 0.0069 | 1.4311 ± 0.0275 | 0.7640 ± 0.0187 | 0.2070 ± 0.0257 | 0.4551 ± 0.0718 | 0.5639 ± 0.0232 |
| smollm | random-head supervised | 0.3853 ± 0.0126 | 0.2071 ± 0.0092 | 1.3868 ± 0.0446 | 0.7554 ± 0.0168 | 0.1777 ± 0.0184 | 0.4644 ± 0.0728 | 0.5698 ± 0.0378 |
| tinyllama | untrained adapter | 0.3212 ± 0.0111 | 0.1656 ± 0.0301 | 1.2564 ± 0.0141 | 0.7045 ± 0.0140 | 0.0791 ± 0.0232 | 0.3425 ± 0.0626 | 0.6026 ± 0.0485 |
| tinyllama | correct teacher | 0.3835 ± 0.0145 | 0.1942 ± 0.0132 | 1.4167 ± 0.0940 | 0.7789 ± 0.0306 | 0.2323 ± 0.0337 | 0.4991 ± 0.0827 | 0.5990 ± 0.0503 |
| tinyllama | mismatched teacher | 0.3123 ± 0.0072 | 0.1447 ± 0.0108 | 1.3323 ± 0.1072 | 0.7395 ± 0.0563 | 0.1238 ± 0.0631 | 0.3700 ± 0.0545 | 0.5606 ± 0.0571 |
| tinyllama | native supervised | 0.4028 ± 0.0059 | 0.2109 ± 0.0084 | 1.5266 ± 0.0722 | 0.8115 ± 0.0218 | 0.2773 ± 0.0300 | 0.4903 ± 0.0705 | 0.5325 ± 0.0444 |
| tinyllama | supervised DecPort | 0.4037 ± 0.0088 | 0.2144 ± 0.0099 | 1.5376 ± 0.0859 | 0.8147 ± 0.0217 | 0.2765 ± 0.0280 | 0.4892 ± 0.0715 | 0.5308 ± 0.0504 |
| tinyllama | random-head supervised | 0.3949 ± 0.0053 | 0.2121 ± 0.0114 | 1.6007 ± 0.1114 | 0.8327 ± 0.0304 | 0.2891 ± 0.0329 | 0.4876 ± 0.0818 | 0.5122 ± 0.0555 |

## Ordered Score metric

Lower ordinal MAE is better.

| Target | Condition | Yelp ID MAE | Amazon OOD MAE |
|---|---|---:|---:|
| gemma | untrained adapter | 1.7353 ± 0.2200 | 1.7200 ± 0.2726 |
| gemma | correct teacher | 0.7756 ± 0.0560 | 1.0660 ± 0.1751 |
| gemma | mismatched teacher | 1.5934 ± 0.1934 | 1.5444 ± 0.2218 |
| gemma | native supervised | 0.7712 ± 0.0605 | 1.0660 ± 0.1213 |
| gemma | supervised DecPort | 0.7252 ± 0.0373 | 0.9492 ± 0.0602 |
| gemma | random-head supervised | 0.7403 ± 0.0237 | 1.0132 ± 0.0843 |
| smollm | untrained adapter | 1.7701 ± 0.2698 | 1.7468 ± 0.3119 |
| smollm | correct teacher | 0.8252 ± 0.0578 | 1.1300 ± 0.0566 |
| smollm | mismatched teacher | 1.6458 ± 0.2269 | 1.6272 ± 0.2668 |
| smollm | native supervised | 0.8953 ± 0.0641 | 1.1976 ± 0.0781 |
| smollm | supervised DecPort | 0.9060 ± 0.0535 | 1.2104 ± 0.0810 |
| smollm | random-head supervised | 1.0068 ± 0.1084 | 1.2324 ± 0.1513 |
| tinyllama | untrained adapter | 1.7882 ± 0.3189 | 1.8100 ± 0.3698 |
| tinyllama | correct teacher | 0.7581 ± 0.1048 | 1.1452 ± 0.1257 |
| tinyllama | mismatched teacher | 1.6266 ± 0.1479 | 1.5892 ± 0.2621 |
| tinyllama | native supervised | 0.6151 ± 0.0719 | 0.9536 ± 0.0941 |
| tinyllama | supervised DecPort | 0.6444 ± 0.1202 | 0.9776 ± 0.1279 |
| tinyllama | random-head supervised | 0.6444 ± 0.0957 | 1.0892 ± 0.0924 |

## Source reference

| Split | Overall acc. | Macro task-family acc. | Macro-F1 | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| ID | 0.6204 ± 0.0052 | 0.6262 ± 0.0051 | 0.5170 ± 0.0063 | 0.8901 ± 0.0131 | 0.4931 ± 0.0035 | 0.0461 ± 0.0126 |
| OOD | 0.4285 ± 0.0042 | 0.4285 ± 0.0042 | 0.2406 ± 0.0078 | 1.3225 ± 0.0588 | 0.7027 ± 0.0164 | 0.1701 ± 0.0257 |

## Interpretation

- **Choice:** strong ID transfer on every target. OOD OpenBookQA remains positive against both controls for SmolLM and TinyLlama; Gemma is above the mismatched teacher but slightly below its untrained control.
- **Boolean / Noul-like:** not established. ID gains over the behavior-mismatched teacher are near zero (and negative for Gemma), while OOD QNLI is also near zero or negative. Aggregate gains must not be read as Boolean success.
- **Ordered Score:** the clearest result. Correct behavior beats both controls on Yelp and Amazon for all three targets, including dataset shift, and ordinal MAE is reported alongside exact accuracy.
- Calibration remains condition- and backbone-dependent. The full NLL, Brier, and ECE table is retained rather than hiding those failures behind accuracy.
- The result is not attributable only to SST-2/AG News (neither appears here), one backbone, or one easy dataset.

## Archive and audit

The archive contains the fixed repository/expanded configs, data provenance and hashes, exact per-seed JSON, all lightweight adapter/head artifacts, aggregate JSON, environment/model revisions, GPU telemetry, and `SHA256SUMS`. `scripts/audit_broad_results.py` passed before this report was rendered.

This remains diagnostic evidence, not an accepted benchmark or a universal portability claim.
