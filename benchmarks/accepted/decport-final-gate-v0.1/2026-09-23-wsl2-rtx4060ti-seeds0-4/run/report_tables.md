## Source system

| System | Split | Overall | Choice | Noul | Score | NLL | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| Open-Jev repr. + learned core | ID | 0.7455 | 0.8930 | 0.8543 | 0.5260 | 0.5778 | 0.0227 |
| Open-Jev repr. + learned core | OOD | 0.6193 | 0.6320 | 0.6920 | 0.5340 | 0.9282 | 0.0945 |
| Open-Jev 2B (own linear head) | ID | 0.7310 | 0.8877 | 0.7929 | 0.5493 | 0.6912 | 0.0808 |
| Open-Jev 2B (own linear head) | OOD | 0.6467 | 0.6080 | 0.8600 | 0.4720 | 0.8465 | 0.0747 |

Learned-core temperature 1.0842; source calibration-split accuracy 0.7600, NLL 0.5816.

## Overall accuracy (mean ± SD over seeds)

| Target | Split | A learned core | B random core | C mismatched | D untrained | E target-specific |
|---|---|---:|---:|---:|---:|---:|
| Gemma 3 270M | ID | 0.4501 ± 0.0091 | 0.4354 ± 0.0411 | 0.3453 ± 0.0068 | 0.3195 ± 0.0489 | 0.4565 ± 0.0188 |
| Gemma 3 270M | OOD | 0.3507 ± 0.0172 | 0.3505 ± 0.0162 | 0.3100 ± 0.0139 | 0.3100 ± 0.0132 | 0.3671 ± 0.0223 |
| SmolLM2-360M | ID | 0.5137 ± 0.0061 | 0.5137 ± 0.0271 | 0.3696 ± 0.0109 | 0.3635 ± 0.0126 | 0.5192 ± 0.0120 |
| SmolLM2-360M | OOD | 0.3564 ± 0.0080 | 0.3507 ± 0.0114 | 0.3141 ± 0.0118 | 0.3137 ± 0.0115 | 0.3523 ± 0.0104 |
| TinyLlama-1.1B | ID | 0.5172 ± 0.0126 | 0.5317 ± 0.0161 | 0.3482 ± 0.0179 | 0.3283 ± 0.0303 | 0.5347 ± 0.0060 |
| TinyLlama-1.1B | OOD | 0.3783 ± 0.0063 | 0.3896 ± 0.0097 | 0.3053 ± 0.0100 | 0.3116 ± 0.0206 | 0.3824 ± 0.0046 |

## Paired per-seed differences: A learned core − control (overall accuracy)

| Target | Split | Control | Mean ± SD | Seeds > 0 | Per seed |
|---|---|---|---:|---:|---|
| Gemma 3 270M | ID | B random core | +0.0147 ± 0.0394 | 2/5 | +0.0820, -0.0060, -0.0125, +0.0180, -0.0080 |
| Gemma 3 270M | ID | C mismatched | +0.1048 ± 0.0133 | 5/5 | +0.0940, +0.0935, +0.0980, +0.1200, +0.1185 |
| Gemma 3 270M | ID | D untrained | +0.1306 ± 0.0525 | 5/5 | +0.0870, +0.1130, +0.1785, +0.1940, +0.0805 |
| Gemma 3 270M | ID | E target-specific | -0.0064 ± 0.0248 | 2/5 | -0.0130, -0.0415, -0.0125, +0.0165, +0.0185 |
| Gemma 3 270M | OOD | B random core | +0.0001 ± 0.0098 | 2/5 | -0.0133, +0.0087, +0.0107, -0.0013, -0.0040 |
| Gemma 3 270M | OOD | C mismatched | +0.0407 ± 0.0139 | 5/5 | +0.0453, +0.0193, +0.0580, +0.0407, +0.0400 |
| Gemma 3 270M | OOD | D untrained | +0.0407 ± 0.0270 | 4/5 | +0.0440, +0.0393, +0.0747, -0.0007, +0.0460 |
| Gemma 3 270M | OOD | E target-specific | -0.0164 ± 0.0299 | 1/5 | -0.0013, -0.0373, +0.0293, -0.0327, -0.0400 |
| SmolLM2-360M | ID | B random core | -0.0000 ± 0.0289 | 1/5 | -0.0055, -0.0105, -0.0230, -0.0115, +0.0505 |
| SmolLM2-360M | ID | C mismatched | +0.1441 ± 0.0082 | 5/5 | +0.1545, +0.1320, +0.1425, +0.1475, +0.1440 |
| SmolLM2-360M | ID | D untrained | +0.1502 ± 0.0100 | 5/5 | +0.1385, +0.1435, +0.1630, +0.1485, +0.1575 |
| SmolLM2-360M | ID | E target-specific | -0.0055 ± 0.0109 | 2/5 | -0.0215, +0.0035, -0.0080, -0.0075, +0.0060 |
| SmolLM2-360M | OOD | B random core | +0.0057 ± 0.0070 | 5/5 | +0.0047, +0.0020, +0.0033, +0.0007, +0.0180 |
| SmolLM2-360M | OOD | C mismatched | +0.0423 ± 0.0190 | 5/5 | +0.0320, +0.0180, +0.0493, +0.0687, +0.0433 |
| SmolLM2-360M | OOD | D untrained | +0.0427 ± 0.0162 | 5/5 | +0.0253, +0.0260, +0.0493, +0.0613, +0.0513 |
| SmolLM2-360M | OOD | E target-specific | +0.0041 ± 0.0117 | 4/5 | +0.0080, -0.0147, +0.0067, +0.0033, +0.0173 |
| TinyLlama-1.1B | ID | B random core | -0.0145 ± 0.0216 | 2/5 | -0.0185, +0.0065, -0.0485, -0.0135, +0.0015 |
| TinyLlama-1.1B | ID | C mismatched | +0.1690 ± 0.0095 | 5/5 | +0.1690, +0.1530, +0.1740, +0.1775, +0.1715 |
| TinyLlama-1.1B | ID | D untrained | +0.1889 ± 0.0333 | 5/5 | +0.1705, +0.2260, +0.1480, +0.1800, +0.2200 |
| TinyLlama-1.1B | ID | E target-specific | -0.0175 ± 0.0138 | 0/5 | -0.0105, -0.0075, -0.0390, -0.0070, -0.0235 |
| TinyLlama-1.1B | OOD | B random core | -0.0113 ± 0.0065 | 0/5 | -0.0020, -0.0153, -0.0093, -0.0107, -0.0193 |
| TinyLlama-1.1B | OOD | C mismatched | +0.0729 ± 0.0122 | 5/5 | +0.0647, +0.0933, +0.0753, +0.0660, +0.0653 |
| TinyLlama-1.1B | OOD | D untrained | +0.0667 ± 0.0198 | 5/5 | +0.0660, +0.0880, +0.0547, +0.0840, +0.0407 |
| TinyLlama-1.1B | OOD | E target-specific | -0.0041 ± 0.0061 | 1/5 | +0.0033, -0.0040, -0.0067, -0.0127, -0.0007 |

## Accuracy by decision type (mean ± SD)

| Target | Split | Type | Source | A learned core | B random core | C mismatched | D untrained | E target-specific |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Gemma 3 270M | ID | choice | 0.8930 | 0.4568 ± 0.0145 | 0.4512 ± 0.0208 | 0.2133 ± 0.0183 | 0.2379 ± 0.0212 | 0.4621 ± 0.0123 |
| Gemma 3 270M | ID | noul | 0.8543 | 0.6291 ± 0.0296 | 0.5994 ± 0.1007 | 0.6386 ± 0.0000 | 0.5060 ± 0.1392 | 0.6189 ± 0.0371 |
| Gemma 3 270M | ID | score | 0.5260 | 0.2732 ± 0.0323 | 0.2658 ± 0.0094 | 0.1671 ± 0.0064 | 0.2044 ± 0.0166 | 0.2964 ± 0.0212 |
| Gemma 3 270M | OOD | choice | 0.6320 | 0.2648 ± 0.0171 | 0.2552 ± 0.0094 | 0.2692 ± 0.0397 | 0.2460 ± 0.0315 | 0.2516 ± 0.0038 |
| Gemma 3 270M | OOD | noul | 0.6920 | 0.4908 ± 0.0202 | 0.5152 ± 0.0372 | 0.4760 ± 0.0000 | 0.4996 ± 0.0240 | 0.5176 ± 0.0340 |
| Gemma 3 270M | OOD | score | 0.5340 | 0.2964 ± 0.0236 | 0.2812 ± 0.0083 | 0.1848 ± 0.0181 | 0.1844 ± 0.0134 | 0.3320 ± 0.0415 |
| SmolLM2-360M | ID | choice | 0.8930 | 0.6561 ± 0.0134 | 0.6604 ± 0.0098 | 0.2814 ± 0.0295 | 0.2312 ± 0.0475 | 0.6474 ± 0.0136 |
| SmolLM2-360M | ID | noul | 0.8543 | 0.6366 ± 0.0113 | 0.5994 ± 0.0668 | 0.6386 ± 0.0000 | 0.6389 ± 0.0006 | 0.5957 ± 0.0264 |
| SmolLM2-360M | ID | score | 0.5260 | 0.2847 ± 0.0114 | 0.3170 ± 0.0115 | 0.1805 ± 0.0214 | 0.2027 ± 0.0049 | 0.3458 ± 0.0131 |
| SmolLM2-360M | OOD | choice | 0.6320 | 0.3324 ± 0.0148 | 0.3176 ± 0.0151 | 0.2644 ± 0.0186 | 0.2780 ± 0.0055 | 0.3228 ± 0.0160 |
| SmolLM2-360M | OOD | noul | 0.6920 | 0.4736 ± 0.0026 | 0.4756 ± 0.0134 | 0.4760 ± 0.0000 | 0.4760 ± 0.0000 | 0.4676 ± 0.0211 |
| SmolLM2-360M | OOD | score | 0.5340 | 0.2632 ± 0.0146 | 0.2588 ± 0.0175 | 0.2020 ± 0.0171 | 0.1872 ± 0.0311 | 0.2664 ± 0.0093 |
| TinyLlama-1.1B | ID | choice | 0.8930 | 0.5263 ± 0.0126 | 0.5368 ± 0.0110 | 0.2558 ± 0.0356 | 0.2411 ± 0.0137 | 0.5242 ± 0.0151 |
| TinyLlama-1.1B | ID | noul | 0.8543 | 0.6011 ± 0.0425 | 0.6080 ± 0.0484 | 0.6186 ± 0.0440 | 0.5349 ± 0.0787 | 0.6480 ± 0.0236 |
| TinyLlama-1.1B | ID | score | 0.5260 | 0.4296 ± 0.0155 | 0.4545 ± 0.0124 | 0.1611 ± 0.0160 | 0.1984 ± 0.0199 | 0.4342 ± 0.0125 |
| TinyLlama-1.1B | OOD | choice | 0.6320 | 0.2932 ± 0.0106 | 0.3052 ± 0.0149 | 0.2396 ± 0.0242 | 0.2576 ± 0.0273 | 0.2984 ± 0.0085 |
| TinyLlama-1.1B | OOD | noul | 0.6920 | 0.4768 ± 0.0018 | 0.4856 ± 0.0238 | 0.4900 ± 0.0183 | 0.4828 ± 0.0388 | 0.4756 ± 0.0009 |
| TinyLlama-1.1B | OOD | score | 0.5340 | 0.3648 ± 0.0119 | 0.3780 ± 0.0248 | 0.1864 ± 0.0130 | 0.1944 ± 0.0140 | 0.3732 ± 0.0141 |

## Calibration and fidelity to the source system (overall, mean ± SD)

| Target | Split | Condition | Macro-F1 | NLL | Brier | ECE | Agreement | KL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Gemma 3 270M | ID | A learned core | 0.2947 ± 0.0107 | 1.1496 ± 0.0222 | 0.6391 ± 0.0068 | 0.0520 ± 0.0152 | 0.4744 ± 0.0064 | 0.5437 ± 0.0089 |
| Gemma 3 270M | ID | B random core | 0.2908 ± 0.0167 | 1.2404 ± 0.1080 | 0.6819 ± 0.0533 | 0.1156 ± 0.0729 | 0.4571 ± 0.0416 | 0.6329 ± 0.1166 |
| Gemma 3 270M | ID | C mismatched | 0.1200 ± 0.0103 | 1.2439 ± 0.0174 | 0.6806 ± 0.0066 | 0.0619 ± 0.0222 | 0.3689 ± 0.0115 | 0.6128 ± 0.0058 |
| Gemma 3 270M | ID | D untrained | 0.1344 ± 0.0140 | 1.6233 ± 0.3826 | 0.7933 ± 0.0805 | 0.1587 ± 0.0730 | 0.3013 ± 0.0495 | 1.0326 ± 0.4123 |
| Gemma 3 270M | ID | E target-specific | 0.3000 ± 0.0104 | 1.1237 ± 0.0351 | 0.6340 ± 0.0112 | 0.0785 ± 0.0097 | 0.4904 ± 0.0161 | 0.5326 ± 0.0236 |
| Gemma 3 270M | OOD | A learned core | 0.1564 ± 0.0106 | 1.3825 ± 0.0461 | 0.7401 ± 0.0147 | 0.1451 ± 0.0260 | 0.4469 ± 0.0304 | 0.6147 ± 0.0353 |
| Gemma 3 270M | OOD | B random core | 0.1496 ± 0.0061 | 1.5182 ± 0.2318 | 0.7979 ± 0.0868 | 0.1987 ± 0.0992 | 0.3916 ± 0.0763 | 0.7196 ± 0.1417 |
| Gemma 3 270M | OOD | C mismatched | 0.1575 ± 0.0263 | 1.2602 ± 0.0074 | 0.7052 ± 0.0062 | 0.0828 ± 0.0200 | 0.4299 ± 0.0050 | 0.4971 ± 0.0133 |
| Gemma 3 270M | OOD | D untrained | 0.1430 ± 0.0233 | 1.7615 ± 0.8005 | 0.8022 ± 0.0759 | 0.1649 ± 0.0776 | 0.2903 ± 0.1002 | 0.8902 ± 0.3669 |
| Gemma 3 270M | OOD | E target-specific | 0.1475 ± 0.0033 | 1.4150 ± 0.0422 | 0.7507 ± 0.0183 | 0.1553 ± 0.0307 | 0.4436 ± 0.0442 | 0.6298 ± 0.0452 |
| SmolLM2-360M | ID | A learned core | 0.4813 ± 0.0150 | 1.0576 ± 0.0202 | 0.5837 ± 0.0055 | 0.0507 ± 0.0182 | 0.5205 ± 0.0030 | 0.4929 ± 0.0294 |
| SmolLM2-360M | ID | B random core | 0.4863 ± 0.0104 | 1.4566 ± 0.6316 | 0.6338 ± 0.0324 | 0.1228 ± 0.0405 | 0.5206 ± 0.0205 | 0.9209 ± 0.6746 |
| SmolLM2-360M | ID | C mismatched | 0.1627 ± 0.0180 | 1.2269 ± 0.0124 | 0.6736 ± 0.0052 | 0.0275 ± 0.0157 | 0.3898 ± 0.0085 | 0.6083 ± 0.0025 |
| SmolLM2-360M | ID | D untrained | 0.1335 ± 0.0304 | 1.6247 ± 0.2049 | 0.7638 ± 0.0168 | 0.1837 ± 0.0328 | 0.3548 ± 0.0133 | 1.0482 ± 0.2115 |
| SmolLM2-360M | ID | E target-specific | 0.4710 ± 0.0145 | 1.0276 ± 0.0145 | 0.5737 ± 0.0062 | 0.0450 ± 0.0125 | 0.5286 ± 0.0086 | 0.4759 ± 0.0157 |
| SmolLM2-360M | OOD | A learned core | 0.2046 ± 0.0094 | 1.5399 ± 0.0934 | 0.7864 ± 0.0212 | 0.2186 ± 0.0316 | 0.4765 ± 0.0069 | 0.7235 ± 0.0992 |
| SmolLM2-360M | OOD | B random core | 0.1936 ± 0.0101 | 2.1099 ± 0.9904 | 0.8666 ± 0.0668 | 0.2867 ± 0.0543 | 0.4396 ± 0.0652 | 0.9882 ± 0.4593 |
| SmolLM2-360M | OOD | C mismatched | 0.1553 ± 0.0145 | 1.2483 ± 0.0082 | 0.6998 ± 0.0081 | 0.0599 ± 0.0107 | 0.4215 ± 0.0199 | 0.4939 ± 0.0084 |
| SmolLM2-360M | OOD | D untrained | 0.1637 ± 0.0030 | 1.7707 ± 0.3384 | 0.8555 ± 0.0378 | 0.2282 ± 0.0434 | 0.4096 ± 0.0071 | 0.7286 ± 0.1809 |
| SmolLM2-360M | OOD | E target-specific | 0.1968 ± 0.0118 | 1.5049 ± 0.0298 | 0.7801 ± 0.0251 | 0.2156 ± 0.0324 | 0.4376 ± 0.0475 | 0.7170 ± 0.0440 |
| TinyLlama-1.1B | ID | A learned core | 0.3528 ± 0.0104 | 1.0673 ± 0.0578 | 0.5938 ± 0.0115 | 0.0689 ± 0.0218 | 0.5394 ± 0.0100 | 0.5250 ± 0.0660 |
| TinyLlama-1.1B | ID | B random core | 0.3614 ± 0.0104 | 1.1191 ± 0.1574 | 0.6072 ± 0.0310 | 0.1026 ± 0.0424 | 0.5425 ± 0.0160 | 0.5944 ± 0.1722 |
| TinyLlama-1.1B | ID | C mismatched | 0.1477 ± 0.0218 | 1.2671 ± 0.0303 | 0.6932 ± 0.0129 | 0.0828 ± 0.0188 | 0.3706 ± 0.0223 | 0.6328 ± 0.0233 |
| TinyLlama-1.1B | ID | D untrained | 0.1365 ± 0.0083 | 1.3995 ± 0.0916 | 0.7406 ± 0.0243 | 0.1312 ± 0.0372 | 0.3356 ± 0.0269 | 0.7949 ± 0.0913 |
| TinyLlama-1.1B | ID | E target-specific | 0.3487 ± 0.0134 | 1.0269 ± 0.0183 | 0.5796 ± 0.0062 | 0.0799 ± 0.0079 | 0.5607 ± 0.0088 | 0.4955 ± 0.0178 |
| TinyLlama-1.1B | OOD | A learned core | 0.1743 ± 0.0071 | 3.5446 ± 1.3515 | 0.9004 ± 0.0563 | 0.3130 ± 0.0416 | 0.5160 ± 0.0070 | 1.6760 ± 0.8242 |
| TinyLlama-1.1B | OOD | B random core | 0.1834 ± 0.0103 | 2.6311 ± 0.8689 | 0.8703 ± 0.0782 | 0.2884 ± 0.0656 | 0.5209 ± 0.0132 | 1.1216 ± 0.4100 |
| TinyLlama-1.1B | OOD | C mismatched | 0.1374 ± 0.0160 | 1.3308 ± 0.0518 | 0.7416 ± 0.0281 | 0.1438 ± 0.0381 | 0.3927 ± 0.0790 | 0.5769 ± 0.0920 |
| TinyLlama-1.1B | OOD | D untrained | 0.1520 ± 0.0172 | 1.3903 ± 0.1363 | 0.7631 ± 0.0586 | 0.1409 ± 0.0748 | 0.3492 ± 0.0555 | 0.6044 ± 0.0650 |
| TinyLlama-1.1B | OOD | E target-specific | 0.1773 ± 0.0064 | 2.8318 ± 0.6855 | 0.8973 ± 0.0156 | 0.3101 ± 0.0108 | 0.5263 ± 0.0086 | 1.1808 ± 0.3484 |

## Noul and Score detail (mean over seeds)

| Target | Split | Condition | Noul P(true) | Noul predicted-true rate | Noul train KL | Score ordinal MAE |
|---|---|---|---:|---:|---:|---:|
| Gemma 3 270M | ID | A learned core | 0.5827 | 0.9237 | 0.3329 | 1.0671 |
| Gemma 3 270M | ID | B random core | 0.6631 | 0.8083 | 0.5104 | 1.1403 |
| Gemma 3 270M | ID | C mismatched | 0.6208 | 1.0000 | 0.3499 | 1.5463 |
| Gemma 3 270M | ID | D untrained | 0.5458 | 0.5274 | 1.4849 | 1.8937 |
| Gemma 3 270M | ID | E target-specific | 0.6402 | 0.8237 | 0.3184 | 0.9567 |
| Gemma 3 270M | OOD | A learned core | 0.5376 | 0.8668 | 0.3329 | 1.2220 |
| Gemma 3 270M | OOD | B random core | 0.5695 | 0.5840 | 0.5104 | 1.2384 |
| Gemma 3 270M | OOD | C mismatched | 0.6327 | 1.0000 | 0.3499 | 1.4300 |
| Gemma 3 270M | OOD | D untrained | 0.5060 | 0.4220 | 1.4849 | 1.8788 |
| Gemma 3 270M | OOD | E target-specific | 0.5659 | 0.7016 | 0.3184 | 1.0636 |
| SmolLM2-360M | ID | A learned core | 0.5763 | 0.9329 | 0.3383 | 1.1222 |
| SmolLM2-360M | ID | B random core | 0.7905 | 0.8306 | 1.6587 | 1.0334 |
| SmolLM2-360M | ID | C mismatched | 0.6169 | 1.0000 | 0.3474 | 1.3918 |
| SmolLM2-360M | ID | D untrained | 0.9626 | 0.9997 | 1.4900 | 1.6619 |
| SmolLM2-360M | ID | E target-specific | 0.5823 | 0.7629 | 0.3204 | 0.9732 |
| SmolLM2-360M | OOD | A learned core | 0.6720 | 0.9920 | 0.3383 | 1.2168 |
| SmolLM2-360M | OOD | B random core | 0.7663 | 0.7748 | 1.6587 | 1.2064 |
| SmolLM2-360M | OOD | C mismatched | 0.6242 | 1.0000 | 0.3474 | 1.3076 |
| SmolLM2-360M | OOD | D untrained | 0.9471 | 1.0000 | 1.4900 | 1.6600 |
| SmolLM2-360M | OOD | E target-specific | 0.5990 | 0.7332 | 0.3204 | 1.1892 |
| TinyLlama-1.1B | ID | A learned core | 0.5826 | 0.6169 | 0.2914 | 0.7197 |
| TinyLlama-1.1B | ID | B random core | 0.6750 | 0.7277 | 0.5984 | 0.6877 |
| TinyLlama-1.1B | ID | C mismatched | 0.6295 | 0.9509 | 0.3624 | 1.5537 |
| TinyLlama-1.1B | ID | D untrained | 0.6284 | 0.6266 | 0.7436 | 1.6205 |
| TinyLlama-1.1B | ID | E target-specific | 0.6663 | 0.7466 | 0.2183 | 0.7014 |
| TinyLlama-1.1B | OOD | A learned core | 0.9596 | 0.9960 | 0.2914 | 0.8932 |
| TinyLlama-1.1B | OOD | B random core | 0.9165 | 0.9624 | 0.5984 | 0.8892 |
| TinyLlama-1.1B | OOD | C mismatched | 0.5919 | 0.7844 | 0.3624 | 1.4852 |
| TinyLlama-1.1B | OOD | D untrained | 0.6318 | 0.6700 | 0.7436 | 1.6100 |
| TinyLlama-1.1B | OOD | E target-specific | 0.9857 | 0.9996 | 0.2183 | 0.8648 |

## Cost per target

| Target | Condition | Trainable params | Artifact bytes | Train s | Inference s (ID+OOD, cached states) |
|---|---|---:|---:|---:|---:|
| Gemma 3 270M | A learned core | 345,344 | 1,381,704 | 27.5 | 0.15 |
| Gemma 3 270M | B random core | 345,344 | 1,381,704 | 28.8 | 0.14 |
| Gemma 3 270M | C mismatched | 345,344 | 1,381,704 | 27.7 | 0.15 |
| Gemma 3 270M | D untrained | 0 | 1,381,704 | 0.0 | 0.24 |
| Gemma 3 270M | E target-specific | 395,265 | 1,581,708 | 29.6 | 0.13 |
| SmolLM2-360M | A learned core | 386,944 | 1,548,104 | 30.2 | 0.19 |
| SmolLM2-360M | B random core | 386,944 | 1,548,104 | 30.6 | 0.15 |
| SmolLM2-360M | C mismatched | 386,944 | 1,548,104 | 31.5 | 0.16 |
| SmolLM2-360M | D untrained | 0 | 1,548,104 | 0.0 | 0.15 |
| SmolLM2-360M | E target-specific | 559,745 | 2,239,628 | 30.8 | 0.14 |
| TinyLlama-1.1B | A learned core | 528,384 | 2,113,872 | 36.3 | 0.24 |
| TinyLlama-1.1B | B random core | 528,384 | 2,113,872 | 35.0 | 0.24 |
| TinyLlama-1.1B | C mismatched | 528,384 | 2,113,872 | 35.2 | 0.23 |
| TinyLlama-1.1B | D untrained | 0 | 2,113,872 | 0.0 | 0.31 |
| TinyLlama-1.1B | E target-specific | 1,118,977 | 4,476,556 | 36.4 | 0.21 |

Shared learned core (stored once, not per target): 1,118,977 parameters.

## JevBench public subset (231/534)

JevBench public subset only: 231 of 534 tasks (72 original, 48 easy, 111 hard). The private and judge tiers are unavailable; this is not a full-benchmark result.

| System | Seed | Correct / 231 | Choice | Noul | Score | Brier | ECE | Agreement with source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma-learned_core_distillation | 3 | 73 | 38/139 | 33/74 | 2/18 | 0.7546 | 0.2143 | 112 |
| gemma-mismatched_teacher_distillation | 1 | 78 | 38/139 | 35/74 | 5/18 | 0.7032 | 0.0964 | 96 |
| gemma-random_core_distillation | 1 | 73 | 34/139 | 34/74 | 5/18 | 0.8155 | 0.3128 | 109 |
| gemma-target_specific_distillation | 1 | 70 | 36/139 | 34/74 | 0/18 | 0.8864 | 0.3294 | 84 |
| gemma-untrained_adapter | 0 | 73 | 33/139 | 35/74 | 5/18 | 0.9389 | 0.3136 | 90 |
| smollm-learned_core_distillation | 3 | 89 | 44/139 | 41/74 | 4/18 | 0.8273 | 0.2474 | 98 |
| smollm-mismatched_teacher_distillation | 0 | 85 | 43/139 | 35/74 | 7/18 | 0.7016 | 0.1143 | 92 |
| smollm-random_core_distillation | 3 | 87 | 47/139 | 36/74 | 4/18 | 0.9503 | 0.3907 | 75 |
| smollm-target_specific_distillation | 0 | 96 | 57/139 | 35/74 | 4/18 | 0.8344 | 0.3004 | 100 |
| smollm-untrained_adapter | 0 | 69 | 30/139 | 35/74 | 4/18 | 0.9431 | 0.3442 | 87 |
| source-system | — | 137 | 81/139 | 43/74 | 13/18 | 0.5861 | 0.2449 | — |
| tinyllama-learned_core_distillation | 1 | 74 | 38/139 | 32/74 | 4/18 | 1.0793 | 0.5100 | 99 |
| tinyllama-mismatched_teacher_distillation | 2 | 64 | 29/139 | 33/74 | 2/18 | 0.7143 | 0.1310 | 76 |
| tinyllama-random_core_distillation | 2 | 74 | 38/139 | 32/74 | 4/18 | 0.9885 | 0.4218 | 92 |
| tinyllama-target_specific_distillation | 3 | 65 | 31/139 | 32/74 | 2/18 | 1.1424 | 0.5349 | 91 |
| tinyllama-untrained_adapter | 0 | 83 | 43/139 | 33/74 | 7/18 | 0.7341 | 0.1993 | 77 |

Uniform-guess expectation: 73.37 / 231.

## Pre-registered criteria

| Criterion | Passing targets | Result |
|---|---|---|
| A_learned_core_matters | none | FAIL |
| B_input_specific | gemma, smollm, tinyllama | pass |
| C_useful_under_shift | gemma, smollm, tinyllama | pass |
| D_external_behavior | smollm | FAIL |
| E_practical_utility | tinyllama | FAIL |

Audit passed: True. Reproduction passed: True.

**FINAL VERDICT: NO-SHIP**
