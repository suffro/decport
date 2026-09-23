Seeds: [0, 1, 2, 3, 4] (mean ± sample SD when more than one seed).


### Overall accuracy and teacher agreement

| Target | Split | Teacher acc. | Correct acc. | Untrained | Mismatch | Random core | Agreement | Δ vs mismatch | Δ vs random core |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| gemma | ID | 0.7310 | 0.4659 ± 0.0284 | 0.3202 ± 0.0396 | 0.2806 ± 0.0249 | 0.4370 ± 0.0119 | 0.5185 ± 0.0161 | +0.1853 ± 0.0284 | +0.0289 ± 0.0319 |
| gemma | OOD | 0.6467 | 0.3799 ± 0.0135 | 0.3268 ± 0.0105 | 0.3115 ± 0.0160 | 0.3672 ± 0.0091 | 0.4560 ± 0.0285 | +0.0684 ± 0.0086 | +0.0127 ± 0.0202 |
| smollm | ID | 0.7310 | 0.4602 ± 0.0074 | 0.3176 ± 0.0449 | 0.2823 ± 0.0281 | 0.4588 ± 0.0082 | 0.5292 ± 0.0086 | +0.1779 ± 0.0266 | +0.0014 ± 0.0112 |
| smollm | OOD | 0.6467 | 0.3755 ± 0.0093 | 0.3219 ± 0.0119 | 0.3112 ± 0.0200 | 0.3716 ± 0.0047 | 0.4289 ± 0.0075 | +0.0643 ± 0.0206 | +0.0039 ± 0.0054 |
| tinyllama | ID | 0.7310 | 0.4478 ± 0.0023 | 0.3190 ± 0.0311 | 0.2727 ± 0.0140 | 0.4453 ± 0.0029 | 0.5267 ± 0.0054 | +0.1751 ± 0.0141 | +0.0025 ± 0.0033 |
| tinyllama | OOD | 0.6467 | 0.3929 ± 0.0048 | 0.3044 ± 0.0095 | 0.3208 ± 0.0049 | 0.3875 ± 0.0085 | 0.4753 ± 0.0117 | +0.0721 ± 0.0024 | +0.0055 ± 0.0060 |

### By decision type — ID

| Target | Type | Teacher acc. | Correct acc. | Agreement | Δ acc. vs untrained | Δ acc. vs mismatch | Δ agreement vs mismatch | Δ acc. vs random core |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gemma | choice | 0.8877 | 0.4596 ± 0.0135 | 0.4582 ± 0.0121 | +0.2007 ± 0.0469 | +0.2267 ± 0.0412 | +0.2246 ± 0.0384 | +0.0207 ± 0.0244 |
| gemma | noul | 0.7929 | 0.5066 ± 0.0669 | 0.5531 ± 0.0123 | +0.0203 ± 0.1088 | +0.1111 ± 0.0774 | +0.0666 ± 0.0047 | +0.0046 ± 0.0615 |
| gemma | score | 0.5493 | 0.4318 ± 0.0176 | 0.5323 ± 0.0418 | +0.2230 ± 0.0213 | +0.2241 ± 0.0232 | +0.2411 ± 0.0526 | +0.0586 ± 0.0328 |
| smollm | choice | 0.8877 | 0.6446 ± 0.0118 | 0.6309 ± 0.0128 | +0.3860 ± 0.0360 | +0.4193 ± 0.0501 | +0.4102 ± 0.0565 | +0.0032 ± 0.0201 |
| smollm | noul | 0.7929 | 0.3631 ± 0.0056 | 0.5091 ± 0.0037 | -0.1094 ± 0.1545 | -0.0217 ± 0.0478 | -0.0074 ± 0.0175 | -0.0049 ± 0.0162 |
| smollm | score | 0.5493 | 0.4093 ± 0.0145 | 0.4690 ± 0.0144 | +0.1942 ± 0.0277 | +0.1808 ± 0.0190 | +0.1595 ± 0.0269 | +0.0060 ± 0.0103 |
| tinyllama | choice | 0.8877 | 0.4881 ± 0.0052 | 0.4951 ± 0.0068 | +0.2596 ± 0.0189 | +0.2418 ± 0.0264 | +0.2404 ± 0.0261 | +0.0074 ± 0.0106 |
| tinyllama | noul | 0.7929 | 0.3606 ± 0.0022 | 0.5083 ± 0.0071 | -0.1734 ± 0.1114 | -0.0011 ± 0.0019 | -0.0034 ± 0.0069 | -0.0009 ± 0.0022 |
| tinyllama | score | 0.5493 | 0.5000 ± 0.0078 | 0.5690 ± 0.0074 | +0.3164 ± 0.0351 | +0.2921 ± 0.0214 | +0.3016 ± 0.0792 | +0.0019 ± 0.0059 |

### By decision type — OOD

| Target | Type | Teacher acc. | Correct acc. | Agreement | Δ acc. vs untrained | Δ acc. vs mismatch | Δ agreement vs mismatch | Δ acc. vs random core |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gemma | choice | 0.6080 | 0.2580 ± 0.0147 | 0.2976 ± 0.0103 | -0.0160 ± 0.0452 | -0.0028 ± 0.0303 | +0.0464 ± 0.0353 | +0.0012 ± 0.0117 |
| gemma | noul | 0.8600 | 0.5340 ± 0.0224 | 0.5556 ± 0.0220 | +0.0316 ± 0.0333 | +0.0544 ± 0.0281 | +0.0704 ± 0.0464 | +0.0080 ± 0.0249 |
| gemma | score | 0.4720 | 0.3476 ± 0.0293 | 0.5148 ± 0.0705 | +0.1436 ± 0.0317 | +0.1536 ± 0.0318 | +0.2168 ± 0.0706 | +0.0288 ± 0.0416 |
| smollm | choice | 0.6080 | 0.3136 ± 0.0111 | 0.3780 ± 0.0063 | +0.0452 ± 0.0452 | +0.0840 ± 0.0339 | +0.1288 ± 0.0195 | +0.0056 ± 0.0080 |
| smollm | noul | 0.8600 | 0.5176 ± 0.0132 | 0.5336 ± 0.0166 | +0.0128 ± 0.0335 | +0.0080 ± 0.0179 | +0.0080 ± 0.0275 | -0.0064 ± 0.0132 |
| smollm | score | 0.4720 | 0.2952 ± 0.0186 | 0.3752 ± 0.0132 | +0.1028 ± 0.0274 | +0.1008 ± 0.0218 | +0.0552 ± 0.0140 | +0.0124 ± 0.0161 |
| tinyllama | choice | 0.6080 | 0.3028 ± 0.0095 | 0.3524 ± 0.0067 | +0.0628 ± 0.0169 | +0.0528 ± 0.0132 | +0.0976 ± 0.0250 | +0.0008 ± 0.0082 |
| tinyllama | noul | 0.8600 | 0.5052 ± 0.0076 | 0.5092 ± 0.0237 | +0.0136 ± 0.0194 | -0.0184 ± 0.0067 | -0.0336 ± 0.0214 | +0.0104 ± 0.0209 |
| tinyllama | score | 0.4720 | 0.3708 ± 0.0064 | 0.5644 ± 0.0248 | +0.1892 ± 0.0124 | +0.1820 ± 0.0133 | +0.2976 ± 0.0542 | +0.0052 ± 0.0151 |

### Accuracy by decision type and condition

| Target | Split | Type | Teacher | untrained adapter | Open-Jev teacher | mismatched teacher | random frozen core |
|---|---|---|---:|---:|---:|---:|---:|
| gemma | ID | choice | 0.8877 | 0.2589 ± 0.0344 | 0.4596 ± 0.0135 | 0.2330 ± 0.0426 | 0.4389 ± 0.0170 |
| gemma | ID | noul | 0.7929 | 0.4863 ± 0.1219 | 0.5066 ± 0.0669 | 0.3954 ± 0.0347 | 0.5020 ± 0.0199 |
| gemma | ID | score | 0.5493 | 0.2088 ± 0.0162 | 0.4318 ± 0.0176 | 0.2077 ± 0.0206 | 0.3732 ± 0.0205 |
| gemma | OOD | choice | 0.6080 | 0.2740 ± 0.0359 | 0.2580 ± 0.0147 | 0.2608 ± 0.0296 | 0.2568 ± 0.0044 |
| gemma | OOD | noul | 0.8600 | 0.5024 ± 0.0288 | 0.5340 ± 0.0224 | 0.4796 ± 0.0387 | 0.5260 ± 0.0032 |
| gemma | OOD | score | 0.4720 | 0.2040 ± 0.0088 | 0.3476 ± 0.0293 | 0.1940 ± 0.0079 | 0.3188 ± 0.0287 |
| smollm | ID | choice | 0.8877 | 0.2586 ± 0.0404 | 0.6446 ± 0.0118 | 0.2253 ± 0.0588 | 0.6414 ± 0.0149 |
| smollm | ID | noul | 0.7929 | 0.4726 ± 0.1515 | 0.3631 ± 0.0056 | 0.3849 ± 0.0532 | 0.3680 ± 0.0140 |
| smollm | ID | score | 0.5493 | 0.2151 ± 0.0229 | 0.4093 ± 0.0145 | 0.2285 ± 0.0121 | 0.4033 ± 0.0087 |
| smollm | OOD | choice | 0.6080 | 0.2684 ± 0.0415 | 0.3136 ± 0.0111 | 0.2296 ± 0.0271 | 0.3080 ± 0.0097 |
| smollm | OOD | noul | 0.8600 | 0.5048 ± 0.0263 | 0.5176 ± 0.0132 | 0.5096 ± 0.0311 | 0.5240 ± 0.0000 |
| smollm | OOD | score | 0.4720 | 0.1924 ± 0.0156 | 0.2952 ± 0.0186 | 0.1944 ± 0.0065 | 0.2828 ± 0.0084 |
| tinyllama | ID | choice | 0.8877 | 0.2284 ± 0.0158 | 0.4881 ± 0.0052 | 0.2463 ± 0.0313 | 0.4807 ± 0.0135 |
| tinyllama | ID | noul | 0.7929 | 0.5340 ± 0.1099 | 0.3606 ± 0.0022 | 0.3617 ± 0.0012 | 0.3614 ± 0.0000 |
| tinyllama | ID | score | 0.5493 | 0.1836 ± 0.0296 | 0.5000 ± 0.0078 | 0.2079 ± 0.0156 | 0.4981 ± 0.0076 |
| tinyllama | OOD | choice | 0.6080 | 0.2400 ± 0.0123 | 0.3028 ± 0.0095 | 0.2500 ± 0.0206 | 0.3020 ± 0.0117 |
| tinyllama | OOD | noul | 0.8600 | 0.4916 ± 0.0205 | 0.5052 ± 0.0076 | 0.5236 ± 0.0009 | 0.4948 ± 0.0198 |
| tinyllama | OOD | score | 0.4720 | 0.1816 ± 0.0131 | 0.3708 ± 0.0064 | 0.1888 ± 0.0098 | 0.3656 ± 0.0129 |

### Noul behavior

| Target | Split | Condition | Accuracy | Mean P(true) | Predicted true rate | Label true rate | KL to teacher |
|---|---|---|---:|---:|---:|---:|---:|
| teacher | ID | Open-Jev 2B | 0.7929 | 0.4934 | 0.4886 | 0.6386 | — |
| teacher | OOD | Open-Jev 2B | 0.8600 | 0.4419 | 0.4560 | 0.4760 | — |
| gemma | ID | untrained adapter | 0.4863 ± 0.1219 | 0.4990 ± 0.0172 | 0.4420 ± 0.4578 | 0.6386 ± 0.0000 | 0.2758 ± 0.0007 |
| gemma | ID | Open-Jev teacher | 0.5066 ± 0.0669 | 0.4934 ± 0.0456 | 0.3994 ± 0.1936 | 0.6386 ± 0.0000 | 0.2748 ± 0.0038 |
| gemma | ID | mismatched teacher | 0.3954 ± 0.0347 | 0.4675 ± 0.0255 | 0.1563 ± 0.1288 | 0.6386 ± 0.0000 | 0.2897 ± 0.0190 |
| gemma | ID | random frozen core | 0.5020 ± 0.0199 | 0.4472 ± 0.0410 | 0.3726 ± 0.1020 | 0.6386 ± 0.0000 | 0.2766 ± 0.0142 |
| gemma | OOD | untrained adapter | 0.5024 ± 0.0288 | 0.4991 ± 0.0166 | 0.5192 ± 0.4648 | 0.4760 ± 0.0000 | 0.2372 ± 0.0037 |
| gemma | OOD | Open-Jev teacher | 0.5340 ± 0.0224 | 0.4384 ± 0.0539 | 0.1300 ± 0.1677 | 0.4760 ± 0.0000 | 0.2316 ± 0.0066 |
| gemma | OOD | mismatched teacher | 0.4796 ± 0.0387 | 0.4847 ± 0.0487 | 0.3884 ± 0.4432 | 0.4760 ± 0.0000 | 0.2417 ± 0.0111 |
| gemma | OOD | random frozen core | 0.5260 ± 0.0032 | 0.3313 ± 0.0759 | 0.0236 ± 0.0246 | 0.4760 ± 0.0000 | 0.2662 ± 0.0564 |
| smollm | ID | untrained adapter | 0.4726 ± 0.1515 | 0.4944 ± 0.0388 | 0.4014 ± 0.5464 | 0.6386 ± 0.0000 | 0.2779 ± 0.0020 |
| smollm | ID | Open-Jev teacher | 0.3631 ± 0.0056 | 0.4779 ± 0.0063 | 0.0120 ± 0.0253 | 0.6386 ± 0.0000 | 0.2751 ± 0.0016 |
| smollm | ID | mismatched teacher | 0.3849 ± 0.0532 | 0.4804 ± 0.0210 | 0.0754 ± 0.1647 | 0.6386 ± 0.0000 | 0.2847 ± 0.0178 |
| smollm | ID | random frozen core | 0.3680 ± 0.0140 | 0.4362 ± 0.0769 | 0.0106 ± 0.0184 | 0.6386 ± 0.0000 | 0.2902 ± 0.0320 |
| smollm | OOD | untrained adapter | 0.5048 ± 0.0263 | 0.4919 ± 0.0361 | 0.4000 ± 0.5477 | 0.4760 ± 0.0000 | 0.2374 ± 0.0075 |
| smollm | OOD | Open-Jev teacher | 0.5176 ± 0.0132 | 0.4345 ± 0.0388 | 0.0936 ± 0.1602 | 0.4760 ± 0.0000 | 0.2378 ± 0.0056 |
| smollm | OOD | mismatched teacher | 0.5096 ± 0.0311 | 0.4910 ± 0.0421 | 0.1848 ± 0.3967 | 0.4760 ± 0.0000 | 0.2393 ± 0.0148 |
| smollm | OOD | random frozen core | 0.5240 ± 0.0000 | 0.3601 ± 0.1504 | 0.0000 ± 0.0000 | 0.4760 ± 0.0000 | 0.3150 ± 0.1772 |
| tinyllama | ID | untrained adapter | 0.5340 ± 0.1099 | 0.5034 ± 0.0177 | 0.6257 ± 0.4032 | 0.6386 ± 0.0000 | 0.2756 ± 0.0010 |
| tinyllama | ID | Open-Jev teacher | 0.3606 ± 0.0022 | 0.4804 ± 0.0133 | 0.0089 ± 0.0167 | 0.6386 ± 0.0000 | 0.2764 ± 0.0005 |
| tinyllama | ID | mismatched teacher | 0.3617 ± 0.0012 | 0.4701 ± 0.0182 | 0.0020 ± 0.0030 | 0.6386 ± 0.0000 | 0.2773 ± 0.0016 |
| tinyllama | ID | random frozen core | 0.3614 ± 0.0000 | 0.4767 ± 0.0103 | 0.0000 ± 0.0000 | 0.6386 ± 0.0000 | 0.2761 ± 0.0006 |
| tinyllama | OOD | untrained adapter | 0.4916 ± 0.0205 | 0.5037 ± 0.0155 | 0.6628 ± 0.4324 | 0.4760 ± 0.0000 | 0.2382 ± 0.0037 |
| tinyllama | OOD | Open-Jev teacher | 0.5052 ± 0.0076 | 0.5431 ± 0.2090 | 0.3924 ± 0.3435 | 0.4760 ± 0.0000 | 0.6930 ± 0.9728 |
| tinyllama | OOD | mismatched teacher | 0.5236 ± 0.0009 | 0.4654 ± 0.0211 | 0.0132 ± 0.0295 | 0.4760 ± 0.0000 | 0.2317 ± 0.0028 |
| tinyllama | OOD | random frozen core | 0.4948 ± 0.0198 | 0.4986 ± 0.0575 | 0.2556 ± 0.1731 | 0.4760 ± 0.0000 | 0.3605 ± 0.1442 |

### Teacher matching by decision type

| Target | Split | Type | Condition | Top-choice agreement | KL to teacher | Probability corr. |
|---|---|---|---:|---:|---:|---:|
| gemma | ID | choice | untrained adapter | 0.2533 ± 0.0333 | 0.4779 ± 0.0003 | 0.0225 ± 0.0135 |
| gemma | ID | choice | Open-Jev teacher | 0.4582 ± 0.0121 | 0.4166 ± 0.0123 | 0.3776 ± 0.0128 |
| gemma | ID | choice | mismatched teacher | 0.2337 ± 0.0364 | 0.4917 ± 0.0175 | -0.0353 ± 0.0476 |
| gemma | ID | choice | random frozen core | 0.4414 ± 0.0113 | 0.4230 ± 0.0087 | 0.3669 ± 0.0079 |
| gemma | ID | noul | untrained adapter | 0.5020 ± 0.0147 | 0.2758 ± 0.0007 | 0.0134 ± 0.0426 |
| gemma | ID | noul | Open-Jev teacher | 0.5531 ± 0.0123 | 0.2748 ± 0.0038 | 0.1610 ± 0.0531 |
| gemma | ID | noul | mismatched teacher | 0.4866 ± 0.0112 | 0.2897 ± 0.0190 | -0.0330 ± 0.0497 |
| gemma | ID | noul | random frozen core | 0.5623 ± 0.0110 | 0.2766 ± 0.0142 | 0.2225 ± 0.0166 |
| gemma | ID | score | untrained adapter | 0.2244 ± 0.0880 | 0.3905 ± 0.0003 | 0.0120 ± 0.1085 |
| gemma | ID | score | Open-Jev teacher | 0.5323 ± 0.0418 | 0.2338 ± 0.0218 | 0.6116 ± 0.0404 |
| gemma | ID | score | mismatched teacher | 0.2912 ± 0.0382 | 0.3813 ± 0.0063 | 0.1370 ± 0.0749 |
| gemma | ID | score | random frozen core | 0.4556 ± 0.0294 | 0.2904 ± 0.0212 | 0.5091 ± 0.0454 |
| gemma | OOD | choice | untrained adapter | 0.2676 ± 0.0360 | 0.2085 ± 0.0003 | 0.0081 ± 0.0680 |
| gemma | OOD | choice | Open-Jev teacher | 0.2976 ± 0.0103 | 0.3383 ± 0.0357 | 0.1252 ± 0.0125 |
| gemma | OOD | choice | mismatched teacher | 0.2512 ± 0.0268 | 0.2221 ± 0.0214 | 0.0019 ± 0.0623 |
| gemma | OOD | choice | random frozen core | 0.3044 ± 0.0071 | 0.3454 ± 0.0204 | 0.1330 ± 0.0060 |
| gemma | OOD | noul | untrained adapter | 0.4952 ± 0.0463 | 0.2372 ± 0.0037 | 0.0022 ± 0.1712 |
| gemma | OOD | noul | Open-Jev teacher | 0.5556 ± 0.0220 | 0.2316 ± 0.0066 | 0.2078 ± 0.0350 |
| gemma | OOD | noul | mismatched teacher | 0.4852 ± 0.0558 | 0.2417 ± 0.0111 | -0.0250 ± 0.1985 |
| gemma | OOD | noul | random frozen core | 0.5508 ± 0.0054 | 0.2662 ± 0.0564 | 0.2296 ± 0.0219 |
| gemma | OOD | score | untrained adapter | 0.2560 ± 0.1297 | 0.2976 ± 0.0002 | 0.0189 ± 0.0997 |
| gemma | OOD | score | Open-Jev teacher | 0.5148 ± 0.0705 | 0.2093 ± 0.0268 | 0.5291 ± 0.0793 |
| gemma | OOD | score | mismatched teacher | 0.2980 ± 0.0146 | 0.2904 ± 0.0034 | 0.1413 ± 0.0474 |
| gemma | OOD | score | random frozen core | 0.4300 ± 0.0416 | 0.2456 ± 0.0168 | 0.4086 ± 0.0652 |
| smollm | ID | choice | untrained adapter | 0.2681 ± 0.0369 | 0.4778 ± 0.0003 | 0.0279 ± 0.0172 |
| smollm | ID | choice | Open-Jev teacher | 0.6309 ± 0.0128 | 0.2953 ± 0.0024 | 0.6280 ± 0.0050 |
| smollm | ID | choice | mismatched teacher | 0.2207 ± 0.0613 | 0.4809 ± 0.0052 | -0.0128 ± 0.0889 |
| smollm | ID | choice | random frozen core | 0.6284 ± 0.0118 | 0.2997 ± 0.0080 | 0.6254 ± 0.0099 |
| smollm | ID | noul | untrained adapter | 0.5020 ± 0.0123 | 0.2779 ± 0.0020 | -0.0005 ± 0.0249 |
| smollm | ID | noul | Open-Jev teacher | 0.5091 ± 0.0037 | 0.2751 ± 0.0016 | 0.0559 ± 0.0371 |
| smollm | ID | noul | mismatched teacher | 0.5166 ± 0.0139 | 0.2847 ± 0.0178 | 0.0139 ± 0.0092 |
| smollm | ID | noul | random frozen core | 0.5151 ± 0.0108 | 0.2902 ± 0.0320 | 0.0624 ± 0.0495 |
| smollm | ID | score | untrained adapter | 0.2238 ± 0.0643 | 0.3905 ± 0.0002 | -0.0040 ± 0.0648 |
| smollm | ID | score | Open-Jev teacher | 0.4690 ± 0.0144 | 0.2742 ± 0.0074 | 0.5396 ± 0.0177 |
| smollm | ID | score | mismatched teacher | 0.3096 ± 0.0193 | 0.3843 ± 0.0039 | 0.1259 ± 0.0266 |
| smollm | ID | score | random frozen core | 0.4732 ± 0.0268 | 0.2760 ± 0.0056 | 0.5373 ± 0.0148 |
| smollm | OOD | choice | untrained adapter | 0.2656 ± 0.0366 | 0.2084 ± 0.0002 | 0.0305 ± 0.0515 |
| smollm | OOD | choice | Open-Jev teacher | 0.3780 ± 0.0063 | 0.3109 ± 0.0134 | 0.2895 ± 0.0073 |
| smollm | OOD | choice | mismatched teacher | 0.2492 ± 0.0190 | 0.2147 ± 0.0071 | -0.0255 ± 0.0298 |
| smollm | OOD | choice | random frozen core | 0.3764 ± 0.0116 | 0.3222 ± 0.0263 | 0.2890 ± 0.0087 |
| smollm | OOD | noul | untrained adapter | 0.5088 ± 0.0482 | 0.2374 ± 0.0075 | 0.0316 ± 0.1961 |
| smollm | OOD | noul | Open-Jev teacher | 0.5336 ± 0.0166 | 0.2378 ± 0.0056 | 0.1281 ± 0.0764 |
| smollm | OOD | noul | mismatched teacher | 0.5256 ± 0.0423 | 0.2393 ± 0.0148 | 0.1064 ± 0.1613 |
| smollm | OOD | noul | random frozen core | 0.5440 ± 0.0000 | 0.3150 ± 0.1772 | 0.1798 ± 0.0102 |
| smollm | OOD | score | untrained adapter | 0.2660 ± 0.0902 | 0.2976 ± 0.0002 | 0.0205 ± 0.0752 |
| smollm | OOD | score | Open-Jev teacher | 0.3752 ± 0.0132 | 0.2643 ± 0.0111 | 0.3772 ± 0.0226 |
| smollm | OOD | score | mismatched teacher | 0.3200 ± 0.0124 | 0.2930 ± 0.0041 | 0.1434 ± 0.0289 |
| smollm | OOD | score | random frozen core | 0.3728 ± 0.0166 | 0.2665 ± 0.0117 | 0.3667 ± 0.0324 |
| tinyllama | ID | choice | untrained adapter | 0.2323 ± 0.0144 | 0.4784 ± 0.0001 | 0.0020 ± 0.0052 |
| tinyllama | ID | choice | Open-Jev teacher | 0.4951 ± 0.0068 | 0.4753 ± 0.0246 | 0.4260 ± 0.0228 |
| tinyllama | ID | choice | mismatched teacher | 0.2547 ± 0.0321 | 0.4960 ± 0.0309 | 0.0075 ± 0.0339 |
| tinyllama | ID | choice | random frozen core | 0.4867 ± 0.0145 | 0.4770 ± 0.0242 | 0.4255 ± 0.0177 |
| tinyllama | ID | noul | untrained adapter | 0.5011 ± 0.0156 | 0.2756 ± 0.0010 | 0.0284 ± 0.0398 |
| tinyllama | ID | noul | Open-Jev teacher | 0.5083 ± 0.0071 | 0.2764 ± 0.0005 | 0.0060 ± 0.0277 |
| tinyllama | ID | noul | mismatched teacher | 0.5117 ± 0.0012 | 0.2773 ± 0.0016 | 0.0173 ± 0.0036 |
| tinyllama | ID | noul | random frozen core | 0.5114 ± 0.0000 | 0.2761 ± 0.0006 | 0.0238 ± 0.0194 |
| tinyllama | ID | score | untrained adapter | 0.1945 ± 0.0703 | 0.3906 ± 0.0003 | -0.0428 ± 0.1277 |
| tinyllama | ID | score | Open-Jev teacher | 0.5690 ± 0.0074 | 0.2081 ± 0.0015 | 0.6894 ± 0.0044 |
| tinyllama | ID | score | mismatched teacher | 0.2674 ± 0.0808 | 0.3845 ± 0.0064 | 0.1011 ± 0.0904 |
| tinyllama | ID | score | random frozen core | 0.5641 ± 0.0123 | 0.2088 ± 0.0053 | 0.6881 ± 0.0114 |
| tinyllama | OOD | choice | untrained adapter | 0.2396 ± 0.0246 | 0.2088 ± 0.0003 | -0.0274 ± 0.0373 |
| tinyllama | OOD | choice | Open-Jev teacher | 0.3524 ± 0.0067 | 0.5304 ± 0.0383 | 0.2278 ± 0.0075 |
| tinyllama | OOD | choice | mismatched teacher | 0.2548 ± 0.0235 | 0.2284 ± 0.0259 | -0.0010 ± 0.0562 |
| tinyllama | OOD | choice | random frozen core | 0.3516 ± 0.0091 | 0.5299 ± 0.0406 | 0.2284 ± 0.0141 |
| tinyllama | OOD | noul | untrained adapter | 0.4852 ± 0.0382 | 0.2382 ± 0.0037 | -0.0473 ± 0.1541 |
| tinyllama | OOD | noul | Open-Jev teacher | 0.5092 ± 0.0237 | 0.6930 ± 0.9728 | 0.0602 ± 0.1198 |
| tinyllama | OOD | noul | mismatched teacher | 0.5428 ± 0.0027 | 0.2317 ± 0.0028 | 0.1796 ± 0.0202 |
| tinyllama | OOD | noul | random frozen core | 0.5036 ± 0.0193 | 0.3605 ± 0.1442 | 0.0078 ± 0.0302 |
| tinyllama | OOD | score | untrained adapter | 0.1788 ± 0.1055 | 0.2978 ± 0.0003 | -0.0659 ± 0.1273 |
| tinyllama | OOD | score | Open-Jev teacher | 0.5644 ± 0.0248 | 0.1744 ± 0.0018 | 0.6491 ± 0.0066 |
| tinyllama | OOD | score | mismatched teacher | 0.2668 ± 0.0714 | 0.2922 ± 0.0054 | 0.0975 ± 0.1042 |
| tinyllama | OOD | score | random frozen core | 0.5636 ± 0.0186 | 0.1754 ± 0.0047 | 0.6449 ± 0.0152 |

### Score ordinal MAE (lower is better)

| Target | Split | Condition | Ordinal MAE | Expected-value MAE |
|---|---|---|---:|---:|
| teacher | ID | Open-Jev 2B | 0.5740 | 0.6417 |
| teacher | OOD | Open-Jev 2B | 0.7520 | 0.7078 |
| gemma | ID | untrained adapter | 1.7490 ± 0.1831 | 1.2743 ± 0.0001 |
| gemma | ID | Open-Jev teacher | 0.9304 ± 0.0684 | 0.9298 ± 0.0774 |
| gemma | ID | mismatched teacher | 2.0636 ± 0.0673 | 1.2885 ± 0.0018 |
| gemma | ID | random frozen core | 1.1208 ± 0.1348 | 1.1119 ± 0.0682 |
| gemma | OOD | untrained adapter | 1.7304 ± 0.2253 | 1.2064 ± 0.0001 |
| gemma | OOD | Open-Jev teacher | 1.2012 ± 0.1137 | 1.0040 ± 0.0570 |
| gemma | OOD | mismatched teacher | 2.0232 ± 0.0226 | 1.2213 ± 0.0036 |
| gemma | OOD | random frozen core | 1.3012 ± 0.1334 | 1.1153 ± 0.0452 |
| smollm | ID | untrained adapter | 1.6874 ± 0.1650 | 1.2742 ± 0.0001 |
| smollm | ID | Open-Jev teacher | 0.9729 ± 0.0723 | 1.0361 ± 0.0223 |
| smollm | ID | mismatched teacher | 1.9362 ± 0.0499 | 1.2853 ± 0.0100 |
| smollm | ID | random frozen core | 0.9942 ± 0.0405 | 1.0484 ± 0.0230 |
| smollm | OOD | untrained adapter | 1.7008 ± 0.1954 | 1.2064 ± 0.0001 |
| smollm | OOD | Open-Jev teacher | 1.3440 ± 0.0812 | 1.1043 ± 0.0199 |
| smollm | OOD | mismatched teacher | 1.9768 ± 0.0540 | 1.2209 ± 0.0136 |
| smollm | OOD | random frozen core | 1.3848 ± 0.0497 | 1.1123 ± 0.0182 |
| tinyllama | ID | untrained adapter | 1.5701 ± 0.2520 | 1.2742 ± 0.0002 |
| tinyllama | ID | Open-Jev teacher | 0.7049 ± 0.0321 | 0.9272 ± 0.0180 |
| tinyllama | ID | mismatched teacher | 1.8710 ± 0.2755 | 1.2842 ± 0.0116 |
| tinyllama | ID | random frozen core | 0.7115 ± 0.0311 | 0.9267 ± 0.0195 |
| tinyllama | OOD | untrained adapter | 1.5292 ± 0.3188 | 1.2063 ± 0.0001 |
| tinyllama | OOD | Open-Jev teacher | 1.0108 ± 0.0323 | 0.9762 ± 0.0071 |
| tinyllama | OOD | mismatched teacher | 1.8896 ± 0.2903 | 1.2174 ± 0.0109 |
| tinyllama | OOD | random frozen core | 1.0236 ± 0.0471 | 0.9772 ± 0.0111 |

### Seed consistency (seeds where the correct teacher wins on accuracy)

| Target | Split | Group | > untrained | > mismatch | > random core |
|---|---|---|---:|---:|---:|
| gemma | ID | overall | 5/5 | 5/5 | 4/5 |
| gemma | ID | choice | 5/5 | 5/5 | 4/5 |
| gemma | ID | noul | 4/5 | 5/5 | 2/5 |
| gemma | ID | score | 5/5 | 5/5 | 5/5 |
| gemma | OOD | overall | 5/5 | 5/5 | 3/5 |
| gemma | OOD | choice | 1/5 | 3/5 | 1/5 |
| gemma | OOD | noul | 4/5 | 5/5 | 2/5 |
| gemma | OOD | score | 5/5 | 5/5 | 3/5 |
| smollm | ID | overall | 5/5 | 5/5 | 3/5 |
| smollm | ID | choice | 5/5 | 5/5 | 3/5 |
| smollm | ID | noul | 1/5 | 1/5 | 1/5 |
| smollm | ID | score | 5/5 | 5/5 | 4/5 |
| smollm | OOD | overall | 5/5 | 5/5 | 4/5 |
| smollm | OOD | choice | 5/5 | 5/5 | 5/5 |
| smollm | OOD | noul | 2/5 | 2/5 | 0/5 |
| smollm | OOD | score | 5/5 | 5/5 | 3/5 |
| tinyllama | ID | overall | 5/5 | 5/5 | 4/5 |
| tinyllama | ID | choice | 5/5 | 5/5 | 4/5 |
| tinyllama | ID | noul | 0/5 | 0/5 | 1/5 |
| tinyllama | ID | score | 5/5 | 5/5 | 2/5 |
| tinyllama | OOD | overall | 5/5 | 5/5 | 5/5 |
| tinyllama | OOD | choice | 5/5 | 5/5 | 3/5 |
| tinyllama | OOD | noul | 4/5 | 0/5 | 3/5 |
| tinyllama | OOD | score | 5/5 | 5/5 | 2/5 |

### Calibration (overall)

| Target | Split | Condition | NLL | Brier | ECE | KL to teacher |
|---|---|---|---:|---:|---:|---:|
| teacher | ID | Open-Jev 2B | 0.6912 | 0.3752 | 0.0808 | — |
| teacher | OOD | Open-Jev 2B | 0.8465 | 0.4569 | 0.0747 | — |
| gemma | ID | untrained adapter | 1.2255 ± 0.0032 | 0.6811 ± 0.0032 | 0.0477 ± 0.0197 | 0.3753 ± 0.0003 |
| gemma | ID | Open-Jev teacher | 1.0883 ± 0.0266 | 0.6201 ± 0.0153 | 0.0656 ± 0.0151 | 0.3003 ± 0.0102 |
| gemma | ID | mismatched teacher | 1.2391 ± 0.0183 | 0.6926 ± 0.0130 | 0.0767 ± 0.0363 | 0.3807 ± 0.0138 |
| gemma | ID | random frozen core | 1.1462 ± 0.0206 | 0.6496 ± 0.0129 | 0.0721 ± 0.0119 | 0.3234 ± 0.0099 |
| gemma | OOD | untrained adapter | 1.2296 ± 0.0004 | 0.6834 ± 0.0005 | 0.0188 ± 0.0148 | 0.2478 ± 0.0012 |
| gemma | OOD | Open-Jev teacher | 1.2483 ± 0.0310 | 0.6936 ± 0.0143 | 0.0749 ± 0.0274 | 0.2597 ± 0.0220 |
| gemma | OOD | mismatched teacher | 1.2380 ± 0.0046 | 0.6889 ± 0.0023 | 0.0434 ± 0.0090 | 0.2514 ± 0.0076 |
| gemma | OOD | random frozen core | 1.2909 ± 0.0323 | 0.7171 ± 0.0204 | 0.1103 ± 0.0289 | 0.2857 ± 0.0260 |
| smollm | ID | untrained adapter | 1.2271 ± 0.0076 | 0.6827 ± 0.0077 | 0.0599 ± 0.0231 | 0.3760 ± 0.0007 |
| smollm | ID | Open-Jev teacher | 1.0390 ± 0.0104 | 0.5873 ± 0.0060 | 0.0943 ± 0.0059 | 0.2805 ± 0.0027 |
| smollm | ID | mismatched teacher | 1.2340 ± 0.0080 | 0.6871 ± 0.0043 | 0.0726 ± 0.0172 | 0.3770 ± 0.0071 |
| smollm | ID | random frozen core | 1.0582 ± 0.0268 | 0.6018 ± 0.0236 | 0.1025 ± 0.0173 | 0.2877 ± 0.0122 |
| smollm | OOD | untrained adapter | 1.2301 ± 0.0010 | 0.6838 ± 0.0010 | 0.0191 ± 0.0057 | 0.2478 ± 0.0026 |
| smollm | OOD | Open-Jev teacher | 1.2383 ± 0.0113 | 0.6889 ± 0.0058 | 0.0779 ± 0.0191 | 0.2710 ± 0.0068 |
| smollm | OOD | mismatched teacher | 1.2394 ± 0.0126 | 0.6886 ± 0.0081 | 0.0371 ± 0.0324 | 0.2490 ± 0.0080 |
| smollm | OOD | random frozen core | 1.2758 ± 0.0783 | 0.7093 ± 0.0449 | 0.1081 ± 0.0600 | 0.3012 ± 0.0668 |
| tinyllama | ID | untrained adapter | 1.2248 ± 0.0032 | 0.6802 ± 0.0032 | 0.0466 ± 0.0162 | 0.3754 ± 0.0004 |
| tinyllama | ID | Open-Jev teacher | 1.0583 ± 0.0055 | 0.6030 ± 0.0040 | 0.1158 ± 0.0054 | 0.3082 ± 0.0066 |
| tinyllama | ID | mismatched teacher | 1.2373 ± 0.0127 | 0.6902 ± 0.0074 | 0.0793 ± 0.0219 | 0.3787 ± 0.0091 |
| tinyllama | ID | random frozen core | 1.0593 ± 0.0045 | 0.6037 ± 0.0037 | 0.1103 ± 0.0103 | 0.3088 ± 0.0058 |
| tinyllama | OOD | untrained adapter | 1.2301 ± 0.0005 | 0.6837 ± 0.0005 | 0.0184 ± 0.0078 | 0.2483 ± 0.0012 |
| tinyllama | OOD | Open-Jev teacher | 1.4448 ± 0.3090 | 0.7355 ± 0.0630 | 0.1304 ± 0.0607 | 0.4659 ± 0.3239 |
| tinyllama | OOD | mismatched teacher | 1.2404 ± 0.0145 | 0.6881 ± 0.0065 | 0.0330 ± 0.0225 | 0.2508 ± 0.0079 |
| tinyllama | OOD | random frozen core | 1.3450 ± 0.0556 | 0.7235 ± 0.0216 | 0.1238 ± 0.0303 | 0.3553 ± 0.0521 |
