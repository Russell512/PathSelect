| row | experiment | order | gamma | ACC_percent | ACC_sd_pp | Masked_ORACLE_percent | Masked_ORACLE_sd_pp | mean_difference_pp | wins | losses | ties | p_raw | slide_ci_low_pp | slide_ci_high_pp | patient_ci_low_pp | patient_ci_high_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a | LIN8_MEAN | paper | 0.01 | 85.6625 | 2.94438 | 92.4323 | 2.00277 | — | — | — | — | — | — | — | — | — |
| a | LIN8_MEAN | reverse | 0.01 | 85.6625 | 2.94438 | 92.4323 | 2.00277 | — | — | — | — | — | — | — | — | — |
| b | ROUTED_MEAN | paper | 0.01 | 90.4479 | 2.39969 | 92.4323 | 2.00277 | 4.78538 | 10 | 0 | 0 | 0.00195312 | 3.36714 | 6.21201 | 3.33526 | 6.3885 |
| b | ROUTED_MEAN | reverse | 0.01 | 90.4479 | 2.39969 | 92.4323 | 2.00277 | 4.78538 | 10 | 0 | 0 | 0.00195312 | 3.36714 | 6.21201 | 3.33526 | 6.3885 |
| c | B_TOP64_RIDGE | paper | 0.001 | 91.9882 | 2.21677 | 94.2044 | 1.53462 | 1.5403 | 9 | 1 | 0 | 0.0214844 | 0.39695 | 2.72666 | 0.439703 | 2.7041 |
| c | B_TOP64_RIDGE | reverse | 0.001 | 91.9882 | 2.21677 | 94.2044 | 1.53462 | 1.5403 | 9 | 1 | 0 | 0.0214844 | 0.39695 | 2.72666 | 0.439703 | 2.7041 |

Percent mean ± sample SD across 10 folds; original / reverse. A rows average five seeds within fold; external rerun uses seed=fold ID. For internal rows, Masked gives true task for both evidence and final class pair (oracle WP); external rerun Masked uses true-task pair argmax. Starred rows are literature reported results with original definitions; new rows from this study only. Delta is current minus preceding row; pp.
