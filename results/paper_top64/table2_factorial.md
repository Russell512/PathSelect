| selection_score | readout | order | gamma_wp | ACC_percent | ACC_sd_pp | Masked_ORACLE_percent | Masked_ORACLE_sd_pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| text similarity | text cosine | paper | — | 88.1022 | 1.97901 | 90.0516 | 1.6167 |
| text similarity | text cosine | reverse | — | 88.1022 | 1.97901 | 90.0516 | 1.6167 |
| text similarity | ridge | paper | 0.001 | 91.9882 | 2.21677 | 94.2044 | 1.53462 |
| text similarity | ridge | reverse | 0.001 | 91.9882 | 2.21677 | 94.2044 | 1.53462 |
| residual head | text cosine | paper | — | 91.2882 | 2.35175 | 93.3698 | 1.82786 |
| residual head | text cosine | reverse | — | 91.2882 | 2.35175 | 93.3698 | 1.82786 |
| residual head | ridge | paper | 0.001 | 92.3822 | 1.98545 | 94.5067 | 1.4669 |
| residual head | ridge | reverse | 0.001 | 92.3822 | 1.98545 | 94.5067 | 1.4669 |

Percent mean ± sample SD across 10 folds; original / reverse. A rows average five seeds within fold; external rerun uses seed=fold ID. For internal rows, Masked gives true task for both evidence and final class pair (oracle WP); external rerun Masked uses true-task pair argmax. Starred rows are literature reported results with original definitions; new rows from this study only. Cosine rows retain the supervised router; they are not zero-shot pipelines.
