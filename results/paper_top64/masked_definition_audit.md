# Masked 欄位來源稽核

固定上游 SHA：f5813cd81c845ff11549d69615c32ede6d32c971。下列上游檔案在本次 frozen source 中未變更。

- `scripts/ext1_c.py:92–111` 的 `cl_eval` 同時計算兩種 Masked。line 102：`pv = th if which == "t1" else task`；line 103：`first(t, pv, task)`。`pv` 決定 evidence 的 task，最後 `task` 決定 true-task class pair。
- 同檔 line 109–110 將 t1 結果存成 `mk1_t`，oracle 結果存成 `wp_t`。
- `scripts/lad1_cells.py:188–195` 明確把 `mk1_t` 寫入 `MaskedACC_table1`，把 `wp_t` 寫入 `WP` 與 `MaskedACC_oracle`。
- `scripts/ext3_gamma.py:203–211` 的逐列 gamma 表延續相同欄位映射；`scripts/ext3_report.py:166` 的舊 Table 1 摘要使用 `MaskedACC_table1`。

因此舊 Table 1 的 Masked 並非普遍等同 oracle WP。對 task-dependent evidence，兩者可能不同；對 LIN8/ROUTED_MEAN 的全片 mean evidence，兩者應完全一致。

本批 paper-ready Table 1/2 的 Masked 定義為 **Masked_ORACLE**：true task 同時指定 evidence texts/head 與最後兩類。**Masked_ROUTED_EVIDENCE** 保留 predicted-task evidence，再限制最後兩類，作 legacy/secondary 對照。兩者逐 slide predictions、逐折 metrics 與配對統計均保存；不得直接把新 oracle 與舊 legacy 數字作選片效果差。

本批 `paper64_report.py` 從原始 scores/predictions 重新核對兩種 argmax，逐 stage 核對 LIN8/ROUTED_MEAN 的 oracle predictions 與 subtype W 完全相同。最終數字另見 per_fold.csv 和 metrics_mapping.txt。
