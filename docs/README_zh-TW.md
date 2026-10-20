# PathSelect 論文程式碼

這份 repository 整理自已驗收的 `paper_top64_20261016` 實驗。
主方法是 **FINAL-B / B_TOP64_RIDGE**：全片 mean 的 task router →
一次選 top-64 patches → raw features 等權平均後 L2 → 累加式 subtype ridge。
FINAL-A 的 I6 residual head 保留作消融，沒有更換主方法或挑選 seeds。

- [安裝、範例及主要結果](../README.md)
- [完整重現流程](REPRODUCING.md)：Slurm 帳號、partition、資料根目錄改為自行設定。
- [資料格式](DATA.md)：需自備原始 CONCH features 與十折 splits；不重切資料。
- [方法與統計定義](METHOD.md)：兩種 Masked 分開，五 seeds 先 fold 內平均。
- [Table 2(b) 核對](TABLE2_AUDIT.md)：是全片 mean，random64 是另一個獨立對照。
- [論文表格](../results/paper_top64/README.md)：CSV、LaTeX、Markdown，以及 budgets、gamma、配對 CI。
- [來源與授權狀態](../NOTICE.md)：保留上游來源，沒有擅自加入 MIT 等授權。

`python -m pathselect.demo` 可以在沒有資料集的情況下跑兩種 order、四個
stage 的合成資料範例。這只是安裝與流程示範，不是論文準確率。

本次公開的是可重現方法與統計的原始碼、固定文字向量及彙總結果；
不包含病人逐張資料、raw WSIs、CONCH 權重、外部模型 checkpoints、
私人伺服器設定或歷史 paper 草稿。原本伺服器的完整封存成果保持不變。
