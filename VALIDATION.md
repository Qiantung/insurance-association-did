# Validation

验证日期：2026-11-10；Python 3.12.14。

- numpy==2.3.5
- pandas==3.0.1
- scikit-learn==1.9.1
- scipy==1.18.1
- statsmodels==0.14.6
- patsy==1.0.3


实际运行 `python research.py --output results/synthetic --placebos 100` 完成；440 行完全合成数据、40 个虚构城市、11 年，100 次按城市状态置换。预设效应为 2，示例估计约 2.01362、城市聚类标准误约 0.21423。该数字只用于工程验证，不是论文结果。

`python -m unittest discover -s tests`：3 项通过（重复面板、时变分组拒绝、已知合成效应）。原研究数据和历史论文结果未重新完整复现。
