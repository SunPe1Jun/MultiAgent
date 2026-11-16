# Exact-set 优化与修正模块记录

Exact-set 低的主要原因不是 mention 完全没有解析，而是：

- 同类实例编号选错；
- 多 mention 中漏掉某个对象；
- `here/there` 额外引入对象；
- 全局 coordinator 对逐 mention 证据重新混合；
- 时间窗口有效但视觉身份不足。

建议的修正模块保持轻量：对每个 mention 复核当前选择和前两个同类候选，输入该 mention 的 RGB 主帧、少量上下文、候选 2D 投影和必要 3D 属性；要求输出保留、替换或不确定及理由，最后按 mention 合并集合。它不读取 GT，不强制重复词组选择不同实例，也不通过拒答人为提高 precision。

`temporal_focus.py` 已实现局部窗口实验，但四条冻结样本的 Exact-set 没有提升，因此目前只通过 `--mention-focus` 启用。下一步应先核查投影与候选裁剪是否真的对应同一物体，再评估轻量修正模块。
