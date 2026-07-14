# SparseFlow Build in Public：九阶段内容主线

当前写作身份：**[WRITER]**

这不是一组互相独立的技术文章，而是一条连续的工程探索故事：从一台资源有限的电脑出发，逐步研究如何让大规模稀疏专家模型以分层存储方式运行。

## 内容原则

1. 每篇文章只讲一个问题，不把所有实现细节堆在一篇里。
2. 所有数字必须能回到代码、命令或 `docs/results/` 中的原始结果。
3. 明确区分“已经验证”“当前假设”和“下一步计划”。
4. 不把单层 probe、MoE-only runtime 或 page-cache timing 写成完整模型性能。
5. 小红书强调问题、过程和视觉结果；X 强调架构、代码和可复现实验。

## 九个阶段

### 1. 起点：一台 16GB 电脑能不能碰 35B MoE？

- Hook：模型 67GB，电脑只有约 16GB RAM 和 4GB VRAM。
- 核心问题：为什么还值得尝试，而不是直接放弃？
- 证据：本地硬件、模型大小、Qwen3.6 的 MoE 参数结构。
- 视觉：电脑资源 vs 模型资源对比卡片。
- 结尾：引出“不是把模型全部装进内存，而是只取当前需要的 expert”。

### 2. 参考：从 Colibri 744B 中借鉴什么？

- Hook：真正值得借鉴的不是 744B 这个数字。
- 核心问题：Colibri 如何把冷 expert 放在 SSD？
- 证据：dense 常驻、expert streaming、cache、tier planner。
- 视觉：RAM / SSD / VRAM 三层流动图。
- 结尾：提出 SparseFlow 的模型无关目标。

### 3. 模型体检：67GB 里面到底是什么？

- Hook：总参数量很吓人，但每个 token 并不会使用全部参数。
- 核心问题：哪些权重必须常驻，哪些可以冷存储？
- 证据：66.97 GiB、40 layers、256 experts、top-8、dense 约 6.97 GiB、每层 expert 约 1.5 GiB。
- 视觉：权重分类堆叠图。
- 结尾：引出“下一步必须精确找到一个 expert 的字节范围”。

### 4. ExpertLocator：一个 expert 只有 6 MiB

- Hook：我们没有读取整个 fused tensor，只读取了一个 expert 的两个 slice。
- 核心问题：fused expert tensor 如何定位和读取？
- 证据：`gate_up_proj`、`down_proj`、file offset、6 MiB 精确读取。
- 视觉：fused tensor → expert slice → `pread`。
- 结尾：引出 cache：如果下一个 token 又需要它，能不能不重复读盘？

### 5. 路由与缓存：模型每一步到底会访问哪些 expert？

- Hook：模拟 uniform route 和真实 router route，结果完全不同。
- 核心问题：cache 命中率应该按 token、batch 还是 unique expert 统计？
- 证据：真实 route trace、prefill/decode、batch-union、不同 slots/layer 的结果。
- 视觉：raw route requests → unique expert requests → cache hits。
- 结尾：引出“缓存命中不代表输出正确，先做 correctness”。

### 6. 正确性闭环：resident 和 streaming 能不能算出同一个结果？

- Hook：两条路径完全不同，但 final output 必须逐元素一致。
- 核心问题：router、routing weights、routed output、shared output 是否都一致？
- 证据：真实 layer 0、Transformers 官方 block 对照、max error 为 0。
- 视觉：完整单层 MoE 数据流图。
- 结尾：引出正式 `ExpertCache` 接入。

### 7. ExpertCache：从“能读”到“能复用”

- Hook：缓存不是字典，而是带容量、驱逐和统计的运行时组件。
- 核心问题：per-layer LRU 和 global byte budget 是否真的有效？
- 证据：真实重复 forward 命中、eviction、6 MiB budget、resident/cached exact equality。
- 视觉：cache hit/miss/eviction 时间线。
- 结尾：引出 SSD 读取本身的 syscall 成本。

### 8. I/O、prefetch、多层 runtime：把实验组件串起来

- Hook：减少 read calls 可能带来额外读取，优化不是免费的。
- 核心问题：coalescing 和异步 prefetch 的收益与代价是什么？
- 证据：64→62 calls 零浪费；64→58 calls 但多读约 20 MiB；两层 MoE-only runtime exact equal。
- 视觉：无 prefetch / gap=0 / gap=6 MiB 三组对照图。
- 结尾：引出完整 Transformer 仍然更难。

### 9. 下一站：完整 Qwen3.6 runtime 与开源评测

- Hook：MoE block 跑通，不等于完整模型已经跑通。
- 核心问题：attention、Gated DeltaNet、KV cache、generation 如何接入？
- 证据：当前已完成项、明确未完成项、Benchmark 评测矩阵。
- 视觉：从 MoE-only runtime 到完整 Transformer 的路线图。
- 结尾：邀请读者关注、复现实验或贡献 adapter/runtime。

## 平台改写规则

### 小红书

结构固定为：

```text
一个反直觉标题
→ 机器/模型限制
→ 我们实际做了什么
→ 一张数字结果图
→ 一个仍未解决的问题
```

每篇控制在一个核心结论，代码只展示最短的关键片段。

### X

采用 thread：

```text
Hook
→ architecture
→ experiment
→ raw numbers
→ caveat
→ next step
```

可以附 CLI、JSON result、架构图和 GitHub 文件链接。

## 证据来源

- 模型结构：`docs/implementation_plan.md`、`inspect` 输出。
- Colibri 参考：`docs/colibri_borrowable_modules.md`。
- ExpertCache：`docs/results/qwen36_moe_cache_correctness_20260714.md`。
- 阶段 5：`docs/results/qwen36_stage5_moe_runtime_20260714.md`。
- Benchmark：`docs/benchmark_setting.md` 和 `docs/results/`。

之后每发布一篇文章，都应在本文件对应阶段下增加发布日期、平台版本和引用的 result 文件。[WRITER]
