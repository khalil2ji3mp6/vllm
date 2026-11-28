# vLLM 无卡 CPU 基准

在仓库根目录运行。已安装离线环境的机器更新代码即可；首次离线安装时，将对应架构的依赖包放在仓库根目录，执行 `./install.sh`。

### 两台机器各跑一次

```bash
./run-suite.sh && cat host-bench-results/latest.md
```

| 测试 | 物理核数 | 负载 |
|---|---|---|
| 原版单算子 | 1、8、16、32 | 文本前处理和图片处理，每个物理核一个独立进程 |
| 整网 | 8、16、32 | DP8/TP1，每 DP 固定 16 个请求，整机并发 128 |
| 新版 worker 单算子 | 1 | 32K、128K、200K 输入。Prefill 7 项（含分词）、Decode 11 项，输出每次耗时和占比 |

海光所有档位都从 NUMA 2、6 各 16 个物理核组成的核池中选取。1 核使用节点 2 的第一个可用物理核；8、16、32 核分别在两个节点各取 4、8、16 核。脚本按进程可见的拓扑选核，完整套件要求两个节点各有 16 核可用。内存分配沿用系统策略。ARM 在调用进程允许的 NUMA 节点间选取物理核。套件及下列独立入口均遵守海光核池限制。每次运行保存在独立目录，三组完成后更新 `host-bench-results/latest.md`。

### 分别运行

```bash
./run-stages.sh                   # 原版单算子，1 / 8 / 16 / 32 核
./run.sh                          # 整网，8 / 16 / 32 核
./run-stages.sh --runner-profile --prompt-tokens 32k 128k 200k  # 新版单算子，同一物理核
```

单独选档可用 `./run.sh --core-sweep 16` 或 `./run-stages.sh --core-sweep 8 16`。原版单算子的报告为 `stages.md`，整网的两张表在 `coverage.md`，新版单算子的两张表在 `runner-profile.md`。进程通信等定位用例可通过原版入口的 `--stages` 单独指定。

### 整网并发与线程

每个请求输入 32K token 和一张 1024×1024 JPEG，输出 1K token。各档均为 8 个 DP，每个 DP 一个 API、一个 EngineCore 和一个 TP1 worker，最多 16 条活动序列。8、16、32 核分别给每个 DP 1、2、4 个物理核。每 API 的 renderer 池和媒体池容量也分别为 1、2、4；先处理图片，再渲染文本，各角色共享该 DP 核集。线程池容量表示并发上限。固定请求量使不同档位的吞吐和单请求耗时可直接对照。新增核心有助于不同请求并行和进程间调度，同一 DP 内连续 Decode 步依序执行。

### Ascend V1 worker 分步测试

新版单算子的 `--prompt-tokens` 接受一个或多个输入长度，默认 32K，K 等于 1024，也可直接写 token 数。长度包括 GLM 对话模板产生的 token。每档使用相同构造方式的重复英文文本，预先渲染后，在同一物理核上计时 RC5 分词和 Python token ID 列表导出。worker 收到这次分词生成的相同内容。

每档 Prefill 占比的分母为整条输入分词一次的均值，加上 worker 首个分块六项各自的均值。32K、128K、200K 的 worker 首步均调度 16K token，后续分块不计入该合计。单算子 Decode 前六项使用普通步，后五项使用独立投机样本。另可通过 `./run.sh --runner-profile --core-sweep 32` 在整网工作负载采集 worker 步骤。整网 Prefill 含混合步，未执行的行标为 `—`。

worker 单算子为单序列固定输入，分别回放首次 16K-token Prefill 分块和无需新增 KV 块的单 token Decode 步。每个操作单独运行，前置数据、还原请求状态和预热在计时窗口外。整网在同步 DP8/TP1 工作负载中逐步取样，每次使用调度器实际发出的整个 batch。两张表分别回答独立调用的成本和在线批处理下的成本。CPU 空间复用、批大小、块表变化和进程竞争都会影响两者。

#### 整网 11 行与源码的对应关系

下表对应 vLLM `ced6857a` 和 Ascend V1 runner `33746f6e`。参考日志提供了步骤和百分比，未给出源码版本、batch 形状、draft 方法、图模式和设备同步点。下表按源码里可识别的调用边界映射。[原始统计](https://gitcode.com/khalilzhk/tmp/blob/main/1127%201040.log)

| 步骤 | Ascend 的调用与数据 | 无卡模式计入什么 |
|---|---|---|
| `_update_states` | `execute_model` 收到 SchedulerOutput，更新新请求和常驻请求、已计算 token、块号和批行。特定配置还处理 PP、回退、KV 复制及延迟修正。 | 更新 CPU 请求缓存，以真实 vLLM `BlockTable.add_row/append_row` 管理逻辑块。新行或行重排时重建该行，稳定行只追加增量。固定生成 token 写入 CPU 输入缓冲区。|
| `prepare_inputs` | `_prepare_inputs` 从每个请求本步的 token 数展开请求索引，累计和构造序列内偏移，叠加已计算长度产生位置，再从二维 token 表用 `torch.index_select` 收集 ID。插件还会提交块表、复制元数据并计算 slot mapping。 | 实际执行 NumPy 的 repeat、cumsum、位置计算和 CPU Torch gather，构造序列长度、query 起点并取块表视图。设备 slot kernel、H2D、投机输入及其索引均跳过。 |
| `padding process` | `_determine_batch_execution_and_padding` 判断是否等宽 Decode、选择 eager 或图捕获规格，特定并行配置会校准跨 DP 长度，还可能补 query 起点。 | TP1/eager 的批形判断和 vLLM `BatchDescriptor` 构造。8 个独立副本各有自己的 batch。没有 ACL Graph bucket 或跨 rank padding 通信。 |
| `build attn metadata` | `_build_attention_metadata` 汇集 query 起点、长度上界、块表、slot 等，构造 Ascend 公共元数据，再按 KV 组和 attention backend 调用 builder。 | 长度规约和真实 vLLM `CommonAttentionMetadata` 构造。只传 CPU 缓冲区引用，slot 占位视图不被读取。每层后端 metadata、MLA/DSA 专用字段和设备 builder 未执行。 |
| `preprocess` | 基类 `_preprocess` 选择 input IDs 或 embeddings 以及位置视图。多模态、PP、M-RoPE、已有 embeddings 会改变实际调用。 | 普通文本 ID 和一维位置的 CPU tensor 切片。整网的 JPEG 读取与解码在前端，本行没有视觉 encoder。 |
| `set_ascend_forward_context` | Ascend 上下文管理器调用 vLLM 配置和 forward context，并设置 MoE 通信策略、并行规模、图及模型状态。forward 在上下文内部运行。 | 计核心库 `set_current_vllm_config` 与 `set_forward_context` 的进入和退出。排除 forward 和 Ascend 专用设置。此行是公共部分测量。 |
| `model_forward` | `_model_forward` 使用 inputs、positions、context 调用模型或回放图。未同步的 Python 墙钟主要反映提交和可能的阻塞，完整设备执行需要时间线或设备事件。 | —，没有加载权重或调用 forward。 |
| `compute logits` | 选取需要采样的隐藏状态，在相应 PP rank 调用 `model.compute_logits`，可能含 LM head 和并行通信。 | —，无 hidden states 和 LM head。 |
| `rej sample` | `sample_tokens → _sample` 依据 spec metadata 选择 sampler。有草稿时 rejection sampler 接收目标 logits、草稿信息及采样参数，产生接受结果。 | —，同步调度器消费固定 token，真实普通采样和拒绝采样都未调用。 |
| `padding process for draft` | 模型型 proposer 可单独校准长度、选择图规格、调整 query 或块表形状。具体边界随 drafter 和图模式变化。 | —，没有配置草稿引擎。|
| `propose` | `propose_draft_token_ids` 分派到具体 proposer。Eagle/MTP 等模型路径准备输入、运行草稿 forward 并产生候选 token，n-gram 等方法走不同代码。真实计时若单列 draft padding，需要从 propose 父区间中扣除。 | —，无草稿 forward 或候选生成。|

源码入口可从 [Ascend execute_model](https://github.com/vllm-project/vllm-ascend/blob/33746f6e0810c0c2886a6fafde1b9ba9ced47ed7/vllm_ascend/worker/v1/model_runner.py#L2332)、[Ascend slot mapping](https://github.com/vllm-project/vllm-ascend/blob/33746f6e0810c0c2886a6fafde1b9ba9ced47ed7/vllm_ascend/worker/block_table.py#L160)、[Ascend forward context](https://github.com/vllm-project/vllm-ascend/blob/33746f6e0810c0c2886a6fafde1b9ba9ced47ed7/vllm_ascend/ascend_forward_context.py#L115)、[Ascend 草稿模型路径](https://github.com/vllm-project/vllm-ascend/blob/33746f6e0810c0c2886a6fafde1b9ba9ced47ed7/vllm_ascend/spec_decode/llm_base_proposer.py#L1010) 和 [vLLM 输入预处理](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu_model_runner.py#L3513) 查看。模拟中对应的代码集中在 [runner_host.py](vllm-host-bench/runner_host.py) 和 [runner_report.py](vllm-host-bench/runner_report.py)。


#### 单算子补测的五个 Host 子段

`./run-stages.sh --runner-profile --core-sweep 1 --keep-diagnostics` 同时测普通输入和投机验证。投机样本固定为 TP1、单序列、已有指定输入长度的上下文、3 个候选。样本接受第一个候选并输出一个替换 token，验证步宽 4，草稿输入保留 2 个位置。候选和验证结果在计时前生成。五项独立回放同一个投机样本，在 Decode 表中与前六项共用分母。

| 单算子 | 此版本具体计时 |
|---|---|
| model_forward | 按 Ascend `_model_forward` 绑定调用参数，在目标模型 eager 路径按 GLM 配置遍历 78 层，执行每层的 token 维度和辅助输出检查。计算层、模块内部调度、算子下发和图回放均跳过。 |
| compute logits | 对预置的 4 行投影结果执行 TP1 普通 LM head 路径的词表切片，生成共享存储的 tensor 视图。隐藏状态 gather 和矩阵乘跳过。 |
| rej sample | 校验候选长度，用 vLLM `SamplingMetadata` 创建 bonus 采样参数并构造 `SamplerOutput`。设备采样及接受判定跳过。同步回包后还有一次真实 `RejectionSampler.parse_output`，开启 `--keep-diagnostics` 时在 `runner-profile.json` 的 `bookkeeping` 中独立记录它过滤占位符、导出 Python IDs 的时间。 |
| padding process for draft | 使用真实 vLLM 图规格 dispatcher，先按本地草稿长度查一次，再按协调后的长度查一次。独立 DP 下长度相同，基准设置 PIECEWISE 规格 1、4、8，2 个有效位置选中 4。计入 CPU 查表及描述符生成，未捕获或启动图。 |
| propose | 选择插件支持的 CPU 紧凑草稿输入路径。执行上游 `prepare_next_token_ids_cpu`，根据拒绝数修正序列长度、重算 query 起点，生成保留 token 的索引。next ID 落在 CPU 上，索引和长度均有实际数值。草稿模型 forward、H2D 以及默认 padded GPU 路径不在本行。 |

投机样本专用于比较两台 CPU 的指定代码段。PIECEWISE 配合紧凑输入，避免 Ascend 对 FULL 图与禁用 padded drafter 的组合限制。正式运行的默认 padded drafter 会把部分工作放在设备上。这五项在 [单算子实现](vllm-host-bench/runner_spec_host.py) 中，与整网固定 token 的 worker 独立。[GLM 模型主循环](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/models/deepseek_v2.py#L1504) · [Ascend logits](https://github.com/vllm-project/vllm-ascend/blob/33746f6e0810c0c2886a6fafde1b9ba9ced47ed7/vllm_ascend/ops/vocab_parallel_embedding.py#L542) · [拒绝采样](https://github.com/vllm-project/vllm-ascend/blob/33746f6e0810c0c2886a6fafde1b9ba9ced47ed7/vllm_ascend/sample/rejection_sampler.py#L146) · [草稿输入压紧](https://github.com/vllm-project/vllm-ascend/blob/33746f6e0810c0c2886a6fafde1b9ba9ced47ed7/vllm_ascend/spec_decode/llm_base_proposer.py#L2251)
