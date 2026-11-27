# vLLM 无卡 CPU 基准

在仓库根目录运行。已安装离线环境的机器更新代码即可；首次离线安装时，将对应架构的依赖包放在仓库根目录，执行 `./install.sh`。

### 两台机器各跑一次

```bash
./run-suite.sh && cat host-bench-results/latest.md
```

脚本自动识别海光 x86 或 ARM，按顺序运行原版单算子、原版整网和新版单算子。自动选核覆盖 1、32、64 和当前可用的全部物理核。整网的 1 核单独用 DP1，其他档位用 DP8。海光每组额外测试 NUMA 2、6 各 16 核。请从可使用整机 CPU 的 shell 运行。每次运行的原始表格、日志和汇总保存在独立目录，全部成功后更新 `host-bench-results/latest.md`。

### 1. 整网

```bash
./run.sh                         # 全部可用物理核
./run.sh --core-sweep 32         # 仅测 32 个物理核
```

默认全机先处理图片，再做文本模板和分词。DP8 / TP1，每 DP 并发 48、最多 48 条活动序列（整机并发 384），每请求输入 32K token 和一张 1024×1024 JPEG，输出 1K token。完整 Prefill 和 Decode 阶段表见 `host-auto/coverage.md`。

默认 DP8 时，整网指定的核数须为 8 的倍数。32 核对应每 DP 4 核。

### 2. 单流程

默认只测请求与文本前处理、图片接入与处理。

```bash
./run-stages.sh                       # 默认测单核和全部可用物理核
./run-stages.sh --core-sweep 1 32     # 仅测 1 核和 32 核两档
```

结果见 `stages/stages.md`。仅需 32 核一档时使用 `--core-sweep 32`。定位其他阶段时用 `--stages` 点名，例如测两段 IPC 和引擎调度：

```bash
./run-stages.sh --stages frontend_ipc worker_ipc drive_scheduler --core-sweep 32
```

### 3. 两台机器固定两个 NUMA 节点，各取 16 个物理核

先在各自机器的 shell 设置核号。海光 x86 使用节点 2、6。以下编号以该机 `lscpu -e=CPU,NODE,SOCKET,CORE` 的映射为准：

```bash
CPUS=32-47,96-111
ARCH=x86
```

ARM 使用节点 0、1。按本机拓扑每个物理核取一个逻辑 CPU，避开重复计算 SMT 线程：

```bash
CPUS=$(PYTHONPATH=vllm-host-bench vllm-host-bench/.venv/bin/python - <<'PYTHON'
from cpu_topology import cpu_topology

topo = cpu_topology()
physical = set(topo["physical_core_cpus"])
selected = []
for node in (0, 1):
    cores = [cpu for cpu in topo["numa_nodes"].get(node, []) if cpu in physical]
    if len(cores) < 16:
        raise SystemExit(f"NUMA 节点 {node} 可用物理核不足 16 个")
    selected.extend(cores[:16])
print(",".join(map(str, selected)))
PYTHON
)
ARCH=arm
```

两台机器各自在同一个 shell 执行以下两条命令。`taskset` 将测试进程绑定到选定的 CPU，内存分配沿用系统策略。

```bash
taskset -c "$CPUS" ./run-stages.sh --core-sweep 1 32 --out "stages-${ARCH}-local32"
taskset -c "$CPUS" ./run.sh --core-sweep 32 --keep-diagnostics --out "host-${ARCH}-local32"
```

两个脚本重复运行都会创建带数字后缀的目录。

### 4. 按 Ascend V1 worker 步骤统计

```bash
./run-stages.sh --runner-profile --core-sweep 1 32 --out runner-stages
./run.sh --runner-profile --core-sweep 32 --out runner-host
```

两个脚本加 `--runner-profile` 都只输出 Prefill 和 Decode 两张表，分别含 6 项和 11 项。核数并排列出，每列按该阶段所有实测项的未舍入均值之和计算占比。单算子 Decode 的前六项来自普通单 token 步，后五项来自投机验证样本，合计是跨样本均值之和。整网按实际执行的 worker 步统计，Prefill 包含混合步，Decode 仅含纯 Decode 步，未执行的行显示 `—`。结果见输出目录下的 `runner-profile.md`，按样本拆分的调用次数和墙钟纳秒总和见 `runner-profile.json`。省略该参数时分别运行原单算子或整网基准。

单算子为单序列固定输入，分别回放首次 16K-token Prefill 分块和无需新增 KV 块的单 token Decode 步。每个操作单独运行，前置数据、还原请求状态和预热在计时窗口外。整网在同步 DP8/TP1 工作负载中逐步取样，每次使用调度器实际发出的整个 batch。两张表分别回答独立调用的成本和在线批处理下的成本。CPU 空间复用、批大小、块表变化和进程竞争都会影响两者。

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

`./run-stages.sh --runner-profile --core-sweep 1 32 --keep-diagnostics` 同时测普通输入和投机验证。投机样本固定为 TP1、单序列、已有 32K 上下文、3 个候选。样本接受第一个候选并输出一个替换 token，验证步宽 4，草稿输入保留 2 个位置。候选和验证结果在计时前生成。五项独立回放同一个投机样本，在 Decode 表中与前六项共用分母。

| 单算子 | 此版本具体计时 |
|---|---|
| model_forward | 按 Ascend `_model_forward` 绑定调用参数，在目标模型 eager 路径按 GLM 配置遍历 78 层，执行每层的 token 维度和辅助输出检查。计算层、模块内部调度、算子下发和图回放均跳过。 |
| compute logits | 对预置的 4 行投影结果执行 TP1 普通 LM head 路径的词表切片，生成共享存储的 tensor 视图。隐藏状态 gather 和矩阵乘跳过。 |
| rej sample | 校验候选长度，用 vLLM `SamplingMetadata` 创建 bonus 采样参数并构造 `SamplerOutput`。设备采样及接受判定跳过。同步回包后还有一次真实 `RejectionSampler.parse_output`，开启 `--keep-diagnostics` 时在 `runner-profile.json` 的 `bookkeeping` 中独立记录它过滤占位符、导出 Python IDs 的时间。 |
| padding process for draft | 使用真实 vLLM 图规格 dispatcher，先按本地草稿长度查一次，再按协调后的长度查一次。独立 DP 下长度相同，基准设置 PIECEWISE 规格 1、4、8，2 个有效位置选中 4。计入 CPU 查表及描述符生成，未捕获或启动图。 |
| propose | 选择插件支持的 CPU 紧凑草稿输入路径。执行上游 `prepare_next_token_ids_cpu`，根据拒绝数修正序列长度、重算 query 起点，生成保留 token 的索引。next ID 落在 CPU 上，索引和长度均有实际数值。草稿模型 forward、H2D 以及默认 padded GPU 路径不在本行。 |

投机样本专用于比较两台 CPU 的指定代码段。PIECEWISE 配合紧凑输入，避免 Ascend 对 FULL 图与禁用 padded drafter 的组合限制。正式运行的默认 padded drafter 会把部分工作放在设备上。这五项在 [单算子实现](vllm-host-bench/runner_spec_host.py) 中，与整网固定 token 的 worker 独立。[GLM 模型主循环](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/model_executor/models/deepseek_v2.py#L1504) · [Ascend logits](https://github.com/vllm-project/vllm-ascend/blob/33746f6e0810c0c2886a6fafde1b9ba9ced47ed7/vllm_ascend/ops/vocab_parallel_embedding.py#L542) · [拒绝采样](https://github.com/vllm-project/vllm-ascend/blob/33746f6e0810c0c2886a6fafde1b9ba9ced47ed7/vllm_ascend/sample/rejection_sampler.py#L146) · [草稿输入压紧](https://github.com/vllm-project/vllm-ascend/blob/33746f6e0810c0c2886a6fafde1b9ba9ced47ed7/vllm_ascend/spec_decode/llm_base_proposer.py#L2251)
