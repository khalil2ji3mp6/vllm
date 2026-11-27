# 无卡 Host 基准的进程、线程和物理核

同一台机器上，单流程测「给某个操作持续供给工作时，每秒能完成多少」，整网测「固定 DP、并发、输入和输出长度时，这批请求何时全部完成」。单流程加核时可以增加完整的执行流水。整网加核时，DP 和 TP 决定的 EngineCore 和 worker 数量保持固定。

## 从宿主机分核到一个 DP

1. `run.sh` 进入 [host_capacity.py](host_capacity.py) 的 `main`。进程从当前 Linux affinity 掩码取可用 CPU，按 sysfs 的 `thread_siblings_list` 每物理核只选一个逻辑 CPU，按 NUMA 节点建池。外部 `taskset` 会缩小这个集合。选择逻辑没有按节点的可用内存容量过滤。[cpu_topology.py](cpu_topology.py)
2. 默认整机全核路径用 `distribute_all_cores` 分配不相交的 DP 核组；显式 `--core-sweep N` 为每 DP 分配等宽的 `N/DP` 个核，按节点轮流取可容纳完整组的 CPU，最后才合并各节点剩余 CPU。整网每档总核数须为 DP 的正整数倍。DP8 的最小值是 8 核，即每 DP 一核；整机一核使用 DP1。单流程的 1 核没有这个 DP8 约束。[capacity_plan、make_plan 和 run_capacity](host_capacity.py)
3. 父进程用 `taskset -c <该 DP 的 CPU 列表> run-native.sh ...` 分别启动副本，副本的主进程运行 EngineCore，通过 spawn 启动本地前端和 TP worker。常用的 `inherit` 模式让子进程和其线程继承整组核，Linux 在组内调度。固定每副本核数恰为 `TP+2` 且允许单前端时，规划器可选 `compact`，分别给引擎、前端和各 worker 一个核。整网默认只设 CPU affinity，内存遵循系统的内存策略。引擎、每个前端和 worker 的主线程实际掩码保存在各副本 summary 的 hardware 字段。[启动和角色绑核](bench.py)
4. 整机控制器为每一轮同步放行各 DP，排除预热。每 DP 自己接收固定数量的请求和运行自己的 Scheduler。这里模拟 8 个独立本地副本；跨 DP 的生产负载均衡、卡侧计算和集合通信没有计时。

### 自动前端参数

当前默认分阶段运行，两个池的线程槽数均为 ⌈min(C,Q)/F⌉。下表对应默认 Q=48。重叠模式（`--no-separate-image-text`）另用下面的池预算。

每个 DP 的物理核数为 C，在途请求数为 Q。共享核组下，前端预算 B 取 max(1, C − TP − 1)。前端进程数 F 取 max(1, min(⌊B/4⌋, ⌈Q/16⌉))。重叠模式下每个前端的 renderer 池容量取 min(⌈Q/F⌉, max(1, ⌊(B−F)/F⌋))，有图片时媒体池默认用同样的容量。预算中的减法只用于算池大小，inherit 模式并没有给引擎和 worker 划出专属 CPU。前端按请求序号轮转分片，每条请求从输入到流式输出由同一个前端负责。[frontend_plan](bench.py) · [front_worker](frontend.py)

| 总物理核 / DP | 每 DP 核组 | 每 DP 前端进程 | 每前端 renderer / 媒体线程槽 | 整机 EngineCore / worker |
|---|---:|---:|---:|---:|
| 1 / DP1 | 1 | 1 | 1 / 1 | 1 / 1 |
| 32 / DP8 | 4 | 1 | 4 / 4 | 8 / 8 |
| 64 / DP8 | 8 | 1 | 8 / 8 | 8 / 8 |
| 128 / DP8 | 16 | 3 | 6 / 6 | 8 / 8 |
| 192 / DP8 | 24 | 3 | 8 / 8 | 8 / 8 |

表中 TP1、Q=48、有图。分词线程按请求并行，同一 API 的校验与建状态共用该进程的事件循环。默认 Torch 算子线程和数值库的内部线程预算为 1。Python 事件循环、线程池与 ZeroMQ 等库的辅助线程另计。线程槽为并发上限，实际同时执行的线程数和整轮平均占核都需要从运行数据读取。

## 一条请求如何通过六个阶段

| 阶段 | 代码入口与数据流 | 执行者和并行边界 |
|---|---|---|
| 请求与文本前处理 | [preprocess_requests、MeasuredRenderer](frontend.py) → Schema 和采样参数 → OnlineRenderer → Chat Template、分词 → InputProcessor、OutputProcessor 建状态。[RC5 适配器](tokenizer_rc5.py) 返回 Python token ID 列表 | 每前端一个 uvloop 事件循环。校验、建状态在主线程，模板和分词交给 renderer 线程池。每个前端有独立解释器，同进程 Python 代码和 ID 列表构造受到 GIL 约束 |
| 图片接入与处理 | 校验后 await [ImageCodec.decode_request](image_codec.py) → MediaConnector → Base64 解码、Pillow 解压与 RGB 转换，随后继续同一请求的文字渲染 | 每前端使用媒体池。不同请求由 asyncio.gather 并发推进，重叠模式下图片与别的请求的文字可以重叠；同一请求先完成图片再进行文字渲染。图片在本地内存中，独立编码参考轮位于业务计时之外 |
| 进程通信与入队 | 前端将 32K 请求序列化，通过 ZMQ PAIR 交给 EngineCore；引擎反序列化、计算块哈希并入队。调度结果通过 SHM MessageQueue 下发，worker 返回固定结果，引擎再将输出发给所属前端。[timed_scenario](bench.py) · [run_worker](worker.py) | 每 DP 一个引擎消费者。引擎对本步 worker 回复进行同步等待，主表通信分组统计收发和解包等已打点区间。完整等回复时间在细表，含对端准备数据的时间 |
| 引擎调度与请求状态 | [build_scheduler、add、classify](engine.py) → Scheduler.schedule → 分配 token 预算和逻辑 KV 块 → 收到 worker 结果 → update_from_output → 释放或继续调度 | 每 DP 一个同步调度循环，整网默认最多 48 条活动序列、每步 16,384 token 预算。自动逻辑 KV 块预算按可同时运行的序列及输入输出长度计算，保留一个空块，至少 8,192 块。48 条 32K 输入 / 1K 输出在块长 128 时分配 12,673 块。基准显式设 `async_scheduling=False`，当前步结果更新后再调下一步 |
| Worker 输入准备 | [CpuBatch.update、CpuBatch.prepare](worker.py) 维护请求与块表，组装 IDs、位置、序列长度和索引；[dummy_result](engine.py) 生成固定返回 token | 每 DP / TP rank 一个 worker。TP1 时同一时刻每个 DP 只有一份 worker 在处理这一步。输入数组在 CPU 上构造，不加载模型权重或启动设备算子 |
| 输出处理与响应 | [stream_outputs、serialize_chunks](frontend.py) 解包返回，增量反分词、检查停止串、序列化本地流式响应，并记录首次及最终 token 的时间 | 每前端主事件循环处理本前端的输出。不同前端可并行；同一请求的下个 token 依赖上一步，1,024 个输出无法靠扩大 renderer 池同时完成 |

## 真机调度与基准调度

早期的 20 来自 [TP8 参考启动脚本](reference-glm52-950dt-tp8.sh)。整网入口的每 DP 并发控制每轮提交数量，`--max-num-seqs` 控制 Scheduler 同时运行的序列上限。每步 token 预算和可用 KV 块进一步限制准入。

基准显式构造同步 Scheduler，发送一个调度步后等待 worker 的固定结果。对应上游版本在执行器及模型配置兼容时默认开启异步调度，PP1 下可保留两个在途批次，由 EngineCore 选择带队列的步进路径。真实卡侧执行能与 Host 工作重叠，本基准用于分析同步路径的 CPU 成本。[上游配置](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/config/vllm.py#L1438) · [上游步进循环](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/engine/core.py#L630)

## 单流程为什么有不同的扩核曲线

[stage_bench.py](stage_bench.py) 从每个 NUMA 节点轮流选核。默认计算类用例每个物理核运行一个固定 affinity 的独立进程，预热后在采样窗口内反复调用同一操作。例如 192 核的调度单测可以同时运行 192 份独立 Scheduler；DP8 整网只有 8 份。可选的共享解释器细测最多启动 8 个进程，每核一个绑核线程。两项 IPC 单测最多启动 8 对收发进程，每对共享一个核组，它们的流水数也不会从 64 增到 192。[八个操作的实现](stage_cases.py) · [IPC 两端](stage_transport.py)

## 从输出判断时间花在哪里

整网前端按请求记录墙钟。请求与文本前处理包含本地提交后等待事件循环、校验，以及媒体返回后至文本建状态完成；图片行从调用媒体处理到结果返回，包含排队。两行相加得到本地提交至前端准备完成的平均延迟。引擎、通信和 worker 行仍为参与该操作的请求记录已打点耗时。

- `coverage.md` 记录单请求平均已打点耗时。Prefill 包含一个请求全部输入分块及首 token；Decode 按一个请求首 token 后的单个输出步平均。共享批操作对每个参与请求计入完整耗时；混合批按请求是否已生成首 token 分开归属。阶段合计逐列相加。端到端还包含排队与空闲等待。[分组目标](flow_groups.py) · [区间聚合与吞吐](report.py)
- `--keep-diagnostics` 保留 `details/coverage.md`。比较前端扩核效果看「Prefill 前处理合计」，请求生命周期看逐轮 TTFT 和完成延迟。整轮吞吐是正式轮请求/s 的中位数。请求总数和 token 数由引擎、前端和整机控制器核对。
- `.runs/cores-N/cpu-plan.json` 记录分配策略；`host-rounds.csv` 给出各角色的进程 CPU 时间除以墙钟所得的有效核及线程峰值。有效核包括同步、轮询等 CPU 时间。`core-usage.md` 另给出部分组活跃期间实际占核，统计分母不同。[整机 CPU 汇总](host_capacity.py) · [阶段占核](stage_profile.py)

## ARM 参考机历史对照（max_num_seqs=20）

同一台鲲鹏 920，共 320 个物理核，测试池从 8 个 NUMA 节点各取 24 个物理核。TP1，每 DP 32 个请求，32,768 个输入 token，每请求一张 1024×1024 JPEG。整网预热 1 轮、正式 5 轮。参考机有其他负载。表中的吞吐和有效核取正式轮中位数，两列均按整轮统计。

| 配置 | 总物理核 | 实际前端进程数 / 每 DP | 每前端 renderer / 媒体线程槽 | 整网吞吐（请求/s） | 整轮有效核 |
|---|---:|---:|---:|---:|---:|
| DP1，输出 1,024 | 1 | 1 | 1 / 1 | 7.98 | 1.00 |
| DP1，输出 1,024 | 64 | 2 | 16 / 16 | 17.43 | 2.56 |
| DP1，输出 1,024 | 192 | 2 | 16 / 16 | 17.55 | 2.56 |
| DP8，输出 1,024 | 64 | 1 | 5 / 5 | 136.56 | 19.17 |
| DP8，输出 1,024 | 192 | 2 | 10 / 10 | 135.10 | 19.81 |
| DP8，输出 1 | 64 | 1 | 5 / 5 | 1,252.10 | 29.62 |
| DP8，输出 1 | 192 | 2 | 10 / 10 | 1,380.82 | 32.26 |

- **前端可以扩核。** DP8 长输出时，64→192 核，按 DP、按轮统计的前端整批准备墙钟中位数从 129.6 ms 降到 57.2 ms。单流程文本吞吐从 28,244 增至 85,631 请求/s，图片从 7,120 增至 21,089 图/s。
- **长输出的主要时间在逐步解码。** 两档每 DP 每轮的纯 Decode 步数中位数均为 2,007，纯 Decode 步的累计墙钟中位数为 1.616 / 1.627 s。整机引擎有效核为 7.89 / 7.95，worker 为 7.92 / 7.90。CPU 计费包含等待时的轮询；引擎和 worker 的这些时间可以重叠。
- **Prefill 整体包含调度等待。** 各 DP 每轮按首 token 时间排序，再跨 40 个 DP 轮次取中位数。64 核第 20 / 21 个首 token 到达于 170 / 1,126 ms；192 核为 163 / 1,123 ms。每 DP 最多同时调度 20 条序列，后一批在前一批解码时等待。缩短到 1 个输出 token 后，64 核第 20 / 21 个变为 154 / 158 ms，192 核为 143 / 146 ms。
- **固定 DP1 的前端也有并发上限。** 32 个请求分给两个前端，每个前端只有 16 个请求，64 核时两个线程池的容量已达到每前端的请求数。增加到 192 核时，其进程数和池大小都没有增加。

复测可在同一组可用物理核内依次执行以下命令。运行时的 NUMA 节点和实际核号由本机拓扑决定。

```bash
./run-stages.sh --core-sweep 1 64 192 --out stages-scaling
./run.sh --dp 1 --core-sweep 1 64 192 --concurrency-per-replica 32 --max-num-seqs 20 --keep-diagnostics --out dp1-scaling
./run.sh --dp 8 --core-sweep 64 192 --concurrency-per-replica 32 --max-num-seqs 20 --keep-diagnostics --out dp8-scaling
./run.sh --dp 8 --core-sweep 64 192 --new-tokens 1 --concurrency-per-replica 32 --max-num-seqs 20 --keep-diagnostics --out dp8-short
```

## 图片和文本分阶段的整机对照

`./run.sh --core-sweep 32 192 --separate-image-text --keep-diagnostics --out serial` 先让每个前端完成本轮所有图片，再等所有 DP 的前端就绪，随后开放 Chat Template 和 tokenizer。提前完成图片的请求等待整批图片的时间计入图片行。每个前端继续使用所在 DP 的 CPU 掩码，两种线程池各自按 DP 核数和在途请求数分配线程槽。线程槽代表可并发的任务数，实际活跃核数可在 `core-usage.md` 核对。此模式用于实验，生产 vLLM 的请求之间可以重叠。

对照命令为 `./run.sh --core-sweep 32 192 --full-frontend-pools --keep-diagnostics --out overlap`。它保留相同的核组、前端进程数和两个池大小，允许一条请求的文本阶段和其他请求的图片阶段并行。
