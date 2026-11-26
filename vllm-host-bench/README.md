# vLLM GLM-5.2 无卡 Host 基准

默认拓扑为 8 个独立 Host 副本，每个副本有一个或多个本地前端、1 个 EngineCore 和 1 个 worker，对应 8 个 DP、每个 DP 1 个模拟卡位。每副本 32 个在途请求，整机并发 256。每 DP 的 Scheduler 同时运行至多 20 条序列。每请求包含 32,768 个文本输入 token、1,024 个输出 token 和 1 张本地 1024×1024 JPEG。两台机器使用相同版本、输入和并发。

文本使用 GLM-5.2 tokenizer、chat template 和原版 vLLM Scheduler。worker 在 CPU 上构造 token、位置、序列长度和逻辑块表。固定 token 驱动状态更新、反分词和响应序列化。模型权重、设备算子下发、执行图、卡上计算、H2D/D2H、HCCL、真实 KV 数据通路和生产 DP 协调均跳过。

## 运行和输出

在完整离线套件根目录运行。默认 DP8 / TP1，前端进程、分词线程和媒体线程自动配置。

```bash
# 使用当前 cpuset 内的全部物理核
./run.sh --out host-auto

# 依次使用 48、96、192 个物理核
./run.sh --core-sweep --out auto-core-sweep

# 定位问题时只跑两个低核数档位
./run.sh --core-sweep 48 96 --keep-diagnostics --out auto-check
```

可追加 `--plan-only` 查看计划。无 `run.sh` 时，在已安装的 `vllm-host-bench` 目录用 `.venv/bin/python host_capacity.py` 加同样的参数。

CPU 按 sysfs 的 SMT 兄弟关系去重，优先让每个 DP 留在同一个 NUMA 节点，DP 之间的核组互不重叠。给定整机核数时，每个 DP 获得相同的物理核数。每 DP 有一个前端且 `--cores-per-replica` 等于 `TP + 2` 时，三个角色各固定在一核。默认全核及其他核数使用共享核组。已有输出目录会自动使用数字后缀。

默认成功后输出目录包含 `coverage.md`、`prefill.csv` 和 `decode.csv`，内容为两张阶段表。各档使用相同工作负载和预热设置。每轮将同一行的子项和 8 个 DP 的绝对墙钟区间合并，重叠时间只算一次。每轮先除以该轮完成的请求数，再跨轮取平均，得到每请求摊销墙钟，单位 µs/请求。默认 5 轮。`—` 表示未执行或无样本。父阶段和子阶段可能覆盖同一段工作。图片编码作为独立的同步参考轮，在所有 Host 轮完成后测量，同样按完成请求数归一化。失败时保留本次测试输出目录中的 `.runs` 日志。追加 `--keep-diagnostics` 时，成功后也保留 `.runs/cores-*/host-rounds.csv` 和 `replica-*/summary.json` 等原始统计，终端仍输出两张阶段表。

整轮墙钟从最早 DP 开始到最后一个请求的最后一段响应完成本地序列化。每个副本的 `diagnostic_tail_ns` 记录业务完成到收齐诊断的额外时长。Decode 原始打点中，`schedule` 计调度，`worker_cpu_metadata` 计模拟 worker 的输入准备，`engine_worker_reply_acquire` 计引擎等待读取共享内存的墙钟及调用线程 CPU；后者包含在 `engine_worker_reply_wait` 中。

默认每请求使用随包的同一张样图。可以用 `--image-file` 替换为本地 JPEG 或 PNG，用 `--request-file` 替换文本，或者用 `--text-only` 跳过图片。双机对照使用相同的输入文件。请求 JSONL 每行一个 OpenAI 样式的 `messages` 数组，行数等于每副本并发，应用 GLM 模板后必须满足指定的文本 token 长度。

## 单个 DP 分到 1 核或多核时的影响

标准负载为 DP8 / TP1，每 DP 同时提交 32 条请求。每个副本的前端进程、EngineCore、worker 和线程都在自己的 CPU 核组内。分到 1 核时所有工作在这一核上轮流运行。分到多核后，前端按请求异步处理。不同请求的图片解码和分词可以同时运行，同一请求仍先解图、再做模板和分词。默认场景中，输入准备完成即发给 EngineCore，调度和前端前处理可以重叠。前端同时接收已入队请求的输出。L1 命中场景先统一构建固定前缀，再按该场景的批次调度。

请求经上游 `OnlineRenderer.render_chat`、`HfRenderer.render_chat_async` 进入 vLLM 自带的执行器和 tokenizer 副本池。模板使用 Transformers 的 GLM-5.2 逻辑，分词使用 RC5 的单文本 `encode`，与分词单测的底层入口一致。流式反分词保留 vLLM 的输出处理，底层使用 RC5 `DecodeStream`。图片复用 vLLM 的媒体线程池。Schema、模型输入校验、反分词和序列化由各个前端的主线程完成。

每 DP 先为 EngineCore 和 TP worker 预留 `1 + TP` 核的预算。自动配置按每 16 个在途请求配置一个前端进程（向上取整），同时每个进程至少分到 4 核预算。预算不足时仍启动一个前端。各前端各有一个 uvloop 事件循环，在接收请求前调用与 vLLM API 服务相同的 `freeze_gc_heap()`。图片和文本使用各自的线程池，两个池可以同时执行不同请求，共享本 DP 的 CPU 掩码。

标准 DP8、每 DP 并发 32 时，96、128、192 核均为每 DP 两个前端，每个前端的渲染线程数依次是 4、6、10，媒体线程池大小与渲染线程池相同。48 核时每 DP 一个前端，每个池 3 个线程。增加前端进程可以让多个请求同时执行主线程上的校验和建状态。本地 RC5 修正版在 `TOKENIZERS_PARALLELISM=false` 时让一次分词在调用线程内完成。原版长文本路径仍可能启动自己的线程池。Decode 使用 vLLM 真实 CPU Scheduler 的同步模式，worker 仍是 Host 元数据实现。并行客户端编码只跑独立参考轮。

### Prefill

| 流程和特性 | 单 DP 扩核的预期影响 | 相关配置 |
| --- | --- | --- |
| 请求接入、参数校验 | 小。同一前端主线程校验，不同前端可并行 | `--api-server-count` 调整前端进程数 |
| Chat Template、文本 Tokenizer | 真实 GLM-5.2 tokenizer 逐条编码，不同请求在前端线程池并行。摊销墙钟可随分词线程数增加而降低 | 基准 `--renderer-num-workers` 按核数自动计算，`--api-server-count` 调整前端数。生产可另用 `VLLM_USE_FASTOKENS` 更换 tokenizer 后端 |
| 图片接入与编解码合计 | 服务端 Base64 解码、图片解压和 RGB 转换的时间并集 | 图片线程按核数和并发自动计算，独立 JPEG 编码使用相同的前端并行预算 |
| 图片 Base64 编码（独立样本） | 小。pybase64 在每个图片任务内编码，多个图片任务可并行 | 图片编码后的字节数 |
| 多模态 Processor（resize、归一化、patch 等） | 取决于生产 Processor。此版 vLLM 每个 renderer 用独立单线程 executor，单前端加核可减少争抢以及与分词重叠，GLM-5.2 文本基准保留空值 | 生产 `--limit-mm-per-prompt`、`--media-io-kwargs`、`--mm-processor-cache-gb` 改变输入量和缓存 |
| 前端与引擎间 IPC、请求入队 | 等待和排队可能明显下降，单条序列化、解码和入队的 CPU 时间通常变化小 | 基准由 `--cores-per-replica` 提供调度余量 |
| 请求调度、batch 组织、token 预算分配 | 小。每个 EngineCore 单路调用 `schedule()`，单次调度的工作可以随 batch 形状变化 | 基准 `--chunk-tokens` 默认 16384，`--max-num-seqs` 默认 32。生产 `--max-num-batched-tokens`、`--max-num-seqs`、`--async-scheduling` |
| 前缀缓存匹配、KV 逻辑块分配 | 小。由同一个调度线程执行 | 生产 `--enable-prefix-caching` 和 `--block-size` 改变工作量 |
| 准备输入 tensor、位置、序列长度和块表 | 小。当前 TP1 worker 的准备逻辑为单路，更多核能减少 worker 等待 CPU 的时间 | 基准 Torch 单线程。生产 Ascend 的 `enable_cpu_binding` 控制 NPU worker 的绑核 |
| 输入及元数据 H2D | 无卡基准跳过 | 取决于设备传输实现 |
| 算子下发、执行图启动 | 无卡基准跳过 | 生产与设备运行时、执行图和 worker 绑核有关 |
| 视觉 Encoder、特征投影 | 无卡基准跳过 | 取决于模型和设备 |
| Token Embedding、GEMM、Norm、RoPE、FFN 等 | 无卡基准跳过 | 取决于模型和设备 |
| Prefill Attention 计算 | 无卡基准跳过 | 取决于模型和设备 |
| 写入新 KV Cache、读取已缓存的 KV | 无卡基准跳过 | 取决于设备 KV 读写 |
| 多卡集合通信（AllReduce、AllGather 等） | 无卡基准跳过 | 取决于生产并行拓扑和通信库 |
| LM Head、logits 计算 | 无卡基准跳过 | 取决于模型和设备 |
| 常规 top-k／top-p、随机／贪心采样 | 无卡基准跳过 | 取决于采样配置和设备 |
| 回传 token ID 和所需 logprobs | Host IPC 的等候可能缩短，worker 把固定 token 打包入队的 CPU 时间通常变化小 | 基准 `--cores-per-replica`。卡侧回传走设备路径 |
| 请求状态更新、停止条件检查 | 小。由 Core 单路处理 | 请求状态和停止条件决定工作量 |
| Detokenize、文本停止串检查、响应序列化 | 单段 CPU 时间变化通常小，前端排队可能下降 | 生产 API 前端数、输出内容和停止串设置 |
| 可选：PD 控制消息、传输元数据交互 | 无卡基准跳过 | 取决于生产 PD 和 connector 配置 |
| 可选：KV 卸载／加载、向 Decode 端传输 Prefill KV | 无卡基准跳过 | 取决于生产 PD 和 connector 配置 |
| 补充：Core 与 worker 的 Host IPC | 大概率对 1 核最敏感。Core 等回复覆盖队列等待、worker 执行和回复，多核可以让收发进程同时运行 | 基准 `--cores-per-replica`。生产 `VLLM_MQ_MAX_CHUNK_BYTES_MB` 控制共享内存消息块大小 |
| 补充：Prefill 前处理合计 | 看整批图片解码、分词、校验和入队前处理的并行收益 | 两个线程池和前端进程数 |
| 补充：Prefill 后处理合计 | 每个进程内多为串行，Core 和前端可在多核上重叠，主要体现在完成时间 | 基准 `--cores-per-replica` |

### Decode

| 流程和特性 | 单 DP 扩核的预期影响 | 相关配置 |
| --- | --- | --- |
| 本轮调度、batch 更新、进程间控制消息 | 等 worker 回复可能明显缩短，调度和 batch 更新的 CPU 时间通常变化小。每步重复的等待会影响输出吞吐和完成 P95 | 基准 `--cores-per-replica`，生产 batch 预算和异步调度 |
| KV 逻辑块扩展、块表管理 | 小。由 Core 单路执行 | 生产 `--block-size` 和 batch 形状 |
| 准备上轮 token、位置、序列长度等输入 | 小。worker 的元数据准备为单路 | 基准 Torch 单线程，生产 Ascend worker 绑核 |
| 必要的输入及元数据 H2D | 无卡基准跳过 | 取决于设备实现 |
| 算子下发、执行图启动 | 无卡基准跳过 | 与设备运行时、执行图和 worker 绑核有关 |
| Token Embedding、GEMM、Norm、RoPE、FFN 等 | 无卡基准跳过 | 取决于模型和设备 |
| 读取历史 KV Cache | 无卡基准跳过 | 取决于设备 KV 读写 |
| Decode Attention 计算 | 无卡基准跳过 | 取决于模型和设备 |
| 写入当前 token 的新 KV | 无卡基准跳过 | 取决于设备 KV 读写 |
| 多卡集合通信（AllReduce、AllGather 等） | 无卡基准跳过 | 取决于生产并行拓扑和通信库 |
| LM Head、logits 计算 | 无卡基准跳过 | 取决于模型和设备 |
| 常规 top-k／top-p、随机／贪心采样 | 无卡基准跳过 | 取决于采样配置和设备 |
| 回传 token ID 和所需 logprobs | Host IPC 的等候可能缩短，打包入队的 CPU 时间通常变化小 | 基准 `--cores-per-replica`。卡侧回传走设备路径 |
| 请求状态更新、EOS／长度等停止条件检查 | 小。由 Core 单路处理 | 终止条件和 batch 中的请求数 |
| 增量 Detokenize、文本停止串检查 | 单次处理的 CPU 时间通常变化小，前端更容易与 Core、worker 同时运行 | 生产 `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` 调整事件循环每次处理的输出数量 |
| 流式响应序列化、网络返回 | 本地序列化的 CPU 时间通常变化小，进程争用可能影响输出延迟。基准未测网络返回 | 生产 `--api-server-count` 和输出 chunk 设置 |
| 请求结束后释放 KV 逻辑块 | 小。由 Core 单路执行 | KV 块几何和已完成请求数 |
| 可选：PD 控制消息、传输元数据交互 | 无卡基准跳过 | 取决于生产 PD 和 connector 配置 |
| 可选：KV 卸载／加载、接收远端 Prefill KV | 无卡基准跳过 | 取决于生产 PD 和 connector 配置 |
| 补充：Decode 后处理合计 | Core 和前端各自串行，更多核可减少两者与 worker 的争抢 | 基准 `--cores-per-replica` |

生产 API、NUMA 和媒体变量的适用范围见 [vLLM Serve 参数](https://docs.vllm.ai/en/v0.30.0/cli/serve/) 和 [Ascend CPU 绑核](https://docs.vllm.ai/projects/ascend/en/v0.30.0/user_guide/feature_guide/cpu_binding.html)。

## 图片的测量边界

每个含图请求在自己的文本处理前调用 vLLM `MediaConnector.load_from_url_async`。输入为本地 fixture 生成的 `data:image/...;base64`，通过现有媒体线程池，实际执行 Base64 解码、Pillow 解压和 RGB 转换。「图片接入与编解码合计」仅计业务轮里 Base64 解码和图片解压、RGB 转换的时间并集。独立 Base64 编码仍单列，独立 JPEG 编码保存在原始诊断中。样图读取在正式轮次前完成。客户端编码作为独立参考，在正式 Host 测量之后同步执行。HTTP、OpenAI 图片字段 JSON 解析、图片 URL 网络下载和模型侧的视觉处理均未计入。

## 离线安装和版本

Linux x86_64 或 aarch64、glibc ≥ 2.28。完整包带两种架构的离线材料，含 uv、独立 Python 3.12.16、CPU PyTorch 2.10.0、锁定 wheels 和固定 vLLM 源码。无需镜像或卡。若只带独立脚本包和某架构的离线包，可以按下面方式安装，ARM 将名称换成 aarch64。

```bash
tar -xzf vllm-host-bench-native.tar.gz
tar -xf vllm-host-offline-x86_64.tar
cd vllm-host-bench
./install-native.sh --offline ../vllm-host-offline-x86_64
```

有网时可运行 `./install-native.sh`，按需要设置 `HTTPS_PROXY`、`HTTP_PROXY`、`UV_SYSTEM_CERTS=true`。如需自行制备对应架构的离线包，用 Python 3 和 pip 运行 `./prepare-offline.sh x86_64 ../vllm-host-offline-x86_64`，ARM 替换架构即可。

固定 vLLM core 提交 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`。随包的 `reference-glm52-950dt-tp8.sh` 保留 GLM-5.2 的真实 8 卡 TP8 配方供核对。基准的 DP8 / TP1 仅复现 8 个独立 Host 路径，生产使用的 DP/TP/EP 配置需按设备和模型部署方案确认。[vLLM 数据并行部署](https://docs.vllm.ai/en/v0.30.0/serving/data_parallel_deployment/) · [vLLM Ascend CPU 绑核](https://docs.vllm.ai/projects/ascend/en/v0.30.0/user_guide/feature_guide/cpu_binding.html) · [GLM-5.2 部署配方](https://docs.vllm.ai/projects/ascend/en/v0.30.0/tutorials/models/GLM5.2.html)

## 单流程扩核、业务占核和双机对照

在已安装的离线套件根目录运行。双机使用同一套件与相同命令。正式单测默认比较 1 个物理核与当前 cpuset 内全部物理核。整网沿用 DP8 / TP1、每 DP 并发 32、32K 文本、1024 输出和一张 1024² JPEG。各测试串行执行。

```bash
./run-stages.sh --out stages
./run.sh --keep-diagnostics --out host-auto

# 定位时可以只使用 1、48、96 核
./run-stages.sh --core-sweep 1 48 96 --out stages-diag
./run.sh --core-sweep 48 96 --keep-diagnostics --out host-diag

# A 为 x86、B 为 ARM。用两份单测 JSON 输出 B/A 吞吐比。
./run-stages.sh --compare x86/stages.json arm/stages.json
```

单测在 `stages.md` 给出吞吐和实际占核，`stages.json` 包含采样轮、各核的亲和掩码、输入指纹、单位工作 CPU 时间、内核态时间、缺页和时钟频率采样。Linux 允许访问 PMU 时还记录每单位操作的用户态周期、指令数和通用 cache-misses。内核不支持或限制读数时记录错误原因。各架构的通用缓存事件可能映射到不同硬件层级。CPU 比较先看同架构扩核前后的变化。

默认单流程表有 8 行。请求与文本前处理在一次调用内完成 Schema、模板、32K 分词及前端建状态；图片接入在一次调用内完成 Base64 解码和 RGB 图片加载。通信拆成前端↔Core 与 Core↔worker 往返，输入准备拆成 Prefill 分块与 Decode 单序列步。调度和输出处理分别以完整请求为单位。扩核倍数为最大实测档吞吐除以单核吞吐。显式选择 `--stages` 时仍显示所选细项。

整网前端按请求记录墙钟。请求与文本前处理包含本地提交后等待事件循环、校验，以及媒体返回后至文本建状态完成；图片行从调用媒体处理到结果返回，包含排队。两行相加得到本地提交至前端准备完成的平均延迟。引擎、通信和 worker 行仍为参与该操作的请求记录已打点耗时。

整网输出同名的六类 CPU 阶段。Decode 从通信开始，共四类。Prefill 按每个请求从接入到首 token 所经历的已打点操作求和，包含该请求的所有输入分块。Decode 按每个请求首 token 之后的一次输出步统计。每个批操作的时间按参与请求全额计入，混合批分别计入各请求当前所处的阶段，之后汇总所有 DP、所有正式轮次并除以请求数或实际后续输出 token 数。通信行计入发送、入队和解包，等待时间在诊断中查看。「阶段合计」逐列相加本阶段各组显示值，保留两位小数。组间并行区间可以在多个组中计入，等待也可能落在所有打点之外。请求延迟、首 token 时间可在逐轮诊断中查看。端到端吞吐取各正式轮吞吐中位数。

单测加 `--keep-diagnostics` 时补跑共享解释器及各细项，另存 `stages-details.md`。整网加同一开关时在 `details/coverage.md` 保留历史细表，包含原 Prefill 前处理合计和等待 worker 的打点。

逐核进程用例为每个选中物理核提供独立解释器和持续工作，测每个阶段能达到的吞吐。线程用例把同一组物理核分给至多 8 个进程，每核一个固定线程，用于复现共享解释器的影响。传输用例至多 8 对，两端共享各自的 DP 核组；1 核时两端在同一核轮流运行。它们分别回答独立容量、线程扩展和跨进程往返三个问题。单个资源争用阶段可能在高核数时吞吐持平甚至下降。独立内存对照每次读取和写入各 32 MiB，结果可乘 64 MiB 得逻辑带宽。

保留诊断时，每个 `.runs/cores-N/core-usage.md` 给出真实业务各组每完成请求的调用线程 CPU 时间、整轮平均占用物理核当量，以及能从原始区间恢复出的组内活跃期间平均占核。后者将 CPU 时间除以各 DP 该组墙钟区间的并集。文件末尾列出六个同步前处理打点的用户态／内核态、缺页与调用数。`prefill-resources.json` 保留总计。异步等回复仍在业务时延表中。没有这些新打点的历史结果也可使用 `.venv/bin/python stage_profile.py <旧测试/.runs/cores-N>` 恢复已有的占核指标。

RC5 原版和本地修正版的发行标识分别为 `1.0.0rc5`、`1.0.0rc5+host.serial1`。修正版使长文本分词路径遵守基准已经设置的 `TOKENIZERS_PARALLELISM=false`，请求之间仍由 vLLM renderer 执行器并行。匹配的 Transformers wheel 仅调整本套件已有的版本适配。完整双架构包运行一次 `./install.sh` 即自动选用本机修正版。已安装套件覆盖更新文件后运行 `./install-rc5-serial.sh`，从随包的 `host-rc5-wheels/` 安装本机对应的离线轮子。构建机可用 Rust、两个 Linux GNU targets、maturin 和 Zig 执行 `vllm-host-bench/build-serial-wheels.sh host-rc5-wheels` 重建。两边使用同一种构建进行横向比较。

### 独立诊断 copy64 的绑核、页放置与带宽

在已有套件的仓库根目录运行 `./run-memory-diag.sh`。使用当前 affinity 中每个物理核的一个逻辑 CPU，自动识别 Linux x86_64 和 aarch64 的 NUMA 布局。默认每项三轮，每轮目标 1.5 秒。长缓冲区每进程占 64 MiB，128 核约 8 GiB、192 核约 12 GiB，另需 Python 与原单测加载开销。测试顺序执行各场景。已有输出目录自动加序号，终端会打印实际路径。两机各保留一份 `report.md` 即可横向比较，逐核样本、页位置、已知的 cgroup 限制和节点距离保存在同目录的 `report.json`。

- 原入口 `run-stages.sh` 只运行 `copy64`，保留原分配及计时逻辑，在窗口外增加页查询。独立入口使用相同大小、类型和初始化的 NumPy 数组，父进程约定统一起跑时间。两者都以已完成的总字节除以实际墙钟，报告逻辑读写 GiB/s。原 0946.log 中 copy64 排在其他阶段之后，本次两个入口都单独测内存。
- 对比单核、每节点一核、每个节点单独用满、相同核数分散到多个节点以及全机。节点独跑列与全机并跑时逐节点贡献，可以区分局部慢节点和整机并发时的损失。
- 每个子进程在首次写入前绑定 CPU。另设子进程显式 MPOL_BIND、私有匿名 mmap、相邻可见节点轮换首次写入三个对照。每个源、目标各取最多 64 个分散页，分别观察计时前后落点，以执行核为基准统计本地率。优先使用只读 move_pages 查询，受限时用 numa_maps 中能明确归属单节点的映射，混合映射标为未知。显式绑定被禁止时在表中标出，其余项目继续。所有内存策略限定在临时子进程内。
- 同一对本地数组上比较 NumPy、循环内无需读时钟的固定次数 NumPy、libc memcpy 和按位异或零的独立 NumPy ufunc。两端结果逐项检查。另有 32 KiB 源和目标的小工作集对照。若仅固定次数快，结合启动偏斜调查计时路径；若只有 xor0 快，应检查拷贝内核选择；若强制绑内存同时提高了本地率和吞吐，应优先检查内存策略。若页已本地且多种实现都慢，比较各节点分别运行和同时运行的数据，再调查内存通道配置、共享带宽和外部负载。单靠这些软件计数无法确定 DIMM 或内存控制器硬件根因。

进程、线程、自动分核、六个阶段的调用链及扩核复测见 [CPU-FLOW.md](CPU-FLOW.md)。
