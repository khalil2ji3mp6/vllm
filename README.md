# vLLM 无卡 CPU 基准

在仓库根目录运行。已安装离线环境的机器更新代码即可；首次离线安装时，将对应架构的依赖包放在仓库根目录，执行 `./install.sh`。

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
