# vLLM GLM-5.2 Host 离线套件 · Tokenizers 1.0.0rc5

套件适用于 Linux x86_64 和 aarch64。各架构包均内置 uv、Python 3.12.16、CPU PyTorch、固定版本的 vLLM 源码、依赖 wheels、GLM-5.2 tokenizer 和一张 1024×1024 图片。安装和测试支持离线环境，跳过模型权重、卡侧计算和算子下发。整机基准与分词单测均使用 `tokenizers==1.0.0rc5`。随包的 `transformers==5.14.1+host.rc5` 提供这套 GLM-5.2 基准需要的模板和配置兼容性。

## 从分支安装

在联网的 Linux x86_64 或 aarch64 机器上，从仓库根目录执行：

```bash
./vllm-host-bench/install-native.sh
```

也可以在联网机器上准备指定架构的离线依赖。将同一份检出目录和生成的架构依赖目录复制到内网机器：

```bash
./vllm-host-bench/prepare-offline.sh x86_64 vllm-host-offline-x86_64
./vllm-host-bench/prepare-offline.sh aarch64 vllm-host-offline-aarch64
```

内网 Linux 机器从检出目录运行 `./install.sh`，自动选择当前架构。模型元数据、tokenizer 和样图随分支提供。

## 使用已打包的离线套件

需要 Linux、glibc ≥ 2.28、常规 C/C++ 运行库、bash、tar、sha256sum、getconf、taskset 及可写的项目目录和 `/dev/shm`。解包加安装建议预留至少 5 GB 磁盘。可先检查可用内存和共享内存，每个主要副本的共享内存队列约 10 MiB。

```bash
uname -m
getconf GNU_LIBC_VERSION
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE
taskset -pc $$
df -h . /dev/shm

tar -xf vllm-host-test-kit-rc5-linux-$(uname -m).tar
cd vllm-host-test-kit-rc5
./install.sh
```

双架构大包使用 `tar -xf vllm-host-test-kit-rc5-linux-dual.tar`。进入同名目录后，`install.sh` 自动选择本机架构。RC5 套件解包到独立目录，当前目录下创建自己的 `.venv`。

## 两台跑相同负载

默认 DP8 / TP1、总并发 256，每个 DP 32 个在途请求，Scheduler 同时运行至多 20 条序列。每请求 32,768 个文本输入 token、1,024 个输出 token 和 1 张 1024×1024 JPEG。预热 1 轮、测量 5 轮。前端配置按每个 DP 的核数自动计算。

```bash
# 用当前 cpuset 内的全部物理核
./run.sh --out host-auto

# 两台机器对比相同核数时使用同一条命令
./run.sh --core-sweep 48 96 --out host-48-96

# 同一机器依次测试 48、96、192 个物理核
./run.sh --core-sweep --out auto-core-sweep
```

机器可用核数低于所需档位时，用 `--core-sweep 48 96` 指定适用档位。各档核数必须是 DP 数的整数倍。追加 `--plan-only` 可查看核组和自动分配。

输出目录只有 `coverage.md`、`prefill.csv`、`decode.csv` 两张阶段表的 Markdown 和 CSV。每轮合并该阶段在 8 个 DP 上的墙钟区间，重叠时间只算一次。用实测轮的总墙钟除以完成请求数。每行一个数，单位 µs/请求。卡上及未执行的阶段显示 `—`。重复使用已有的 `--out` 时，脚本自动创建带数字后缀的目录。

## 单测 tokenizer

```bash
./run-tokenizer.sh --core-sweep 1 48 96 --out tokenizer-rc5
./run-tokenizer.sh --core-sweep 1 48 96 192 --out tokenizer-rc5-large
```

单核用 1 个 DP 和 1 个分词线程。多核档位使用 8 个 DP，分词线程数等于每个 DP 获得的物理核数。每个 DP 提交 32 条 32K 输入，默认测量 10 轮。结果在 `tokenizer-results.md` 和 `tokenizer-results.json`。
