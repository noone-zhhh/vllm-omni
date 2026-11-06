# Fish DAC 的 NPU MRV2 迁移方案

状态：依赖已集成；已实现 opt-in 配置、配置回归测试、合成 codec 的完整窗口交接测试、
NPU V1/V2 在线验收测试及 nightly 入口。NPU 硬件验收和质量/性能比较尚未运行。

## 目标与发布方式

完成 [#9446 的 D1](https://github.com/vllm-project/vllm-omni/issues/9446)：
让 Drama3 的独立 DAC 阶段通过 NPU MRV2 generation runner 执行。
第一版保留 NPU 默认 DAC V1，提供显式 V2 profile；验收通过后再提默认切换。
这一方式与 [#9831 的 review](https://github.com/vllm-project/vllm-omni/pull/9831#pullrequestreview-5760186740) 一致。

Stage 0、权重、codec、窗口 ramp、dtype 和设备布局保持现有基线，便于比较。
首版使用 eager DAC；ACL graph、stateful DAC 和算子优化作为后续工作。
Fish Speech 共用 decoder，但 Drama3 的验证不能代替 Fish Speech 的模型级资格验证。

## 集成基线

开发分支：`feats/fish_dac`，记录日期：2026-11-06。

| 来源 | SHA | 用途 |
| --- | --- | --- |
| origin/main | `3bde1414c20e87dbba33f059cbc05c2acf508e85` | 主线基线 |
| [#9831](https://github.com/vllm-project/vllm-omni/pull/9831) | `e6352e4adf5ef5451ce150bfb0caf63f0ba63f25` | NPU MRV2 generation runner 与平台 hooks |
| [#9625](https://github.com/vllm-project/vllm-omni/pull/9625) | `92b36f8e6f35966243bbaab6287f622cc39151fb` | Fish DAC native MRV2 接收、窗口调度与 codec 预加载 |

本地 main 的旧版 #9831 合并记录未重写。此前四个文件的导入修改已备份至
stash；新依赖已直接使用 `vllm_ascend.worker.v1.model_runner`，因此无需重放旧补丁。
提交上游前，应在依赖合入后整理自己的提交，避免把依赖代码重复计入 D1 diff。

## 数据与执行链

```text
Drama3 Stage 0 / NPU MRV2 AR
  → CPU codes.audio [10, W] + meta
  → SharedMemoryConnector
  → native MRV2 receive / scheduling metadata
  → NPUGenerationWorker
  → NPUOmniGenerationModelRunner
  → FishSpeechDACDecoder / NPU eager
  → request 对应的 float32 波形 → 同步物化到 CPU → 客户端
```

`W = C + N`：C 是历史帧数，N 是新增帧数。
非空二维窗口只占一个调度 token，真实 codes 保留在 runtime information 中。
元信息包含 `left_context_size`、`chunk_seq` 和 `finished`。
普通 NPU 路径重解码历史帧并裁掉对应音频，每个新增帧应输出 2048 个采样点，
采样率 44100 Hz。空结束窗口 `[10, 0]` 必须清除旧 codes，不能再次生成旧音频。

## 工作包一：复用公共实现并验证边界

以下内容已由依赖提供，不重复实现：

- Fish pipeline 的 `supports_native_mrv2_data_plane=True`。
- native async receiver 的二维窗口占位调度。
- DAC 的 `capture_auxiliary_graphs()` codec 预加载 hook。
- NPU worker 根据 `use_v2_model_runner` 选择 V1/V2。
- Ascend 输入准备、dummy batch、forward context 和 KV 初始化接入。

需要针对 Fish 核对的边界：

- 新 runner 拒绝非空 `slot_mapping`。Fish codec 内部有 Transformer，
  但应区别于 runner 管理的 paged KV：验证 Fish 实际不要求后者，
  不得直接删除断言以绕过初始化失败。
- 验证 MRO 中 load、capture、dummy run、profile 的实际调用路径；
  优先复用 #9831 的测试，不新增一套 Fish 专用 runner。
- 验证辅助预加载 hook 在服务 ready 前执行，设备、dtype 正确，
  codec 内存分配被启动内存核算覆盖。
- NPU 应继续关闭 CUDA 专属 graph/stateful 路径，不触发 CUDA 内核或编译。
- 当前公共 MRV2 runner 仅对 CUDA 启用异步 D2H，NPU 走同步物化路径；
  不把 `async_scheduling` 等同于异步输出拷贝。验证后续 forward、取消和
  输出丢弃不会破坏已有输出，性能比较需包含这一实际路径差异。

只有发现可复现缺口时才修改公共 runner，并补与缺口对应的回归测试。

## 工作包二：显式 NPU V2 配置

已新增 `vllm_omni/deploy/fish_omni_npu_mrv2.yaml`：

```yaml
base_config: fish_omni.yaml

platforms:
  npu:
    stages:
      - stage_id: 1
        model_runner: v2
        enforce_eager: true
        compilation_config:
          cudagraph_mode: NONE
```

配置解析回归测试覆盖以下要求；需在具备 vLLM 的环境运行：

- 基础 `fish_omni.yaml` 的 NPU stage 1 仍为 V1。
- 新 profile 的 NPU stage 1 为 V2/eager/NONE。
- Stage 0 的 MRV2、TP/EP、图模式，以及两个阶段的设备和内存参数未丢失。
- CUDA/其他平台的默认行为不受影响；文档只声明已验证的 NPU 使用方式。

## 工作包三：测试与硬件验收

### CPU / 配置及交接测试

复用并扩展现有测试位置：

- `tests/config/test_config_factory.py`：基础配置和 opt-in profile 的解析结果。
- `tests/worker/test_scheduling_metadata_adapter.py`：窗口占位和空窗口语义。
- `tests/model_executor/stage_input_processors/test_fish_omni_async_chunk.py`：
  ramp、尾包及 native handoff，保持每个 request 的序号和 metadata 一致。
- `tests/model_executor/models/fish_speech/test_fish_speech_regressions.py`：
  codec 预加载 hook 和输入契约。
- `tests/platforms/npu/test_npu_qwen3_tts_mrv2.py` 已覆盖部分公共 runner 接入；
  Fish 新用例仅补 Fish 特有边界，避免依赖 Qwen3-TTS 测试作为唯一资格证明。

窗口用例覆盖新增帧 6/8/12/16/24/32/64、历史帧 0/部分/32、尾部不足窗口、
空结束窗口、不同长度混合 batch、请求重排和调度预算不足时整体延后。

### 固定 codes 的 NPU V1/V2 比较

先固定同一组 codec codes，通过各自 runner/接收路径解码，隔离 Stage 0 采样差异。
固定窗口切分、dtype、设备和 codec 权重，核对每个 request 的顺序、样本数、
最大/平均数值误差和有限值。先测重复运行的确定性，再制定有依据的误差范围。
已增加 `tests/platforms/npu/test_fish_dac_mrv2.py`：真实 codec 解码固定窗口后，
分别调用 V1/V2 的 `sample_tokens` 并在下一次 decode 后检查输出一致性。
它使用合成调度状态，只验证 runner 的输出边界；完整输入与引擎路径由在线测试覆盖。

### Drama3 端到端

已新增 `tests/e2e/online_serving/test_fish_omni_npu_mrv2.py`，针对 V1/V2 profile 覆盖：

- 首请求、重复请求、完整输出和流式输出。
- 生成足够多帧以覆盖全部 ramp、64 帧稳定窗口和尾包。
- 多请求混合长度与 batch 重排，无音频串流、重复或遗漏。
- 首包后取消、处理中取消、并发中部分取消；检查请求指标归零及后续服务可用。
- 指标缺失须单独报错，不能把缺失误判为清理成功。
- 取消后的重复请求与请求指标归零；长期设备内存稳定性仍需硬件测量。

硬件初始目标为用户计划使用的 Ascend 910B；记录实际卡型、卡数和显存。
基线与候选使用同一集成代码，唯一主要变量为 Stage 1 的 V1/V2 选择。
先做 C=1/4/16，再根据实际容量扩展到目标并发；保持 Stage 0 参数一致。

记录 TTFA、块间隔/播放断流、RTF、音频吞吐、设备内存峰值；进行充分预热和
重复运行，报告分布和波动。质量按 #9446 要求测 WER 与 speaker similarity，
不能用固定 codes 数值测试替代整条服务的质量验证。

每份结果记录 Omni、vLLM、vLLM-Ascend、PyTorch、torch-npu、CANN、模型 SHA、
硬件、完整配置、执行模式、数据集、命令和重复次数；未测项显式标注。
新增 NPU nightly A3 双卡任务，尚未触发或取得 CI 结果；合入时由平台负责人确认资源安排。
不以本机语法检查或合成 codec 测试代替硬件资格。固定 codes 的硬件测试已编写但未运行；
完整引擎路径验收、长期内存测量和质量/性能实验仍是默认切换前的待办。

## 执行顺序与退出条件

1. 依赖集成和静态检查：当前分支已完成集成，硬件测试未运行。
2. 实现 opt-in profile 和配置测试，确认实际 runner 选择。
3. 跑通启动、固定 codes 和单请求；据实际失败修复平台边界。
4. 完成流式、并发、长输入、取消和状态释放测试。
5. 完成相同 NPU 工作负载的 V1/V2 质量与性能比较。
6. 更新 Drama3 recipe，给出启动、回退、限制与复现结果，提交 opt-in PR。
7. 达到 #9446 的质量、性能和正确性标准后，另行决定默认 V2 切换。

验收底线：无崩溃/挂起、无请求串流和重复窗口、样本数正确、取消后资源可回收。
默认切换还要求质量处于 V1 重复波动范围，RTF/TTFA/吞吐无回退。
依赖通过 Qwen3-TTS 验证不能代替 Fish DAC 的这些证据。

## 本次本地检查

实现基于 `ec9ac458f`，在 macOS 上检查，未使用 NPU 或模型权重：

- 变更文件的完整 pre-commit 通过，包括 Ruff、mypy 3.10、Markdown、
  SPDX、测试标记、禁止导入、CUDA 调用检查及 Buildkite schema。
- Python 语法、YAML 解析、`git diff --check` 通过。
- 隔离运行生产代码的 YAML 继承函数通过：Stage 0、connector、其他平台配置保持一致，
  NPU Stage 1 为 V2/eager/NONE。此检查不代替完整 StageConfigFactory 测试。
- 完整 pytest 在 fixture 加载阶段被临时环境缺少 `torch` 阻断；本机也没有
  `vllm` / `vllm-ascend` 运行环境，因此不报告 pytest 或硬件通过结果。

具备项目运行依赖后，先执行 CPU 回归：

```bash
pytest -q tests/config/test_config_factory.py \
    tests/model_executor/stage_input_processors/test_fish_omni_async_chunk.py \
    -k fish_omni --run-level core_model
```

硬件测试命令与 V1 回退方法见 Drama3 recipe。当前实现只作为显式 opt-in，
没有提交质量、性能或稳定性达标的声明。
