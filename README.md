# 智能 OCR 测试验证系统

> 面向 UI 自动化测试的双层 OCR 断言基础设施。

## 背景

在客户端 UI 自动化中，OCR 断言常因弹窗遮挡、图像模糊、低对比度而识别失败，是脚本误报的主要来源。误报的代价不只是一次重跑：测试人员要逐个打开报告、比对截图、判断「是产品的 bug 还是 OCR 认错了」，一旦这个比例长期偏高，团队就会停止相信自动化报告。

一个直接的解法是全量改用多模态大模型识别，但它按次计费、延迟是本地推理的百倍量级，且强依赖网络 —— 而实测中九成以上的样本，本地 OCR 本来就能完全识别正确，这部分钱花得没有价值。

本项目的解法是**快慢双层**：

- **第一层**本地 OCR 覆盖常规场景，本地推理、零 API 成本；
- 只有被判定为「不可信」的样本才升级到**第二层**多模态模型做语义级复核；
- 判定**不只看置信度**，还覆盖空识别、正则不匹配、文本框重叠三类信号 —— 因为置信度反映的是模型对自身输出的确信程度，不等于输出正确。文字被弹窗盖住时，OCR 完全可能对一个残缺的结果给出 0.98 的高置信度。
- **指纹缓存前置在第一层之前**，让重复图片同时省下两层的推理开销。

与实现同等重要的是**验证方式**：150 张自建合成 UI 评测集、固定随机种子、三方案横向对比、按扰动类型分维度拆解，目的是让「识别率提升了多少」这个结论能被任何人用一条命令复现，而不是一个只能选择相信的数字。

**实测结果**（150 张合成集，完整环境与口径见下方「实测复现记录」）：精确匹配率 **81.3% → 88.0%**，字符准确率 **90.5% → 97.0%**，第二层实际调用 20 次全部成功、零降级。提升几乎全部来自重度模糊样本（0% → 81.8%）；而重度遮挡样本在当前阈值下**一次都没有触发升级**，原因分析与改进方向一并记录在下方，不做粉饰。

---

## 能力概览

- **第一层(PaddleOCR)**：覆盖常规场景，本地推理、毫秒级、零 API 成本；内置 RapidOCR 兜底，跨平台更稳。
- **第二层(多模态模型)**：处理遮挡、模糊、低对比度等复杂场景；支持 gemini / openai / anthropic 三种协议，换厂商零改码。
- **指纹缓存(MD5 + 可选 dHash)**：同图重复计算直接命中，显著降低多模态模型调用开销；缓存前置在 PaddleOCR 之前。
- **路由不止看置信度**：补充空识别 / 正则不匹配 / 文本框重叠三类升级条件，VLM 失败自动降级首层。
- **评测体系**：三方案横向对比 + 阈值校准曲线 + 分扰动类型拆解，
  让「识别率提升」这个数字可被复现、可被验证。

> **数据口径说明**
> - 评测集为**自建合成数据**，未使用任何真实业务截图。合成数据的优势是 ground truth
>   天然准确（文字在渲染前已知）且逐像素可复现，代价是分布比真实截图窄。
> - **数字随第一层引擎版本而变，引用时必须带环境**。见下方「实测复现记录」：
>   同一套 150 张评测集，在 RapidOCR 3.x（PP-OCRv6）上是 **81.3% → 88.0%（真实 VLM）**；
>   而 `reports/` 里早期报告的 82.0% → 93.3% 来自另一套环境、且第二层是 Mock（dry-run 理想
>   上限），**不可与本表混用**。历史报告保留未删，是为了让数字的来源可追溯。
> - 0.7 阈值是**起点**，最终值应以 `benchmark/calibrate.py` 在你自己数据上的校准曲线为准。

### 实测复现记录（2026-08-14，真实 API，非 dry-run）

环境：Windows / Python 3.13.15 / 第一层 = RapidOCR 3.9.2（PP-OCRv6 small，
未装 paddlepaddle 故自动回落）/ 第二层 = `gemini-3.5-flash`，经
`api.claudecode.net.cn/api/gemini` 中转，`protocol: gemini`。

| 方案 | 精确匹配 | 字符准确 | P95 延迟 | 升级数 | 降级/报错 |
|------|---------|---------|---------|-------|----------|
| 纯第一层 | 81.3% | 90.5% | 2012ms | 0 | 0 / 0 |
| 双层（真实 VLM） | **88.0%** | **97.0%** | 3774ms | 10 | 0 / 0 |
| 双层（禁用缓存） | 88.0% | 97.0% | 3668ms | 10 | 0 / 0 |

复现命令：`python benchmark/run_benchmark.py --config config.yaml`
（结果见 `reports/benchmark_real.json`，20 次真实 API 调用全部成功）

**这组数据有三个必须一起读的限制条件**：

1. **提升几乎全部来自模糊样本，不是遮挡**。`blur_heavy` 精确匹配 0% → 81.8%
   （n=11，其中 9 个样本升级；样本量小，区间很宽，引用时请带上 n），
   而 `occlusion_heavy`（n=21）**升级数为 0**、精确匹配 57.1% 毫无变化 ——
   RapidOCR 对这些遮挡样本给出的置信度高于 0.7 阈值，也没触发文本框重叠条件，
   路由根本没把它们送去第二层。「遮挡靠第二层救」这个设计意图在当前引擎 + 阈值下**尚未生效**，
   需要用 `calibrate.py` 重新校准阈值、或收紧 `box_overlap_threshold` 才能兑现。
2. **缓存收益在本评测集上为 0**。150 张图片互不重复，缓存命中数为 0，
   所以「带缓存 / 禁用缓存」两行数字完全一致。缓存的价值要用
   「同一页面反复截图」的场景单独证明，不能引用这张表。
3. 第二层实际调用 20 次（两条链路各 10 次）全部成功，`degraded=0 / errors=0`，
   所以 88.0% 是真实模型的成绩，不是 Mock 上限。

## 快速开始

### 1. 环境准备

Python 3.10+,推荐创建虚拟环境:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
pip install -e .          # 让 `python -m ocr_verify.cli` / `ocr-verify` 可用
```

> **依赖注意事项(实测踩过的坑)**
> - `rapidocr-onnxruntime` 已停更在 Python 3.12,3.13 上要用继任包 `rapidocr`。
>   `requirements.txt` 已用环境标记自动按版本二选一,代码两个包名都兼容。
> - `paddleocr` 不会自动装 `paddlepaddle`。只装 `paddleocr` 时首次推理会报
>   "A dependency error occurred during pipeline creation",然后**自动回落 RapidOCR**
>   ——链路不会断,但第一层实际跑的是 RapidOCR。要真用 Paddle 后端请另外
>   `pip install paddlepaddle`。
> - `pip install -e .` 是 CLI 的前提:本仓库是 src-layout,不装包时
>   `python -m ocr_verify.cli` 会报 `No module named 'ocr_verify'`。

### 2. 配置 API Key

复制模板并填写:

```bash
cp config.example.yaml config.yaml
```

编辑 `config.yaml` 中的 `vlm` 部分:

```yaml
vlm:
  protocol: gemini   # 或 openai,视你的转发服务而定
  base_url: "https://api.claudecode.net.cn/api/gemini"
  model: "gemini-3.5-flash"
  api_key: "YOUR_API_KEY"
```

> **密钥只写进 `config.yaml`**(已在 `.gitignore` 中),不要写进
> `config.example.yaml` —— 后者是要进仓库的模板文件。

**关于 `protocol` 怎么选(以 `api.claudecode.net.cn/api/gemini` 为例,均为实测)**

| 请求头 | 网关响应 | 结论 |
|--------|----------|------|
| `x-goog-api-key`(gemini 协议) | 通过鉴权 | ✅ 该 base_url 用这个 |
| `Authorization: Bearer`(openai 协议) | 通过鉴权 | ⚠️ 头能过,路由需自测 |
| `x-api-key`(anthropic 协议) | `401 AUTH_MISSING` | ❌ 网关不认这个头 |

**key 的形状(哪怕是 `sk-ant-api03-` 开头)不代表要用 anthropic 协议** ——
决定协议的是网关认哪个请求头、哪条路径,与 key 的前缀无关。
只有当你的网关本身是 Claude 原生接口(`/v1/messages`)时才选 `anthropic`。

**模型名要查,不要猜**。中转站的模型清单和 Google 官方名不一致,且列表里
出现 ≠ 你的账号能调:

```bash
curl -H "x-goog-api-key: $KEY" https://api.claudecode.net.cn/api/gemini/v1/models
```

`supported_endpoint_types` 为空的模型调用会返回
`400 MODEL_PRICE_NOT_CONFIGURED`(该账号未开通),非空的才可用。

### 3. 生成评测数据集

```bash
python benchmark/build_dataset.py --count 50 --clean
```

默认生成 50 张合成 UI 截图,并自动构造 4 类扰动样本(遮挡、模糊、低对比度、JPEG/缩放)。

### 4. 运行评测

```bash
# Dry-run 模式:用 ground truth 模拟完美第二层,不消耗 API,用于验证链路
python benchmark/run_benchmark.py --dry-run

# 真实 API 模式:先用小样本确认链路与成本,再跑全量
python benchmark/run_benchmark.py --limit 20
python benchmark/run_benchmark.py

# 生成 HTML 报告
python benchmark/report.py reports/benchmark_*.json
```

> 真实模式会跑「双层(带缓存)」和「双层(无缓存)」两条链路,
> 每条链路的每个升级样本都会真实调用一次 API,全量 150 张约 150 次调用量级。

### 5. 单张图片识别

```bash
python -m ocr_verify.cli --config config.yaml recognize datasets/raw/ui_000_dialog.png

# 强制走第二层,用于单独验证 VLM 链路是否通
python -m ocr_verify.cli --config config.yaml recognize \
  datasets/perturbed/ui_000_dialog__occlusion_heavy.png --force-vlm --json
```

> 注意:`--config` 是全局参数,必须放在子命令 `recognize` 之前。
> 本命令要求先 `pip install -e .`(见第 1 步)。

## 项目结构

```
ocr-verify/
├── src/ocr_verify/           # 核心库
│   ├── cache/                # LRU 缓存 + 图片指纹
│   ├── engines/              # OCR 引擎抽象与实现
│   ├── router.py             # 双层路由决策
│   ├── verifier.py           # 对外门面
│   ├── metrics.py            # 评测指标
│   └── config.py             # YAML 配置加载
├── benchmark/                # 评测工具
│   ├── build_dataset.py      # 数据集生成
│   ├── run_benchmark.py      # 三方案横向评测
│   ├── calibrate.py          # 阈值校准曲线
│   └── report.py             # HTML 报告生成
├── tests/                    # pytest 单测
├── datasets/                 # 自动生成的数据集(不提交)
├── reports/                  # 自动生成的报告(不提交)
├── pyproject.toml            # 打包配置(src-layout,pip install -e . 用)
├── config.example.yaml       # 配置模板(进仓库,不含密钥)
├── config.yaml               # 本地真实配置(含密钥,.gitignore 已忽略)
└── README.md
```

## 核心设计决策

### 为什么缓存层放在 PaddleOCR 之前?

你最初的描述里缓存是为了省 VLM 调用,我把它挪到了 PaddleOCR 之前:
这样两层推理都能被缓存拦截。UI 自动化的典型模式是**同一页面被反复截图断言**,
前置缓存的收益比只挡 VLM 更大。

### 为什么升级条件不止置信度?

置信度反映的是"模型对自己输出的确信度",不是"输出是否正确"。
文本被遮挡时,模型可能对错误结果给出高置信度。
因此路由还包含三条补充条件:

1. 检测到文本框但识别为空
2. 不符合调用方指定的预期格式(如金额正则)
3. 文本框大面积重叠(疑似弹窗遮挡)

### 为什么 0.7 只是起点?

`confidence_threshold` 必须在**你自己的数据集**上跑校准曲线确定,
不能凭经验写死。使用:

```bash
python benchmark/calibrate.py
```

它会输出推荐阈值以及准确率/VLM 调用率曲线。

### 为什么默认关闭感知缓存?

`dHash` 感知缓存适合"同一页面、仅状态栏/时间/电量变化"的场景;
跨页面数据集上,布局相似的页面容易被误判为同一张图。
MVP 阶段默认关闭,保留实现作为可选项。

## 评测指标说明

| 指标 | 含义 | 适用场景 |
|------|------|----------|
| 精确匹配率 | 识别结果与标注完全一致 | `assert text == "..."` |
| 包含匹配率 | 标注文本是识别结果的子串 | `assert "..." in text` |
| 字符准确率(1 - CER) | 基于编辑距离的相似度 | 衡量错了多少字 |
| P95 延迟 | 95% 请求的延迟上限 | 评估断言是否拖慢用例 |
| VLM 调用率 | 走第二层的样本比例 | 成本核算 |
| 缓存命中率 | 命中缓存的请求比例 | 缓存有效性 |

## 设计决策速查

每条决策的完整论证在对应源文件的模块 docstring 里,这里只列结论与代价。

| 决策 | 结论 | 代价 / 边界 |
|------|------|------------|
| 缓存放在第一层之前 | 两层推理开销一起省 | 多存一份结果;跨页面误命中风险由默认关闭 dHash 规避 |
| 升级条件不只看置信度 | 补空识别 / 正则不匹配 / 框重叠三条 | 框重叠这条在实测中从未触发,假设待修正 |
| 指纹用 MD5 而非 SHA-256 | 非安全用途,快约 2 倍,额外拼字节长度做二次校验 | 理论碰撞会导致**静默**错误结果;强一致场景应换 SHA-256 |
| LRU 而非 LFU / TTL | UI 自动化有强时间局部性 | 无 TTL:降级结果会被缓存到进程结束(见「已知问题」) |
| 手写 LRU 而非 `OrderedDict` | 需嵌入命中率 / 淘汰数 / 省下调用数等统计与淘汰回调 | 多一份需自己维护的代码 |
| dHash 感知缓存默认关闭 | 跨页面布局相似易误命中,而误命中是静默的 | 牺牲了"仅状态栏变化"场景的命中率 |
| 第二层失败降级第一层 | 外部依赖抖动不应让整条用例失败 | 结果质量下降,靠 `degraded` 标记向调用方透出 |
| `decide()` 写成纯函数 | 决策与执行分离,单测无需图片/模型/网络 | 无 |
| 三协议抽象 | 换厂商只改配置一行 | 各协议的响应差异需各自维护提取逻辑 |
| 上传前压缩到 1600px / JPEG 85 | 按尺寸计费,实测该档位对识别无影响(`jpeg_low` 类 100%) | 极小字号文本场景需调高上限 |

## 已知问题与建议

- **PaddleOCR 初始化失败**:本项目已内置 **RapidOCR** 自动兜底。
  RapidOCR 是 PP-OCR 模型的 ONNXRuntime 移植,识别效果接近,
  但依赖更轻、跨平台更稳。在 `config.yaml` 中把 `paddle.backend` 设为
  `rapidocr` 可强制使用。
  > 实测:没装 `paddlepaddle` 时(或 Paddle 环境有问题时),`backend: auto`
  > 会打三条 "后端 paddleocr 初始化失败" 的 WARNING 然后回落 RapidOCR,
  > 链路正常但**第一层实际是 RapidOCR**。对外描述性能数字时请说明这一点,
  > 别把 RapidOCR 的结果说成 PaddleOCR 的。
- **中转网关的坑**:`protocol` 由网关认哪个请求头决定,与 key 前缀无关;
  模型列表里有 ≠ 账号能调(`MODEL_PRICE_NOT_CONFIGURED`)。详见「配置 API Key」一节。
- **评测数据**:MVP 使用合成 UI 截图,后续建议补充:
  - 自己手机/电脑的 UI 截图(脱敏后)
  - 公开中文场景文本数据集(ICDAR / CTW1500 子集)
- **真实业务截图**:请勿使用前雇主的业务截图作为个人项目数据。

## License

MIT
