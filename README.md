# Industry Trading Agent (LangChain)

一个基于 LangChain 的行业资产配置与仓位管理回测系统。

目标：
- 输入行业行情、重大事件、券商研报
- 每个交易日自动检索相关上下文
- 调用可替换 LLM 生成行业权重
- 在模拟交易环境中执行并输出收益指标

## 1. 目录结构

```text
.
├── config/
│   └── example.yaml
├── data/
│   ├── events.csv
│   ├── market/
│   │   ├── 新能源.csv
│   │   ├── 半导体.csv
│   │   ├── 银行.csv
│   │   └── 医药.csv
│   └── reports/
│       ├── 0001_新能源_20250801.pdf
│       ├── 0002_半导体_20250803.pdf
│       └── ...
└── src/industry_trading_agent/
```

## 2. 数据格式

### 2.1 研报目录（`data/reports`）
- 支持：`.pdf` / `.txt` / `.md`
- 文件名规则：`编号_行业_YYYYMMDD.xxx`
- 示例：`0345_半导体_20250801.pdf`

### 2.2 事件文件（`data/events.csv`）
必需列：
- `date` (YYYY-MM-DD)
- `industry`
- `title`
- `content`

也支持 `json/jsonl/txt`：
- `json/jsonl` 每条记录字段一致
- `txt` 每行格式：`YYYY-MM-DD|industry|title|content`

### 2.3 行情目录（`data/market`）
- 每个行业一个 CSV，文件名 `行业.csv`
- 必需列：`date`, `close`

## 3. 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 4. 运行

设置模型密钥（OpenAI 示例）：

```bash
export OPENAI_API_KEY="your_key"
```

执行回测：

```bash
trade-sim --config config/example.yaml
```

## 5. 输出

默认写入 `output.output_dir`，包括：
- `daily_records.csv`：每日 NAV、PnL、换手、执行仓位
- `decisions.json`：每日 LLM 推理与目标仓位
- `summary.json`：总收益、年化收益、波动率、夏普、最大回撤

## 6. 可替换 LLM

在 `config` 中切换：
- `llm.provider: openai`
- `llm.provider: rule_based`（离线规则模型，便于先联调流程）
- 或 `llm.provider: custom`，并在 `llm.extra` 提供 LangChain 兼容 `BaseChatModel` 类路径

示例：

```yaml
llm:
  provider: custom
  extra:
    class_path: your_package.your_module:YourChatModel
    init_kwargs:
      model: your_model
      temperature: 0.0
```

## 7. 系统流程

1. 加载外部文件（研报、事件、行情）
2. 按交易日构建窗口上下文（lookback）
3. LLM 输出行业目标权重 JSON
4. 模拟交易环境执行调仓与交易成本
5. 记录收益并导出结果

## 8. 注意

- 当前策略是 long-only 行业权重分配。
- 风控约束：单行业上限 `max_position`，并自动归一化。
- 如果某天缺少某行业行情，该行业当日收益按 0 处理。
