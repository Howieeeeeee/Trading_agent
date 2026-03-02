# Industry Trading Agent (LangChain)

面向“代码走 Git、原始数据只在 Linux 服务器”的行业交易回测系统。

## 1. 运行逻辑（按你的场景）

- `T` 日读取信息（行业行情窗口、事件、研报摘要）
- LLM 在 `T` 日给出目标仓位
- 在 `T+1` 交易日执行该仓位并计算当日收益
- 循环整个观察窗口，输出累计收益与风险指标

这避免了同日信息同日收益的前视偏差。

## 2. 配置入口

配置文件：
- [example.yaml](/Users/howie/Documents/New project/config/example.yaml)
- [rule_based.yaml](/Users/howie/Documents/New project/config/rule_based.yaml)

## 3. 服务器数据路径字段

`data` 段支持：
- `report_raw_dir`: 原始研报目录（如 `.../report_pdfs`）
- `report_summary_dir`: 研报摘要目录（如 `.../report_texts`，优先读取）
- `events_file`: 事件文件（可用 `events_all.csv` 或 `significant_impact_events.csv`）
- `market_dir`: 行业板块行情目录（如 `半导体__BK1036.csv`）
- `industry_match_file`: 行业到板块映射（`industry_match_test_report.csv`）
- `market_index_file`: 大盘指数文件（如沪深300）

## 4. 当前已适配的数据格式

### 4.1 研报
- 文件名匹配前缀：`编号_行业_YYYYMMDD`
- 支持：`.txt/.md/.pdf`
- 示例：
  - `1_半导体_20251207.txt`
  - `1_半导体_20251207_original.pdf`

### 4.2 事件
- 支持 `csv/json/jsonl/txt`
- 对你给的 CSV 会自动识别：
  - 日期：`event_date` / `event_date_dt` / `report_date` / `report_date_dt`
  - 标题：`event_name` / `event_type`
  - 内容：优先 `evidence_quote`，否则拼接 `actors/action/object/location`

### 4.3 行情
- 行业文件：`行业__BKxxxx.csv` 或 `行业.csv`
- 必需列：`date, close`
- 若有 `ret` 列，优先使用该收益列

## 5. 行业选择

两种方式：
- 手动：`trading.industries: [半导体, 电池, ...]`
- 自动：`industries: []` 且 `auto_select_industries: true`
  - 在回测窗口内按事件数量排序
  - 取前 `auto_select_top_n`
  - 过滤 `min_event_count`

## 6. LLM 接口

已支持你的一套调用方式（单 key 版本）：
- `provider: openai`
- `base_url: https://api2.aigcbest.top/v1`
- `model: gpt-4o-mini`
- `api_key_env: OPENAI_API_KEY`

说明：当前 Agent 只需要一个 LLM 实例，不做多 key 并发池。

## 7. 安装与运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

```bash
export OPENAI_API_KEY="your_key"
trade-sim --config config/example.yaml
```

离线调试：

```bash
trade-sim --config config/rule_based.yaml
```

## 8. 输出

`output.output_dir` 下：
- `daily_records.csv`（含 `decision_date` 与 `execute_date`）
- `decisions.json`
- `summary.json`
