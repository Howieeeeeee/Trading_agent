# Industry Trading Agent (LangChain)

面向“代码与算法走 Git、原始数据留在服务器”的行业交易回测系统。

## 1. 你当前要的工作方式

- 本地仓库：只维护代码、配置模板、运行脚本
- Linux 服务器：存放原始数据（研报、摘要、事件、行情）
- Git 同步：你把代码同步到服务器后，只改配置里的绝对路径即可运行
- 结果输出：写入仓库内 `outputs/...`，可选择提交回 Git

## 2. 数据路径配置

使用 [config/example.yaml](/Users/howie/Documents/New project/config/example.yaml) 或 [config/rule_based.yaml](/Users/howie/Documents/New project/config/rule_based.yaml)。

`data` 字段：
- `report_raw_dir`: 原始研报目录（可为空）
- `report_summary_dir`: 研报摘要目录（可为空，且优先读取）
- `events_file`: 事件文件路径
- `market_dir`: 行情目录路径（每行业一个 CSV，文件名 `行业.csv`）

说明：
- 系统优先用 `report_summary_dir`；如果不存在则回退到 `report_raw_dir`
- 两者都不存在时，系统会在“无研报上下文”下继续运行

## 3. 研报文件命名规则

研报文件名保持：`编号_行业_YYYYMMDD.xxx`

支持扩展名：`.pdf` / `.txt` / `.md`

## 4. 运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

OpenAI 模型：

```bash
export OPENAI_API_KEY="your_key"
trade-sim --config config/example.yaml
```

离线联调（不依赖外部 API）：

```bash
trade-sim --config config/rule_based.yaml
```

## 5. 输出结果

输出目录由 `output.output_dir` 控制，默认在仓库内：
- `daily_records.csv`
- `decisions.json`
- `summary.json`

## 6. 当前已支持的数据输入

- 研报：目录扫描（按命名规则过滤）
- 事件：`csv/json/jsonl/txt`
- 行情：每行业一个 CSV，必须包含 `date, close`

后续你给出更细的数据格式后，我可以继续改 loader 适配。
