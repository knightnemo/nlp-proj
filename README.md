# LLM Text-Based World Simulator

这是一个基于论文 [Can Language Models Serve as Text-Based World Simulators?](https://arxiv.org/abs/2406.06485) 的实现。

## 安装

```bash
pip install -r requirements.txt
```

## 环境配置

你需要一个 OpenAI API key 来运行代码。请将 OpenAI API key 设置为环境变量。详细说明请参见 [OpenAI 文档](https://platform.openai.com/docs/quickstart?context=python)。

## 项目结构

```
.
├── data/               # 数据文件夹
├── experiments/        # 实验脚本
├── rules/             # 规则文件
├── scripts/           # 分析脚本
├── paper/             # 论文相关代码
├── requirements.txt   # 项目依赖
└── README.md          # 项目说明
```

## 使用方法

1. 数据生成
```bash
python data/get_game_states.py --game_code_folder games --output_file data/data.jsonl
```

2. 运行模拟实验
```bash
python experiments/quest_gpt.py --model gpt-4-0125-preview --data_type full
```

3. 分析结果
```bash
python scripts/results_analysis.py --prefix experiment_name --exp_type full
```

## 许可证

MIT License 