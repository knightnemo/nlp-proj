
<p align="center">
    <img src="assets/icon1.png" width="250"/>
</p>
<h2 align="center"> DreamFactory:
Grounding Language Models to World
Model through Decentralized Generation
and Centralized Verification</h2>


We introduce DreamFactory, a step towards Text-Based World Models that:
1. Simulates world states by leveraging the power of Large Language Models
2. Combines decentralized generation and centralized verification

**For Details, Check Out [[Paper]](https://cloud.tsinghua.edu.cn/f/9611f6eef7114ff7b679/?dl=1)**


[[Paper]](https://cloud.tsinghua.edu.cn/f/9611f6eef7114ff7b679/?dl=1)
[[Code]](https://github.com/knightnemo/nlp-proj.)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The LMDWM project aims to provide more accurate game state predictions and decision support by integrating multiple predictive models and large language models. The project uses PyTorch for model training and leverages LLM for result analysis.

## Features

- **State Prediction**: Utilize the `StatePredictor` class for predicting game states.
- **Ensemble Prediction**: Combine multiple predictions using the `EnsemblePredictor` class.
- **Model Training**: Train a multi-layer perceptron (MLP) model for game states using the `train_wm.py` script.
- **Data Processing**: Process game data through the `GameStateProcessor` class.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/LMDWM.git
   cd LMDWM
   ```

2. Create and activate a virtual environment (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Use venv\Scripts\activate on Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Train the model:
   ```python
   from train_wm import train_model

   model, processor = train_model('path/to/your/datafile.csv', epochs=100, batch_size=32)
   ```

2. Use ensemble prediction:
   ```python
   from state_predictor import EnsemblePredictor
   from llama_interface import LlamaInterface

   predictor = StatePredictor()
   llm = LlamaInterface()
   ensemble_predictor = EnsemblePredictor(predictor, llm)

   state = {...}  # Current game state
   action = "move_forward"
   reward = 1.0
   predictions = ensemble_predictor.predict_with_ensemble(state, action, reward)
   ```

## Dependencies

- `torch` - For deep learning models
- `numpy` - For numerical computations
- `requests` - For HTTP requests (if using LLM API)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Create a new Pull Request

## License

This project is licensed under the [MIT License](LICENSE).