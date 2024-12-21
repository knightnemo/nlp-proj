#!/bin/bash

# Check HuggingFace login status
if ! huggingface-cli whoami &>/dev/null; then
    echo "Error: Not logged in to HuggingFace"
    echo "Please run: huggingface-cli login"
    exit 1
fi

# Create models directory
mkdir -p ./models

# Model configuration
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
SAVE_MODEL_LOCALLY="--save_model_locally"

# Game configuration
GAME_CONFIG="./experiments/one_game.json"  # 使用游戏配置文件
OUTPUT_DIR="./results/llama_lora"
TRAIN_DATA="./data/train.jsonl"
RULE_FOLDER="./rules/human_written_rules"

# Check if required files exist
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data not found at $TRAIN_DATA"
    exit 1
fi

if [ ! -d "$RULE_FOLDER" ]; then
    echo "Error: Rules folder not found at $RULE_FOLDER"
    exit 1
fi

if [ ! -f "$GAME_CONFIG" ]; then
    echo "Error: Game configuration file not found at $GAME_CONFIG"
    exit 1
fi

# Print configuration
echo "=== Training Configuration ==="
echo "Model: $BASE_MODEL_PATH"
echo "Game config: $GAME_CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Training data: $TRAIN_DATA"
echo "Rules folder: $RULE_FOLDER"
echo "==========================="

# Create output directory
mkdir -p $OUTPUT_DIR

# Training hyperparameters
NUM_EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=16
LEARNING_RATE=2e-5
MAX_LENGTH=2048

# LoRA parameters
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1

# Run training
echo "Starting training..."
python ./experiments/train_llama_lora.py \
    --base_model_path $BASE_MODEL_PATH \
    $SAVE_MODEL_LOCALLY \
    --train_data $TRAIN_DATA \
    --rule_folder $RULE_FOLDER \
    --output_dir $OUTPUT_DIR \
    --game_file_names $GAME_CONFIG \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --data_type "action" \
    --max_examples_per_game 500

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed"
    exit 1
fi