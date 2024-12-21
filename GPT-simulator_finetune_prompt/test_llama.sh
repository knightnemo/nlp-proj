#!/bin/bash

# Check HuggingFace login status
if ! huggingface-cli whoami &>/dev/null; then
    echo "Error: Not logged in to HuggingFace"
    echo "Please run: huggingface-cli login"
    exit 1
fi

# Model configuration
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
LORA_WEIGHTS_PATH="./results/llama_lora"
GAME_CONFIG="./experiments/one_game.json"

# Data paths
STATE_DATA_FOLDER="./data"
TEST_DATA="test.jsonl"
RULE_FOLDER="./rules/human_written_rules"
OUTPUT_FOLDER="./results/evaluation"

# Create output directory
mkdir -p $OUTPUT_FOLDER

# Model configurations to test
declare -a LORA_CONFIGS=("with_lora" "no_lora")
declare -a DATA_TYPES=("full")
declare -a RULE_CONFIGS=("")

# Run tests for each configuration
for LORA_CONFIG in "${LORA_CONFIGS[@]}"
do
    echo "Testing ${LORA_CONFIG} configuration"
    
    # Set LORA argument based on configuration
    LORA_ARG=""
    MODEL_PREFIX="llama3_base"
    if [ "$LORA_CONFIG" = "with_lora" ]; then
        LORA_ARG="--lora_weights_path $LORA_WEIGHTS_PATH"
        MODEL_PREFIX="llama3_lora"
    fi

    for DATA_TYPE in "${DATA_TYPES[@]}"
    do
        for RULE_CONFIG in "${RULE_CONFIGS[@]}"
        do
            OUTPUT_PREFIX="${MODEL_PREFIX}_${DATA_TYPE}${RULE_CONFIG:+_no_rule}"
            
            echo "Running test for ${DATA_TYPE} with ${LORA_CONFIG}"
            
            python ./experiments/quest_llama_lora.py \
                --base_model_path $BASE_MODEL_PATH \
                $LORA_ARG \
                --state_data_folder $STATE_DATA_FOLDER \
                --test_data $TEST_DATA \
                --rule_folder $RULE_FOLDER \
                --game_file_names $GAME_CONFIG \
                --output_folder $OUTPUT_FOLDER \
                --output_prefix $OUTPUT_PREFIX \
                --data_type $DATA_TYPE \
                $RULE_CONFIG
                
            echo "Completed test for ${DATA_TYPE} with ${LORA_CONFIG}"
            echo "----------------------------------------"
        done
    done
done

echo "All tests completed!"