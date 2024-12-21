#!/usr/bin/env bash

# Default values
MODEL="gpt-4o"
OUTPUT_FOLDER="results"
STATE_DATA_FOLDER="data"
TEST_DATA="test.jsonl"
API_KEY="sk-LNlERgnRs3JFZaVPOXU9umJXecxAeD8AuQdOnOa0tCTizq1L"
BASE_URL="https://api.chatanywhere.tech/v1"
GAME_FILE="experiments/one_game.json"

# Function to display usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --model MODEL             GPT model to use (default: gpt-4-0125-preview)"
    echo "  --output_folder PATH      Output directory for results (default: results)"
    echo "  --state_data_folder PATH  Directory containing state data (default: data)"
    echo "  --test_data FILE          Test data file (default: test.jsonl)"
    echo "  --game_file FILE          Game configuration file (default: experiments/games.json)"
    echo "  --api_key KEY             OpenAI API key (required)"
    echo "  --base_url URL            API base URL (default: https://api.chatanywhere.tech/v1)"
    echo "  --data_type TYPE          Data type: action, tick, score, full (default: action)"
    echo "  --partial                 Enable partial state output"
    echo "  --no_rule                 Run without rules"
    exit 1
}

# Function to check if required directories exist
check_directories() {
    if [ ! -d "$STATE_DATA_FOLDER" ]; then
        echo "Error: State data folder '$STATE_DATA_FOLDER' does not exist"
        exit 1
    fi
    
    if [ ! -f "$STATE_DATA_FOLDER/$TEST_DATA" ]; then
        echo "Error: Test data file '$STATE_DATA_FOLDER/$TEST_DATA' does not exist"
        exit 1
    fi

    if [ ! -f "$GAME_FILE" ]; then
        echo "Error: Game file '$GAME_FILE' does not exist"
        exit 1
    fi
}

# Parse command line arguments
PARTIAL=""
NO_RULE=""
DATA_TYPE="full"

while [ $# -gt 0 ]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output_folder)
            OUTPUT_FOLDER="$2"
            shift 2
            ;;
        --state_data_folder)
            STATE_DATA_FOLDER="$2"
            shift 2
            ;;
        --test_data)
            TEST_DATA="$2"
            shift 2
            ;;
        --game_file)
            GAME_FILE="$2"
            shift 2
            ;;
        --api_key)
            API_KEY="$2"
            shift 2
            ;;
        --base_url)
            BASE_URL="$2"
            shift 2
            ;;
        --data_type)
            DATA_TYPE="$2"
            shift 2
            ;;
        --partial)
            PARTIAL="--partial"
            shift
            ;;
        --no_rule)
            NO_RULE="--no_rule"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown parameter '$1'"
            usage
            ;;
    esac
done

# Check if API key is provided
if [ -z "$API_KEY" ]; then
    echo "Error: API key is required"
    usage
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Check required directories and files
check_directories

# Get current date for output prefix
DATE=$(date +%Y%m%d)

# Define modes to run
modes="simple zero_shot_cot cot_sc few_shot baseline"

# Run experiments for each mode
for mode in $modes; do
    echo "Running experiment with mode: $mode"
    
    OUTPUT_PREFIX="${MODEL}_${DATE}_${mode}"
    
    if python experiments/quest_gpt_prompts.py \
        --mode "$mode" \
        --model "$MODEL" \
        --output_folder "$OUTPUT_FOLDER" \
        --output_prefix "$OUTPUT_PREFIX" \
        --state_data_folder "$STATE_DATA_FOLDER" \
        --test_data "$TEST_DATA" \
        --game_file_names "$GAME_FILE"\
        --api_key "$API_KEY" \
        --base_url "$BASE_URL" \
        --data_type "$DATA_TYPE" \
        $PARTIAL \
        $NO_RULE; then
        echo "✓ Successfully completed experiment with mode: $mode"
    else
        echo "✗ Error running experiment with mode: $mode"
        exit 1
    fi
done

echo "🎉 All experiments completed successfully!" 