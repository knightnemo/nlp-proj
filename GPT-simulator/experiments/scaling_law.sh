export HF_ENDPOINT=https://hf-mirror.com
# TinyLlama (1.1B) - A small but efficient model trained on chat data
CUDA_VISIBLE_DEVICES=1 nohup python experiments/quest_llama.py --model_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --model_type local --output_prefix tinyllama_hwr_diff_full --device cuda --rule_folder ./rules/human_written_rules --output_folder results --data_type full --partial > tinyllama.log 2>&1 &

# Phi-1.5 (1.3B) - Microsoft's small language model
CUDA_VISIBLE_DEVICES=3 nohup python experiments/quest_llama.py --model_path "microsoft/phi-1_5" --model_type local --output_prefix phi15_hwr_diff_full --device cuda --rule_folder ./rules/human_written_rules --output_folder results --data_type full --partial > phi15.log 2>&1 &

# Phi-2 (2.7B) - Microsoft's improved version of Phi-1.5
CUDA_VISIBLE_DEVICES=4 nohup python experiments/quest_llama.py --model_path "microsoft/phi-2" --model_type local --output_prefix phi2_hwr_diff_full --device cuda --rule_folder ./rules/human_written_rules --output_folder results --data_type full --partial > phi2.log 2>&1 &

# StableLM-3B - Stability AI's 3B parameter model
CUDA_VISIBLE_DEVICES=5 nohup python experiments/quest_llama.py --model_path "stabilityai/stablelm-3b-4e1t" --model_type local --output_prefix stablelm_hwr_diff_full --device cuda --rule_folder ./rules/human_written_rules --output_folder results --data_type full --partial > stablelm.log 2>&1 &

# SOLAR-70B - Upstage's 70B parameter model
CUDA_VISIBLE_DEVICES=2 nohup python experiments/quest_llama.py --model_path "upstage/SOLAR-0-70b-16bit" --model_type local --output_prefix solar_hwr_diff_full --device cuda --rule_folder ./rules/human_written_rules --output_folder results --data_type full --partial > solar.log 2>&1 &

# Llama-2-7B - Meta's 7B parameter chat model
CUDA_VISIBLE_DEVICES=2 nohup python experiments/quest_llama.py --model_path "meta-llama/Llama-2-7b-chat-hf" --model_type local --output_prefix llama2_hwr_diff_full --device cuda --rule_folder ./rules/human_written_rules --output_folder results --data_type full --partial > llama2.log 2>&1 &

# Mistral-7B - Mistral AI's 7B parameter base model
CUDA_VISIBLE_DEVICES=2 nohup python experiments/quest_llama.py --model_path "mistralai/Mistral-7B-v0.1" --model_type local --output_prefix mistral_hwr_diff_full --device cuda --rule_folder ./rules/human_written_rules --output_folder results --data_type full --partial > mistral.log 2>&1 &

# Llama-3-8B - Meta's 8B parameter chat model from Llama 3 family
CUDA_VISIBLE_DEVICES=2 nohup python experiments/quest_llama.py --model_path "meta-llama/Llama-3-8b-chat-hf" --model_type local --output_prefix llama3_hwr_diff_full --device cuda --rule_folder ./rules/human_written_rules --output_folder results --data_type full --partial > llama3.log 2>&1 &

# Llama-3-10B - Meta's 10B parameter chat model from Llama 3 family
CUDA_VISIBLE_DEVICES=2 nohup python experiments/quest_llama.py --model_path "meta-llama/Llama-3-10b-chat-hf" --model_type local --output_prefix llama3_10b_hwr_diff_full --device cuda --rule_folder ./rules/human_written_rules --output_folder results --data_type full --partial --load_in_8bit True > llama3_10b.log 2>&1 &

# Llama-3-14B - Meta's 14B parameter chat model from Llama 3 family
CUDA_VISIBLE_DEVICES=2 nohup python experiments/quest_llama.py --model_path "meta-llama/Llama-3-14b-chat-hf" --model_type local --output_prefix llama3_14b_hwr_diff_full --device cuda --rule_folder ./rules/human_written_rules --output_folder results --data_type full --partial --load_in_8bit True > llama3_14b.log 2>&1 &

# Llama-3-34B - Meta's 34B parameter chat model from Llama 3 family
CUDA_VISIBLE_DEVICES=2 nohup python experiments/quest_llama.py --model_path "meta-llama/Llama-3-34b-chat-hf" --model_type local --output_prefix llama3_34b_hwr_diff_full --device cuda --rule_folder ./rules/human_written_rules --output_folder results --data_type full --partial --load_in_4bit True > llama3_34b.log 2>&1 &

# Mixtral-8x7B - Mistral AI's mixture of experts model with 8 experts of 7B parameters each
CUDA_VISIBLE_DEVICES=2 nohup python experiments/quest_llama.py --model_path "mistralai/Mixtral-8x7B-v0.1" --model_type local --output_prefix mixtral_hwr_diff_full --device cuda --rule_folder ./rules/human_written_rules --output_folder results --data_type full --partial --load_in_4bit True > mixtral.log 2>&1 &

