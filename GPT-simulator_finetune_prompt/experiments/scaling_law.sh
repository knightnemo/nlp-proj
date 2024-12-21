export HF_ENDPOINT=https://hf-mirror.com

# TinyLlama (1.1B)
CUDA_VISIBLE_DEVICES=2 nohup python experiments/quest_llama.py --model_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --model_type local --device cuda --output_prefix tinyllama_test > tinyllama.log 2>&1 &
# Phi-1.5 (1.3B)
CUDA_VISIBLE_DEVICES=3 nohup python experiments/quest_llama.py --model_path "microsoft/phi-1_5" --model_type local --device cuda --output_prefix phi15_test > phi15.log 2>&1 &
# Phi-2 (2.7B)
CUDA_VISIBLE_DEVICES=4 nohup python experiments/quest_llama.py --model_path "microsoft/phi-2" --model_type local --device cuda --output_prefix phi2_test > phi2.log 2>&1 &
# StableLM (3B)
CUDA_VISIBLE_DEVICES=5 nohup python experiments/quest_llama.py --model_path "stabilityai/stablelm-3b-4e1t" --model_type local --device cuda --output_prefix stablelm_test > stablelm.log 2>&1 &
# SOLAR (6.7B)
CUDA_VISIBLE_DEVICES=2 nohup python experiments/quest_llama.py --model_path "upstage/SOLAR-0-70b-16bit" --model_type local --device cuda --output_prefix solar_test > solar.log 2>&1 &
# Llama-2 (7B)
CUDA_VISIBLE_DEVICES=2 nohup python experiments/quest_llama.py --model_path "meta-llama/Llama-2-7b-chat-hf" --model_type local --device cuda --output_prefix llama2_test > llama2.log 2>&1 &
# Mistral (7B)
CUDA_VISIBLE_DEVICES=2 nohup python experiments/quest_llama.py --model_path "mistralai/Mistral-7B-v0.1" --model_type local --device cuda --output_prefix mistral_test > mistral.log 2>&1 &
# Llama-3 (8B)
CUDA_VISIBLE_DEVICES=2 nohup python experiments/quest_llama.py --model_path "meta-llama/Llama-3-8b-chat-hf" --model_type local --device cuda --output_prefix llama3_test > llama3.log 2>&1 &
# Llama-3 (10B) ≈ 20GB -> 10GB with 8-bit
CUDA_VISIBLE_DEVICES=2 nohup python experiments/quest_llama.py --model_path "meta-llama/Llama-3-10b-chat-hf" --model_type local --device cuda --output_prefix llama3_10b_test --load_in_8bit True > llama3_10b.log 2>&1 &
# Llama-3 (14B) ≈ 28GB -> 14GB with 8-bit
CUDA_VISIBLE_DEVICES=2 nohup python experiments/quest_llama.py --model_path "meta-llama/Llama-3-14b-chat-hf" --model_type local --device cuda --output_prefix llama3_14b_test --load_in_8bit True > llama3_14b.log 2>&1 &
# Llama-3 (34B) ≈ 68GB -> 17GB with 4-bit
CUDA_VISIBLE_DEVICES=2 nohup python experiments/quest_llama.py --model_path "meta-llama/Llama-3-34b-chat-hf" --model_type local --device cuda --output_prefix llama3_34b_test --load_in_4bit True > llama3_34b.log 2>&1 &
# Mixtral-8x7B (47B) ≈ 94GB -> 23.5GB with 4-bit
CUDA_VISIBLE_DEVICES=2 nohup python experiments/quest_llama.py --model_path "mistralai/Mixtral-8x7B-v0.1" --model_type local --device cuda --output_prefix mixtral_test --load_in_4bit True > mixtral.log 2>&1 &