CUDA_VISBLE_DEVICES=0 python evaluate_without_persona_without_type_length_limit.py \
--dataset_path /home1/sxy/transfer-learning-conv-ai/datasets/multiwoz/multiwoz_test.json \
--dataset_cache /home1/sxy/transfer-learning-conv-ai/datasets/multiwoz/multiwoz_test.cache \
--model_checkpoint /home1/sxy/transfer-learning-conv-ai/runs/multiwoz_without_mc_without_type_length_limit/3e-4/Oct26_15-23-49_IW4202/ \
--optimal_step 9115 \
--save_result_path /home1/sxy/transfer-learning-conv-ai/results/multiwoz_without_mc_without_type_length_limit_3e-4_9115.txt \
--max_history 2 \
--max_input_len 128 \
--max_length 20 --min_length 1