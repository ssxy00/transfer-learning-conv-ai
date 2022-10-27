CUDA_VISBLE_DEVICES=0 python evaluate_without_persona_without_type_length_limit_gpt2.py \
--dataset_path /home1/sxy/transfer-learning-conv-ai/datasets/multiwoz/multiwoz_test.json \
--dataset_cache /home1/sxy/transfer-learning-conv-ai/datasets/multiwoz/multiwoz_test.cache \
--model_checkpoint /home1/sxy/transfer-learning-conv-ai/runs/multiwoz_without_mc_without_type_length_limit_gpt2/3e-4/Oct27_01-52-48_IW4202/ \
--optimal_step 10938 \
--save_result_path /home1/sxy/transfer-learning-conv-ai/results/multiwoz_without_mc_without_type_length_limit_gpt2_3e-4_10938.txt \
--max_history 2 \
--max_input_len 128 \
--max_length 20 --min_length 1