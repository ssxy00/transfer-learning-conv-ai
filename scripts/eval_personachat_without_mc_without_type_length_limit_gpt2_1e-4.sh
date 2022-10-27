CUDA_VISBLE_DEVICES=0 python evaluate_without_type_length_limit_gpt2.py \
--dataset_path /home1/sxy/transfer-learning-conv-ai/datasets/personachat/personachat_test.json \
--dataset_cache /home1/sxy/transfer-learning-conv-ai/datasets/personachat/personachat_test.cache \
--model_checkpoint /home1/sxy/transfer-learning-conv-ai/runs/personachat_without_mc_without_type_length_limit_gpt2/1e-4/Oct27_00-32-45_IW4202/ \
--optimal_step 17220 \
--save_result_path /home1/sxy/transfer-learning-conv-ai/results/personachat_without_mc_without_type_length_limit_gpt2_1e-4_17220.txt \
--max_history 2 \
--max_length 20 --min_length 1