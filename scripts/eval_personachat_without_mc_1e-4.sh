CUDA_VISBLE_DEVICES=0 python evaluate.py \
--dataset_path /home1/sxy/transfer-learning-conv-ai/datasets/personachat/personachat_test.json \
--dataset_cache /home1/sxy/transfer-learning-conv-ai/datasets/personachat/personachat_test.cache \
--model_checkpoint /home1/sxy/transfer-learning-conv-ai/runs/personachat_without_mc/1e-4/Oct24_22-33-30_IW4202/ \
--optimal_step 34434 \
--save_result_path /home1/sxy/transfer-learning-conv-ai/results/personachat_without_mc_1e-4_34434.txt \
--max_history 2 \
--max_length 20 --min_length 1