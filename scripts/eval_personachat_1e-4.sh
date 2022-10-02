CUDA_VISBLE_DEVICES=1 python evaluate.py \
--dataset_path /home1/sxy/transfer-learning-conv-ai/datasets/personachat/personachat_test.json \
--dataset_cache /home1/sxy/transfer-learning-conv-ai/datasets/personachat/personachat_test.cache \
--model_checkpoint /home1/sxy/transfer-learning-conv-ai/runs/personachat/1e-4/Sep30_18-48-19_IW4202/ \
--optimal_step 482055 \
--save_result_path /home1/sxy/transfer-learning-conv-ai/results/personachat_1e-4_482055.txt \
--max_history 2 \
--max_length 20 --min_length 1