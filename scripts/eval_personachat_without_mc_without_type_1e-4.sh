CUDA_VISBLE_DEVICES=0 python evaluate_without_type.py \
--dataset_path /home1/sxy/transfer-learning-conv-ai/datasets/personachat/personachat_test.json \
--dataset_cache /home1/sxy/transfer-learning-conv-ai/datasets/personachat/personachat_test.cache \
--model_checkpoint /home1/sxy/transfer-learning-conv-ai/runs/personachat_without_mc_without_type/1e-4/Oct26_03-52-52_IW4202/ \
--optimal_step 34436 \
--save_result_path /home1/sxy/transfer-learning-conv-ai/results/personachat_without_mc_without_type_1e-4_34436.txt \
--max_history 2 \
--max_length 20 --min_length 1