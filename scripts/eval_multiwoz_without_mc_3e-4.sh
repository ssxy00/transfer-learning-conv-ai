CUDA_VISBLE_DEVICES=0 python evaluate_without_persona.py \
--dataset_path /home1/sxy/transfer-learning-conv-ai/datasets/multiwoz/multiwoz_test.json \
--dataset_cache /home1/sxy/transfer-learning-conv-ai/datasets/multiwoz/multiwoz_test.cache \
--model_checkpoint /home1/sxy/transfer-learning-conv-ai/runs/multiwoz_without_mc/3e-4/Oct25_08-56-03_IW4202/ \
--optimal_step 21870 \
--save_result_path /home1/sxy/transfer-learning-conv-ai/results/multiwoz_without_mc_3e-4_21870.txt \
--max_history 2 \
--max_length 20 --min_length 1