CUDA_VISBLE_DEVICES=2 python evaluate_without_persona.py \
--dataset_path /home1/sxy/transfer-learning-conv-ai/datasets/multiwoz/multiwoz_test.json \
--dataset_cache /home1/sxy/transfer-learning-conv-ai/datasets/multiwoz/multiwoz_test.cache \
--model_checkpoint /home1/sxy/transfer-learning-conv-ai/runs/multiwoz/1e-4/Oct03_19-43-10_IW4202/ \
--optimal_step 364500 \
--save_result_path /home1/sxy/transfer-learning-conv-ai/results/multiwoz_1e-4_364500.txt \
--max_history 2 \
--max_length 20 --min_length 1