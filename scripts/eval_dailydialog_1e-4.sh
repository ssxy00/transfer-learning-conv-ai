CUDA_VISBLE_DEVICES=2 python evaluate_without_persona.py \
--dataset_path /home1/sxy/transfer-learning-conv-ai/datasets/dailydialog/dailydialog_test.json \
--dataset_cache /home1/sxy/transfer-learning-conv-ai/datasets/dailydialog/dailydialog_test.cache \
--model_checkpoint /home1/sxy/transfer-learning-conv-ai/runs/dailydialog/1e-4/Oct02_15-04-01_IW4202/ \
--optimal_step 501975 \
--save_result_path /home1/sxy/transfer-learning-conv-ai/results/dailydialog_1e-4_501975.txt \
--max_history 2 \
--max_length 20 --min_length 1