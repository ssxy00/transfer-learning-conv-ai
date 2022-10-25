CUDA_VISBLE_DEVICES=0 python evaluate_without_persona.py \
--dataset_path /home1/sxy/transfer-learning-conv-ai/datasets/dailydialog/dailydialog_test.json \
--dataset_cache /home1/sxy/transfer-learning-conv-ai/datasets/dailydialog/dailydialog_test.cache \
--model_checkpoint /home1/sxy/transfer-learning-conv-ai/runs/dailydialog_without_mc/3e-4/Oct25_11-37-42_IW4202/ \
--optimal_step 19523 \
--save_result_path /home1/sxy/transfer-learning-conv-ai/results/dailydialog_without_mc_3e-4_19523.txt \
--max_history 2 \
--max_length 20 --min_length 1