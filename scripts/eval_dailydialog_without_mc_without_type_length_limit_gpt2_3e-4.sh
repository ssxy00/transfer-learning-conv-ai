CUDA_VISBLE_DEVICES=0 python evaluate_without_persona_without_type_length_limit_gpt2.py \
--dataset_path /home1/sxy/transfer-learning-conv-ai/datasets/dailydialog/dailydialog_test.json \
--dataset_cache /home1/sxy/transfer-learning-conv-ai/datasets/dailydialog/dailydialog_test.cache \
--model_checkpoint /home1/sxy/transfer-learning-conv-ai/runs/dailydialog_without_mc_without_type_length_limit_gpt2/3e-4/Oct27_00-33-11_IW4202/ \
--optimal_step 4188 \
--save_result_path /home1/sxy/transfer-learning-conv-ai/results/dailydialog_without_mc_without_type_length_limit_gpt2_3e-4_4188.txt \
--max_history 2 \
--max_length 20 --min_length 1