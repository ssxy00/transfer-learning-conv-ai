lr=5e-4
SAVE_DIR=/home1/sxy/transfer-learning-conv-ai/runs/dailydialog_without_mc_without_type_length_limit/$lr
CUDA_VISIBLE_DEVICES=1 python train_without_persona_without_mc_without_type_length_limit.py \
--gradient_accumulation_steps=2 \
--max_history=2 --n_epochs=15 --train_batch_size=32 --valid_batch_size=32 \
--max_input_len 128 \
--lr=$lr \
--save_dir $SAVE_DIR \
--dataset_path /home1/sxy/transfer-learning-conv-ai/datasets/dailydialog/dailydialog.json \
--dataset_cache /home1/sxy/transfer-learning-conv-ai/datasets/dailydialog/dailydialog.cache