lr=1e-4
SAVE_DIR=/home1/sxy/transfer-learning-conv-ai/runs/dailydialog/$lr
CUDA_VISIBLE_DEVICES=2 python train_without_persona.py \
--gradient_accumulation_steps=32 \
--lm_coef=2.0 \
--max_history=2 --n_epochs=100 --num_candidates=4 --train_batch_size=2 --valid_batch_size=2 \
--lr=$lr \
--save_dir $SAVE_DIR \
--dataset_path /home1/sxy/transfer-learning-conv-ai/datasets/dailydialog/dailydialog.json \
--dataset_cache /home1/sxy/transfer-learning-conv-ai/datasets/dailydialog/dailydialog.cache
