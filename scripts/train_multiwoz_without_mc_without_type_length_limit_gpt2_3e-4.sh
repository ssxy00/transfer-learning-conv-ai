lr=3e-4
SAVE_DIR=/home1/sxy/transfer-learning-conv-ai/runs/multiwoz_without_mc_without_type_length_limit_gpt2/$lr
CUDA_VISIBLE_DEVICES=2 python train_without_persona_without_mc_without_type_length_limit_gpt2.py \
--gradient_accumulation_steps=2 \
--max_history=2 --n_epochs=15 --train_batch_size=32 --valid_batch_size=32 \
--max_input_len 128 \
--lr=$lr \
--save_dir $SAVE_DIR \
--dataset_path /home1/sxy/transfer-learning-conv-ai/datasets/multiwoz/multiwoz.json \
--dataset_cache /home1/sxy/transfer-learning-conv-ai/datasets/multiwoz/multiwoz.cache >train_logs/multiwoz_without_mc_without_type_length_limit_gpt2_${lr}.log 2>&1