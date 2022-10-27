lr=1e-4
SAVE_DIR=/home1/sxy/transfer-learning-conv-ai/runs/personachat_without_mc_without_type_length_limit_gpt2/$lr
CUDA_VISIBLE_DEVICES=0 python train_without_mc_without_type_length_limit_gpt2.py \
--gradient_accumulation_steps=2 \
--max_input_len 128 \
--max_history=2 --n_epochs=6 --personality_permutations=1 --train_batch_size=32 --valid_batch_size=32 \
--lr=$lr \
--save_dir $SAVE_DIR >train_logs/personachat_without_mc_without_type_length_limit_gpt2_${lr}.log 2>&1