lr=1e-4
SAVE_DIR=/home1/sxy/transfer-learning-conv-ai/runs/personachat_without_mc_without_type/$lr
CUDA_VISIBLE_DEVICES=1 python train_without_mc_without_type.py \
--gradient_accumulation_steps=4 \
--max_history=2 --n_epochs=6 --personality_permutations=1 --train_batch_size=16 --valid_batch_size=16 \
--lr=$lr \
--save_dir $SAVE_DIR >train_logs/personachat_without_mc_without_type_${lr}.log 2>&1