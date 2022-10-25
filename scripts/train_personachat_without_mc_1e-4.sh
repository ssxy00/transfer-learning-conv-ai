lr=1e-4
SAVE_DIR=/home1/sxy/transfer-learning-conv-ai/runs/personachat_without_mc/$lr
CUDA_VISIBLE_DEVICES=0 python train_without_mc.py \
--gradient_accumulation_steps=4 \
--max_history=2 --n_epochs=100 --personality_permutations=2 --train_batch_size=16 --valid_batch_size=16 \
--lr=$lr \
--save_dir $SAVE_DIR
