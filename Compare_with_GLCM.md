# Compare with GLCM
对代码的修改：
- model 不载入预训练参数，改变模型规模
- tokenizer 改变 special tokens 从而与 GLCM 统一
- training 不设置 lr schedule
- build input segment 原代码的写法只兼容对话从 speaker2 开始，因此修改
- personachat 因为看不到官方数据处理的脚本只能看到处理完的 json，因此只好模仿 `example_entry.py` 来造数据： personachat 数据预处理时，split 时可以从 speaker1 开始也可以从 speaker2 开始，但是保证每个对话至少有 7 条 history utterances，从而与 GLCM 数据一致

对 group[nlp] 环境的修改：
```
python -m pip install spacy
python -m spacy download en
python -m pip install pytorch-ignite
```

TODO:
+ test (已经完成 lr=1e-4 的实验，之后可以再补其他 lr)
+ dailydialog (已经完成 lr=1e-4 的实验，之后可以再补其他 lr)
+ multiwoz (正在重新训练 1e-4)

## Exp
### PersonaChat
lr=1e-4
总共训练了 12 个 epochs，在 7 epochs ppl 达到最低：
Validation: {'accuracy': 0.7769, 'nll': 3.7763, 'ppl': 43.6525},
此时 accuracy 不是最高，最高在 8 epochs accuracy: 0.7988，但是考虑到我们是要做生成任务，还是选择 ckpt with lowest valid ppl

Test:
loss: 3.7859, ppl: 44.0732, bleu: 8.11, meteor: 11.10, nist: 44.38, rougeL: 15.19

lr=8e-5
总共训练了 11 个 epochs，在 8 epochs ppl 达到最低：
Validation: {'accuracy': 0.7167, 'nll': 3.7837, 'ppl': 43.9765},
此时 accuracy 不是最高，最高在 11 epochs accuracy: 0.7716，但是考虑到我们是要做生成任务，还是选择 ckpt with lowest valid ppl


### DailyDialog
lr=1e-4
总共训练了 51 个 epochs，在 45 epochs ppl 达到最低：
Validation: {'accuracy': 0.6255, 'nll': 3.6395, 'ppl': 38.0739},
此时 accuracy 不是最高，最高在 47 epochs accuracy: 0.6380，但是考虑到我们是要做生成任务，还是选择 ckpt with lowest valid ppl

(ps: 在 dailydialog 上 lr=1e-4 跑过两次实验，发现结果略有差别，说明这一代码结果是不可复现的)

Test:
loss: 3.5975, ppl: 36.5051, bleu: 6.89, meteor: 11.73, nist: 27.42, rougeL: 17.11


### MultiWOZ
lr=1e-4
总共训练了 31 个 epochs，在 25 epochs ppl 达到最低：
Validation: {'accuracy': 0.8849, 'nll': 1.7309, 'ppl': 5.6459},
此时 accuracy 不是最高，最高在 16 epochs accuracy: 0.8910，但是考虑到我们是要做生成任务，还是选择 ckpt with lowest valid ppl

Test:
loss: 1.6772, ppl: 5.3507, bleu: 10.03, meteor: 16.81, nist: 47.10, rougeL: 19.48