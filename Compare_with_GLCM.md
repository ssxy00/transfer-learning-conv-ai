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

lr=3e-4
总共训练了 10 个 epochs，在 7 epochs ppl 达到最低：
Validation: {'accuracy': 0.7515, 'nll': 3.8322, 46.1655},
此时 accuracy 不是最高，最高在 10 epochs accuracy: 0.7725，但是考虑到我们是要做生成任务，还是选择 ckpt with lowest valid ppl

### DailyDialog
lr=1e-4
总共训练了 51 个 epochs，在 45 epochs ppl 达到最低：
Validation: {'accuracy': 0.6255, 'nll': 3.6395, 'ppl': 38.0739},
此时 accuracy 不是最高，最高在 47 epochs accuracy: 0.6380，但是考虑到我们是要做生成任务，还是选择 ckpt with lowest valid ppl

(ps: 在 dailydialog 上 lr=1e-4 跑过两次实验，发现结果略有差别，说明这一代码结果是不可复现的)

Test:
loss: 3.5975, ppl: 36.5051, bleu: 6.89, meteor: 11.73, nist: 27.42, rougeL: 17.11

lr=8e-5
总共训练了 62 个 epochs，在 51 epochs ppl 达到最低：
Validation: {'accuracy': 0.6351, 'nll': 3.7073, 'ppl': 40.7432},
此时 accuracy 不是最高，最高在 42 epochs accuracy: 0.6404，但是考虑到我们是要做生成任务，还是选择 ckpt with lowest valid ppl

lr=3e-4
总共训练了 41 个 epochs，在 39 epochs ppl 达到最低：
Validation: {'accuracy': 0.6260, 'nll': 3.6683, 'ppl': 39.1864},
此时 accuracy 不是最高，最高在 38 epochs accuracy: 0.6361，但是考虑到我们是要做生成任务，还是选择 ckpt with lowest valid ppl

### MultiWOZ
lr=1e-4
总共训练了 31 个 epochs，在 25 epochs ppl 达到最低：
Validation: {'accuracy': 0.8849, 'nll': 1.7309, 'ppl': 5.6459},
此时 accuracy 不是最高，最高在 16 epochs accuracy: 0.8910，但是考虑到我们是要做生成任务，还是选择 ckpt with lowest valid ppl

Test:
loss: 1.6772, ppl: 5.3507, bleu: 10.03, meteor: 16.81, nist: 47.10, rougeL: 19.48

lr=8e-5
总共训练了 100 个 epochs，在 27 epochs ppl 达到最低：
Validation: {'accuracy': 0.8807, 'nll': 1.7348, 'ppl': 5.6676},
此时 accuracy 不是最高，最高在 19 epochs accuracy: 0.8910，但是考虑到我们是要做生成任务，还是选择 ckpt with lowest valid ppl

lr=3e-4
总共训练了 100 个 epochs，在 21 epochs ppl 达到最低：
Validation: {'accuracy': 0.8846, 'nll': 1.7314, 'ppl': 5.6486},
此时 accuracy 不是最高，最高在 85 epochs accuracy: 0.8952，但是考虑到我们是要做生成任务，还是选择 ckpt with lowest valid ppl

## without mc
去掉 multitask
### PersonaChat without mc
lr=8e-5
总共训练了 5 个 epochs，在 2 epochs ppl 达到最低：
Validation: {'nll': 3.7211, 'ppl': 41.3093}

lr=1e-4
总共训练了 11 个 epochs，在 2 epochs ppl 达到最低：
Validation: {'nll': 3.7163, 'ppl': 41.1120}

Test:
checkpoint_mymodel_34434.pt
loss: 3.7076, ppl: 40.7578, bleu: 8.19, meteor: 11.25, nist: 45.08, rougeL: 15.30


lr=3e-4
总共训练了 3 个 epochs，在 1 epochs ppl 达到最低：
Validation: {'nll': 3.7329, 'ppl': 41.7995}


### Dailydialog without mc
lr=8e-5
总共训练了 15 个 epochs，在 13 epochs ppl 达到最低：
Validation: {'nll': 3.7930, 'ppl': 44.3878}

lr=1e-4
总共训练了 79 个 epochs，在 12 epochs ppl 达到最低：
Validation: {'nll': 3.7680, 'ppl': 43.2915}

lr=3e-4
总共训练了 15 个 epochs，在 7 epochs ppl 达到最低：
Validation: {'nll': 3.6424, 'ppl': 38.1839}

Test:
checkpoint_mymodel_19523.pt
loss: 3.4608, ppl: 31.8431, bleu: 7.11, meteor: 11.76, nist: 30.33, rougeL: 16.44

lr=5e-4
总共训练了 11 个 epochs，在 6 epochs ppl 达到最低：
Validation: {'nll': 3.6480, 'ppl': 38.3970}

### MultiWOZ without mc
lr=1e-4
总共训练了 53 个 epochs，在 10 epochs ppl 达到最低：
Validation: {'nll': 1.7036, 'ppl': 5.4935}

lr=3e-4
总共训练了 13 个 epochs，在 6 epochs ppl 达到最低：
Validation: {'nll': 1.6941, 'ppl': 5.4415}

Test:
checkpoint_mymodel_21870.pt
loss: 1.6401, ppl: 5.1558, bleu: 10.5040, meteor: 17.7016, nist: 51.8052, rougeL: 19.9565

lr=5e-4
总共训练了 10 个 epochs，在 7 epochs ppl 达到最低：
Validation: {'nll': 1.7017, 'ppl': 5.4833}


## without mc without type
进一步去掉 token_type_ids，并且输入 input_ids 也改成和 dialogpt 一样的形式 x1 eos y1 eos ...
对于 personachat，也把 persona permutation 去掉
TODO: 三个数据集的 evaluation 都还没改

### Personachat without mc without type
lr=8e-5
总共训练了 6 个 epochs，在 4 epochs ppl 达到最低：
Validation: {'nll': 3.7296, 'ppl': 41.6604}

lr=1e-4
总共训练了 6 个 epochs，在 4 epochs ppl 达到最低：
Validation: {'nll': 3.6955, 'ppl': 40.2675}

Test:
checkpoint_mymodel_34436.pt
loss: 3.6798, ppl: 39.6384, bleu: 7.90, meteor: 11.03, nist: 43.35, rougeL: 15.35

lr=3e-4
总共训练了 6 个 epochs，在 2 epochs ppl 达到最低：
Validation: {'nll': 3.7023, 'ppl': 40.5405}

### Dailydialog without mc without type
lr=1e-4
总共训练了 15 个 epochs，在 11 epochs ppl 达到最低：
Validation: {'nll': 3.8291, 'ppl': 46.0192}

lr=3e-4
总共训练了 15 个 epochs，在 6 epochs ppl 达到最低：
Validation: {'nll': 3.6550, 'ppl': 38.6661}

lr=5e-4
总共训练了 15 个 epochs，在 6 epochs ppl 达到最低：
Validation: {'nll': 3.6411, 'ppl': 38.1327}

Test:
checkpoint_mymodel_16734.pt
loss: 3.4537, ppl: 31.6169, bleu: 7.17, meteor: 11.25, nist: 29.96, rougeL: 15.70

lr=8e-4
总共训练了 15 个 epochs，在 6 epochs ppl 达到最低：
Validation: {'nll': 3.6951, 'ppl': 40.2498}

### Multiwoz without mc without type
lr=1e-4
总共训练了 12 个 epochs，在 10 epochs ppl 达到最低：
Validation: {'nll': 1.7054, 'ppl': 5.5038}

lr=3e-4
总共训练了 11 个 epochs，在 7 epochs ppl 达到最低：
Validaton: {'nll': 1.6987, 'ppl': 5.4669}

Test:
checkpoint_mymodel_25515.pt
loss: 1.6419, ppl: 5.1650, bleu: 10.25, meteor: 17.32, nist: 49.73, rougeL: 19.70

lr=5e-4
总共训练了 12 个 epochs，在 7 epochs ppl 达到最低：
Validation: {'nll': 1.7062, 'ppl': 5.5081}

## without mc without type length limit (128)
TODO evaluate 还没写

### Personachat without mc without type length limit

lr=1e-4
total: 6, choose 4
Validation: {'nll': 3.6458, 'ppl': 38.3129}

lr=3e-4
total: 6, choose 2
Validation: {'nll': 3.6429, 'ppl': 38.2025}

Test:
checkpoint_mymodel_8610.pt
loss: 3.6377, ppl: 38.0041, bleu: 8.12, meteor: 10.61, nist: 46.31, rougeL: 14.95

lr=5e-4
total: 3, choose 2
Validation: {'nll': 3.6938, 'ppl': 40.1972}


### DailyDialog without mc without type length limit
lr=1e-4
total 15, choose 12
Validation: {'nll': 3.8030, 'ppl': 44.8364}

lr=3e-4
total: 15, choose 6
Validation: {'nll': 3.6076, 'ppl': 36.8782}

lr=5e-4
total 15, choose 6
Validation: {'nll': 3.5975, 'ppl': 36.5081}

Test:
checkpoint_mymodel_4188.pt
loss: 3.3765, ppl: 29.2686, bleu: 7.10, meteor: 11.48, nist: 30.61, rougeL: 16.44

lr=8e-4
total 9, choose 7
Validation: {'nll': 3.6791, 'ppl': 39.6108}


### Multiwoz without mc without type length limit
lr=1e-4
total: 15, choose 9
Validation: {'nll': 1.6577, 'ppl': 5.2473}

lr=3e-4
total: 15, choose 5
Validation: {'nll': 1.6560, 'ppl': 5.2383}

Test:
checkpoint_mymodel_9115.pt
loss: 1.6073, ppl: 4.9893, bleu: 11.33, meteor: 18.38, nist: 57.08, rougeL: 20.87


lr=5e-4
total: 15, choose 6
Validation: {'nll': 1.6618, 5.2689}






