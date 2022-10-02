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
+ test (personachat 上的 evaluate 已经写好了，dailydialog 和 multiwoz 需要做一些修改)
+ dailydialog (已经造了数据，训练的部分已经修改好了，正在训练 1e-4)
+ multiwoz (已经造了数据，训练的部分已经修改好了，正在训练 1e-4)

## Exp
### PersonaChat


### DailyDialog
lr=8e-5
总共训练了 66 个 epoch，在 48 epochs ppl 达到最低：
Validation: {'accuracy': 0.6217, 'nll': 3.6939, 'ppl': 40.2011},
此时 accuracy 不是最高，最高在 63 epochs accuracy: 0.6313，但是考虑到我们是要做生成任务，还是选择 ckpt with lowest valid ppl
lr=1e-4
总共训练了 56 个 epochs，在 45 epochs ppl 达到最低：
Validation: {'accuracy': 0.6246, 'nll': 3.6900, 'ppl': 40.0451},
此时 accuracy 不是最高，最高在 54 epochs accuracy: 0.6337，但是考虑到我们是要做生成任务，还是选择 ckpt with lowest valid ppl

### MultiWOZ
lr=1e-4
总共训练了 37 个 epochs，在 25 epochs ppl 达到最低：
Validation: {'accuracy': 0.8918, 'nll': 1.7458, 'ppl': 5.7303},
此时 accuracy 不是最高，最高在 18 epochs accuracy: 0.8930，但是考虑到我们是要做生成任务，还是选择 ckpt with lowest valid ppl