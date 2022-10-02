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
+ test (personachat 上的 evaluate 已经写好了)
+ dailydialog (已经造了数据，训练的部分已经修改好了，正在重新训练 1e-4 因为之前跑的 ckpt 没有存下来，测试部分已经改好还没 debug)
+ multiwoz (已经造了数据，训练的部分已经修改好了，准备训练 1e-4)

## Exp
### PersonaChat
lr=1e-4
总共训练了 12 个 epochs，在 7 epochs ppl 达到最低：
Validation: {'accuracy': 0.7769, 'nll': 3.7763, 'ppl': 43.6525},
此时 accuracy 不是最高，最高在 8 epochs accuracy: 0.7988，但是考虑到我们是要做生成任务，还是选择 ckpt with lowest valid ppl

Test:
loss: 3.7859, ppl: 44.0732, bleu: 8.11, meteor: 11.10, nist: 44.38, rougeL: 15.19


### DailyDialog
lr=1e-4
总共训练了  个 epochs，在  epochs ppl 达到最低：
Validation: {'accuracy': , 'nll': , 'ppl': },
此时 accuracy 不是最高，最高在  epochs accuracy: ，但是考虑到我们是要做生成任务，还是选择 ckpt with lowest valid ppl

### MultiWOZ
lr=1e-4
