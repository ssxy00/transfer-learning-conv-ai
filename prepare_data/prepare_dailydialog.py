# -*- coding: utf-8 -*-
# @Time        : 2022/9/27 16:02
# @Author      : ssxy00
# @File        : prepare_dailydialog.py
# @Description :


"""
处理数据时要保证 history 中的 utterance 个数不少于 7，即除了 last utterance 至少还有三轮对话历史
在切数据时，不限制对话起始于 x or y
"""

import os
import logging
import random
import json
from tqdm import tqdm


logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def parse_data(data_path):
    """
    :param data_path:
    :return: List[{'dialog': List[str]}]
    """
    with open(data_path, 'r', encoding='utf-8') as file:
        data = []
        for line in tqdm(file.readlines()):
            line = line.strip()
            if len(line) == 0:
                continue
            dialog = [seq.strip() for seq in line.split('__eou__')[:-1]]
            assert len(dialog) > 0
            # 如果 dialog 中 utterance 个数不足 8，则无法满足 context 中至少有 7 条
            if len(dialog) > 7:
                data.append({'dialog': dialog})
        return data

def split_data(data, need_candidates=True):
    all_utterances = []
    for dialog in data:
        all_utterances += dialog["dialog"]
    logger.info("splitting multi-turn dialogs into personality + utterances")
    processed_data = []
    for dialog in tqdm(data):
        processed_dialog = {"utterances": []}
        for utter_idx in range(7, len(dialog['dialog'])):
            history = dialog['dialog'][: utter_idx]
            tgt = dialog['dialog'][utter_idx]
            if need_candidates:
                candidates = random.sample(all_utterances, 19) + [tgt]
            else:
                candidates = [tgt]
            processed_dialog["utterances"].append({"candidates": candidates, "history": history})
        processed_data.append(processed_dialog)
    return processed_data

def process_dailydialog(save_dir, data_dir):
    logger.info("Now parsing training data:")
    train_multi_turn_data = parse_data(os.path.join(data_dir, "train", "dialogues_train.txt"))
    logger.info("Now parsing validation data:")
    valid_multi_turn_data = parse_data(os.path.join(data_dir, "validation", "dialogues_validation.txt"))
    logger.info("Now parsing test data:")
    test_multi_turn_data = parse_data(os.path.join(data_dir, "test", "dialogues_test.txt"))
    logger.info("Now splitting data:")
    processed_train_data = split_data(train_multi_turn_data)
    processed_valid_data = split_data(valid_multi_turn_data)
    train_valid_json_data = {"train": processed_train_data, "valid": processed_valid_data}
    with open(os.path.join(save_dir, "dailydialog.json"), 'w') as fout:
        fout.write(json.dumps(train_valid_json_data))
    processed_test_data = split_data(test_multi_turn_data, need_candidates=False)
    test_json_data = {"test": processed_test_data}
    with open(os.path.join(save_dir, "dailydialog_test.json"), 'w') as fout:
        fout.write(json.dumps(test_json_data))


if __name__ == "__main__":
    data_dir = "/home1/sxy/datasets/DailyDialog/ijcnlp_dailydialog"
    save_dir = "/home1/sxy/transfer-learning-conv-ai/datasets/dailydialog"
    process_dailydialog(save_dir=save_dir, data_dir=data_dir)
