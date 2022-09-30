# -*- coding: utf-8 -*-
# @Time        : 2022/9/26 23:07
# @Author      : ssxy00
# @File        : prepare_personachat.py
# @Description :


"""
处理数据时要保证 history 中的 utterance 个数不少于 7，即除了 last utterance 至少还有三轮对话历史
在切数据时，不限制对话起始于 x or y
"""

import json
import os
import random
import logging
import torch
from tqdm import tqdm


logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def parse_data(data_path):
    """
    :param data_path:
    :return: List[{'your_persona': List[str],
                   'partner_persona': List[str],
                   'dialog': List[str]}]
    """
    logger.info("loading dialogs from file")
    with open(data_path, 'r', encoding='utf-8') as file:
        data = []
        for line in tqdm(file.readlines()):
            line = line.strip()

            if len(line) == 0:
                continue

            space_idx = line.find(' ')
            if space_idx == -1:
                dialog_idx = int(line)
            else:
                dialog_idx = int(line[:space_idx])

            if int(dialog_idx) == 1:
                data.append({'your_persona': [], 'partner_persona': [], 'dialog': []})

            dialog_line = line[space_idx + 1:].split('\t')
            dialog_line = [l.strip() for l in dialog_line]

            if dialog_line[0].startswith('your persona:'):
                persona_info = dialog_line[0].replace('your persona: ', '')
                data[-1]['your_persona'].append(persona_info)
            if dialog_line[0].startswith('partner\'s persona:'):
                persona_info = dialog_line[0].replace('partner\'s persona: ', '')
                data[-1]['partner_persona'].append(persona_info)

            elif len(dialog_line) > 1:
                data[-1]['dialog'].append(dialog_line[0])
                data[-1]['dialog'].append(dialog_line[1])
        return data


def split_data(data, need_candidates=True):
    all_utterances = []
    for dialog in data:
        all_utterances += dialog["dialog"]
    logger.info("splitting multi-turn dialogs into personality + utterances")
    processed_data = []
    for dialog in tqdm(data):
        assert len(dialog['dialog']) % 2 == 0  # dialog starts from partner and ends with you
        processed_dialog_your = {"personality": dialog["your_persona"], "utterances": []}
        processed_dialog_partner = {"personality": dialog["partner_persona"], "utterances": []}
        for utter_idx in range(7, len(dialog['dialog'])):
            history = dialog['dialog'][: utter_idx]
            tgt = dialog['dialog'][utter_idx]
            if need_candidates:
                candidates = random.sample(all_utterances, 19) + [tgt]
            else:
                candidates = [tgt]
            if utter_idx % 2:  # you
                processed_dialog_your["utterances"].append({"candidates": candidates, "history": history})
            else:  # partner
                processed_dialog_partner["utterances"].append({"candidates": candidates, "history": history})
        processed_data.append(processed_dialog_your)
        processed_data.append(processed_dialog_partner)
    return processed_data

def process_personachat(save_dir, train_data_path, valid_data_path):
    logger.info("Now parsing training data:")
    train_multi_turn_data = parse_data(train_data_path)
    logging.info(f"After parsing, we get {len(train_multi_turn_data)} training dialogs.")
    logger.info("Now processing valid and test data:")
    multi_turn_data = parse_data(valid_data_path)
    # 原本的验证集分出一半作为测试集
    random.seed(0)
    random.shuffle(multi_turn_data)
    n_valid_data = len(multi_turn_data)
    valid_multi_turn_data = multi_turn_data[: n_valid_data // 2]
    test_multi_turn_data = multi_turn_data[n_valid_data // 2:]
    print(f"valid data has {n_valid_data} samples, "
          f"now split into valid: {len(valid_multi_turn_data)} and test: {len(test_multi_turn_data)}")
    processed_train_data = split_data(train_multi_turn_data)
    processed_valid_data = split_data(valid_multi_turn_data)
    train_valid_json_data = {"train": processed_train_data, "valid": processed_valid_data}
    with open(os.path.join(save_dir, "personachat.json"), 'w') as fout:
        fout.write(json.dumps(train_valid_json_data))
    processed_test_data = split_data(test_multi_turn_data, need_candidates=False)
    test_json_data = {"test": processed_test_data}
    with open(os.path.join(save_dir, "personachat_test.json"), 'w') as fout:
        fout.write(json.dumps(test_json_data))


if __name__ == "__main__":
    save_dir = "/home1/sxy/transfer-learning-conv-ai/datasets/personachat"
    train_data_path = "/home1/sxy/datasets/ConvAI2_data/ConvAI2/train_both_original_no_cands.txt"
    valid_data_path = "/home1/sxy/datasets/ConvAI2_data/ConvAI2/valid_both_original_no_cands.txt"
    process_personachat(save_dir=save_dir, train_data_path=train_data_path, valid_data_path=valid_data_path)
