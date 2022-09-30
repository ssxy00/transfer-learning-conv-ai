# -*- coding: utf-8 -*-
# @Time        : 2022/9/27 16:02
# @Author      : ssxy00
# @File        : prepare_multiwoz.py
# @Description :


"""
处理数据时要保证 history 中的 utterance 个数不少于 7，即除了 last utterance 至少还有三轮对话历史
在切数据时，不限制对话起始于 x or y
"""

import re
import os
import random
import json
import logging
import torch
from tqdm import tqdm


logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def parse_data(data_path):
    """
    :param data_path:
    :return: List[{'dialog': List[str]}]
    参考了 https://github.com/guxd/DialogBERT/blob/master/prepare_data.py
    """
    logger.info("loading dialogs from file")
    timepat = re.compile("\d{1,2}[:]\d{1,2}")
    pricepat = re.compile("\d{1,3}[.]\d{1,2}")

    def normalize(text):
        text = text.lower()
        text = re.sub(r'^\s*|\s*$', '', text)  # replace white spaces in front and end
        # hotel domain pfb30
        text = re.sub(r"b&b", "bed and breakfast", text)
        text = re.sub(r"b and b", "bed and breakfast", text)
        # normalize phone number
        ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m[0], sidx)
                if text[sidx - 1] == '(': sidx -= 1
                eidx = text.find(m[-1], sidx) + len(m[-1])
                text = text.replace(text[sidx:eidx], ''.join(m))
        # normalize postcode
        ms = re.findall(
            '([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})', text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m, sidx)
                eidx = sidx + len(m)
                text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]
        text = re.sub(u"(\u2018|\u2019)", "'", text)  # weird unicode bug
        # replace time and and price
        text = re.sub(timepat, ' [value_time] ', text)
        text = re.sub(pricepat, ' [value_price] ', text)
        # text = re.sub(pricepat2, '[value_price]', text)
        # replace st.
        text = text.replace(';', ',')
        text = re.sub('$\/', '', text)
        text = text.replace('/', ' and ')
        # replace other special characters
        text = text.replace('-', ' ')
        text = re.sub('[\":\<>@\(\)]', '', text)
        text = re.sub(' +', ' ', text)  # remove multiple spaces
        # concatenate numbers
        tmp = text
        tokens = text.split()
        i = 1
        while i < len(tokens):
            if re.match(u'^\d+$', tokens[i]) and re.match(u'\d+$', tokens[i - 1]):
                tokens[i - 1] += tokens[i]
                del tokens[i]
            else:
                i += 1
        text = ' '.join(tokens)
        return text

    dialogs = []
    data = json.load(open(data_path, 'r'))
    for dialogue_name in tqdm(data):
        utts = []
        dialogue = data[dialogue_name]
        for idx, turn in enumerate(dialogue['log']):
            # normalization, split and delexicalization of the sentence
            sent = normalize(turn['text'])
            utts.append(sent)
        dialogs.append({"dialog": utts})
    return dialogs[:-2000], dialogs[-2000:-1000], dialogs[-1000:]


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


def process_multiwoz(save_dir, raw_data_path):
    train_multi_turn_data, valid_multi_turn_data, test_multi_turn_data = parse_data(raw_data_path)
    print(f"train data has {len(train_multi_turn_data)} dialogs")
    print(f"valid data has {len(valid_multi_turn_data)} dialogs")
    print(f"test data has {len(test_multi_turn_data)} dialogs")
    logger.info("Now splitting data:")
    processed_train_data = split_data(train_multi_turn_data)
    processed_valid_data = split_data(valid_multi_turn_data)
    train_valid_json_data = {"train": processed_train_data, "valid": processed_valid_data}
    with open(os.path.join(save_dir, "multiwoz.json"), 'w') as fout:
        fout.write(json.dumps(train_valid_json_data))
    processed_test_data = split_data(test_multi_turn_data, need_candidates=False)
    test_json_data = {"test": processed_test_data}
    with open(os.path.join(save_dir, "multiwoz_test.json"), 'w') as fout:
        fout.write(json.dumps(test_json_data))


if __name__ == "__main__":
    save_dir = "/home1/sxy/transfer-learning-conv-ai/datasets/multiwoz"
    raw_data_path = "/home1/sxy/datasets/MultiWOZ/MultiWOZ_1.0/data.json"
    process_multiwoz(save_dir=save_dir, raw_data_path=raw_data_path)
