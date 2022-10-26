# -*- coding: utf-8 -*-
# @Time        : 2022/10/26 10:31
# @Author      : ssxy00
# @File        : evaluate_without_type.py
# @Description :


# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from transformers import OpenAIGPTLMHeadModel, GPT2Tokenizer, OpenAIGPTConfig
from train_without_mc_without_type import SPECIAL_TOKENS, build_input_from_segments, pad_dataset
from utils import get_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


class Metrics:
    """
    copy from github.com/guxd/DialogBERT/blob/master/learner.py
    """

    def __init__(self):
        super(Metrics, self).__init__()
        '''
        self.rouge_evaluator = rouge.Rouge(metrics=['rouge-l'],
                           max_n=4,
                           limit_length=True,
                           length_limit=200,
                           length_limit_type='words',
                           apply_avg=True,
                           apply_best=False,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)
        '''

    @classmethod
    def sim_bleu(self, hyps, ref):
        """
        :param ref - a list of tokens of the reference
        :param hyps - a list of tokens of the hypothesis
        :return maxbleu - recall bleu
        :return avgbleu - precision bleu
        """
        scores = []
        for hyp in hyps:
            try:
                scores.append(sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method7,
                                            weights=[1. / 4, 1. / 4, 1. / 4, 1. / 4]))
            except:
                scores.append(0.0)
        return np.max(scores), np.mean(scores)

    @classmethod
    def sim_meteor(self, hyps, ref):
        """
        :param refs - a list of strings representing references
        :param hyps - a list of tokens of the hypothesis
        :return maxbleu - recall bleu
        :return avgbleu - precision bleu
        """
        scores = []
        for hyp in hyps:
            # try:
            scores.append(meteor_score([ref], hyp))
            # except:
            #    scores.append(0.0)
        return np.max(scores), np.mean(scores)

    @classmethod
    def sim_nist(self, hyps, ref):
        """
        :param refs - a list of strings representing references
        :param hyps - a list of tokens of the hypothesis
        :return maxbleu - recall bleu
        :return avgbleu - precision bleu
        """
        scores = []
        for hyp in hyps:
            try:
                scores.append(sentence_nist([ref], hyp))
            except:
                scores.append(0.0)
        return np.max(scores), np.mean(scores)

    @classmethod
    def sim_rougeL(self, hyps, ref):
        """
        Compute ROUGE-L score given a list of candidates and a reference
        :param hyps: list : candidate sentences to be evaluated
        :param ref: list: reference sentence to be evaluated
        :returns score: float (ROUGE-L score for the candidate evaluated against references)
        This class is copied from https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/rouge/rouge.py
        """

        def lcs(string, sub):
            """
            Calculates longest common subsequence for a pair of tokenized strings
            :param string : list : tokens from a string split using whitespace
            :param sub : list: shorter string, also split using whitespace
            :returns: length (list of int): length of the longest common subsequence between the two strings
            Note: only gives length of the longest common subsequence, not the actual LCS
            This function is copied from https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/rouge/rouge.py
            """
            if len(string) < len(sub): sub, string = string, sub
            lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]
            for j in range(1, len(sub) + 1):
                for i in range(1, len(string) + 1):
                    if string[i - 1] == sub[j - 1]:
                        lengths[i][j] = lengths[i - 1][j - 1] + 1
                    else:
                        lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])
            return lengths[len(string)][len(sub)]

        def rougeL(hyp, refs):
            assert len(refs) > 0 and type(refs[0]) is list, "number of references should >0 for rouge"
            beta = 1.2
            prec, rec = [], []
            for ref in refs:
                _lcs = lcs(ref, hyp)  # compute the longest common subsequence
                prec.append(_lcs / float(len(hyp)))
                rec.append(_lcs / float(len(ref)))
            prec_max, rec_max = max(prec), max(rec)

            if prec_max != 0 and rec_max != 0:
                score = ((1 + beta ** 2) * prec_max * rec_max) / float(rec_max + beta ** 2 * prec_max)
            else:
                score = 0.0
            return score

        scores = []
        for hyp in hyps:
            try:
                scores.append(rougeL(hyp, [ref]))
            except:
                print('exception in RougeL')
                scores.append(0.0)
        return np.max(scores), np.mean(scores)

    @classmethod
    def tok_f1(self, predictions, pred_lens, targets, target_lens):
        batch_size = predictions.shape[0]
        f1s = []
        for b in range(batch_size):
            pred = predictions[b][:pred_lens[b]]
            target = targets[b][:target_lens[b]]
            common = Counter(target) & Counter(pred)
            num_same = sum(common.values())
            if num_same == 0:
                return 0.
            precision = 1. * num_same / pred_lens[b]
            recall = 1. * num_same / target_lens[b]
            f1 = (2. * recall * precision) / (precision + recall)
            f1s.append(f1)
        return np.mean(f1)


def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for evaluation """
    raw_dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)["test"]

    logger.info("Build inputs and labels")
    processed_dataset = defaultdict(list)
    for dialog in raw_dataset:
        persona = dialog["personality"]
        for utterance in dialog["utterances"]:
            history = utterance["history"][-(2 * args.max_history + 1):]
            tgt = utterance["candidates"][0]
            lm_labels = True
            instance = build_input_from_segments(persona, history, tgt, tokenizer, lm_labels)
            for input_name, input_array in instance.items():
                processed_dataset[input_name].append(input_array)

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = []
    processed_dataset = pad_dataset(processed_dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-2]))
    for input_name in ["input_ids", "lm_labels"]:
        tensor = torch.tensor(processed_dataset[input_name])
        tensor_datasets.append(tensor)

    logger.info("Build test dataloaders")
    test_dataset = TensorDataset(*tensor_datasets)
    test_loader = DataLoader(test_dataset, sampler=None, batch_size=1, shuffle=False)

    logger.info("Test dataset (Batch, Candidates, Seq length): {}".format(test_dataset.tensors[0].shape))
    return test_loader, raw_dataset


def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str,
                        default="/home1/sxy/transfer-learning-conv-ai/datasets/personachat/personachat_test.json",
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str,
                        default='/home1/sxy/transfer-learning-conv-ai/datasets/personachat/personachat_test.cache',
                        help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str,
                        help="Path, url or short name of the model")
    parser.add_argument("--optimal_step", type=int, default=482055, help="optimal checkpoint to load")
    parser.add_argument("--save_result_path", default="/home1/sxy/transfer-learning-conv-ai/results/tmp.txt",
                        help="path to save prediction results")

    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")

    args = parser.parse_args()

    logger.info(pformat(args))

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
    model_config = OpenAIGPTConfig(vocab_size=len(tokenizer), n_embd=512, n_layer=6, n_head=8)
    model = OpenAIGPTLMHeadModel(model_config)
    model.to(args.device)
    checkpoint = torch.load(os.path.join(args.model_checkpoint, f"checkpoint_mymodel_{args.optimal_step}.pt"),
                            map_location=args.device)
    model.load_state_dict(checkpoint, strict=False)

    # prepare dataset
    logger.info("Prepare datasets")
    test_loader, raw_test_dataset = get_data_loaders(args, tokenizer)

    # evaluate ppl
    model.eval()
    tqdm_data = tqdm(test_loader, desc='Test: ')
    test_losses = []
    with torch.no_grad():
        for i, data in enumerate(tqdm_data):
            input_ids, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in data)
            lm_loss = model(
                input_ids, labels=lm_labels
            )[0]
            test_losses.append(lm_loss.item())

    # evaluate other metrics
    bleus, meteors, nists, rougeLs = [], [], [], []
    sample_idx = 1
    with torch.no_grad():
        with open(args.save_result_path, "w") as fout:
            for dialog in tqdm(raw_test_dataset):
                persona = dialog["personality"]
                for utterance in dialog["utterances"]:
                    history = utterance["history"][-(2 * args.max_history + 1):]
                    tgt = utterance["candidates"][0]
                    out_ids = sample_sequence(persona, history, tokenizer, model, args)
                    context_strings = []
                    for sent_idx, sent in enumerate(history):
                        sent_string = tokenizer.decode(sent, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=False)
                        context_strings.append(sent_string)
                    ref_sent = tokenizer.decode(tgt, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
                    pred_sent = tokenizer.decode(out_ids, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)
                    fout.write(f"sample {sample_idx}:\n")
                    sample_idx += 1
                    for sent_idx, sent_string in enumerate(context_strings):
                        fout.write(f"context {sent_idx + 1}: {sent_string}\n")
                    fout.write(f"gt: {ref_sent}\n")
                    fout.write(f"pr: {pred_sent}\n\n")
                    _, avg_bleu = Metrics.sim_bleu([pred_sent.split()], ref_sent.split())
                    bleus.append(avg_bleu)
                    _, avg_meteor = Metrics.sim_meteor([pred_sent.split()], ref_sent.split())
                    meteors.append(avg_meteor)
                    _, avg_nist = Metrics.sim_nist([pred_sent.split()], ref_sent.split())
                    nists.append(avg_nist)
                    _, avg_rougeL = Metrics.sim_rougeL([pred_sent.split()], ref_sent.split())
                    rougeLs.append(avg_rougeL)

    # output
    ave_lm_loss = float(np.mean(test_losses))
    ppl = torch.exp(torch.tensor(ave_lm_loss)).item()
    bleu = float(np.mean(bleus))
    meteor = float(np.mean(meteors))
    nist = float(np.mean(nists))
    rougeL = float(np.mean(rougeLs))
    print(f"loss: {ave_lm_loss}\n"
          f"ppl: {ppl}\n"
          f"bleu: {bleu}\n"
          f"meteor: {meteor}\n"
          f"nist: {nist}\n"
          f"rougeL: {rougeL}")


if __name__ == "__main__":
    run()
