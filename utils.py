import json
import math

from tqdm import tqdm
from collections import Counter

import nltk
from nltk.util import ngrams
from nltk import word_tokenize, sent_tokenize

from datasets import Dataset
import torch.nn as nn


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """
    loss with label smoothing
    from fairseq, edit by Bin
    """

    lprobs = lprobs[~target.eq(-100)]
    target = target[~target.eq(-100)]

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    # mean()? Scared to break other math.
    # bin: change from sum to mean
    nll_loss = nll_loss.mean()
    smooth_loss = smooth_loss.mean()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def postprocess_text(preds, labels):
    """
    use for decoding
    """
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def len_adjust(args, split_dict, split_type=None):
    """add length to the input"""

    id_list = split_dict["id"]
    dialogue_list = split_dict["dialogue"]
    summary_list = split_dict["summary"]
    topic_list = split_dict["topic"]
    
    if args.contrastive != "no":
        top_topic_list = split_dict["top_topic"]
        tail_topic_list = split_dict["tail_topic"]

    if args.len_input == "topic-speaker":
        speaker_list = split_dict["speaker"]

    if args.len_input == "topic-speaker-length":
        speaker_list = split_dict["speaker"]

    if args.len_input == "no":
        new_dialogue_list = dialogue_list    

    elif args.len_input == "topic":
        new_dialogue_list = []
        for dialogue, topic in zip(dialogue_list, topic_list):
            new_dialogue = "Topic of Summary: {}. Dialogue: ".format(topic) + dialogue
            new_dialogue_list.append(new_dialogue)

    elif args.len_input == "length":
        new_dialogue_list = []
        for dialogue, summary in zip(dialogue_list, summary_list):
            sum_len = len(summary.split(" "))
            new_dialogue = ("Length of Summary: {}. Dialogue: ".format(sum_len) + dialogue)
            new_dialogue_list.append(new_dialogue)

    elif args.len_input == "topic-length":
        new_dialogue_list = []
        for dialogue, topic, summary in zip(dialogue_list, topic_list, summary_list):
            sum_len = len(summary.split(" "))
            new_dialogue = ("Topic of Summary: {}. Length of Summary: {}. Dialogue: ".format(topic, sum_len) + dialogue)
            new_dialogue_list.append(new_dialogue)

    elif args.len_input == "topic-speaker":
        new_dialogue_list = []
        for dialogue, topic, speaker, summary in zip(dialogue_list, topic_list, speaker_list, summary_list):
            new_dialogue = ("Topic of Summary: {}. Speaker: {}. Dialogue: ".format(topic, speaker) + dialogue)
            new_dialogue_list.append(new_dialogue)

    elif args.len_input == "topic-speaker-length":
        new_dialogue_list = []
        for dialogue, topic, speaker, summary in zip(dialogue_list, topic_list, speaker_list, summary_list):
            sum_len = len(summary.split(" "))
            new_dialogue = ("Topic of Summary: {}. Speaker: {}. Length of Summary: {}. Dialogue: ".format(topic, speaker, sum_len) + dialogue)
            new_dialogue_list.append(new_dialogue)
    
    if args.contrastive != "no":
        if args.len_input == "topic":
            new_top_topic_list = []
            new_tail_topic_list = []
            for dialogue, summary, top_topic, tail_topic in zip(dialogue_list, summary_list, top_topic_list, tail_topic_list):
                new_dialogue = ("Topic of Summary: {}. Dialogue: ".format(top_topic) + dialogue)
                new_top_topic_list.append(new_dialogue)
                
                new_dialogue = ("Topic of Summary: {}. Dialogue: ".format(tail_topic) + dialogue)
                new_tail_topic_list.append(new_dialogue)
        
        elif args.len_input == "topic-length":
            new_top_topic_list = []
            new_tail_topic_list = []
            for dialogue, summary, top_topic, tail_topic in zip(dialogue_list, summary_list, top_topic_list, tail_topic_list):
                sum_len = len(summary.split(" "))
                new_dialogue = ("Topic of Summary: {}. Length of Summary: {}. Dialogue: ".format(top_topic, sum_len) + dialogue)
                new_top_topic_list.append(new_dialogue)
                
                new_dialogue = ("Topic of Summary: {}. Length of Summary: {}. Dialogue: ".format(tail_topic, sum_len) + dialogue)
                new_tail_topic_list.append(new_dialogue)

        elif args.len_input == "topic-speaker":
            new_top_topic_list = []
            new_tail_topic_list = []
            for dialogue, topic, speaker, top_topic, tail_topic in zip(dialogue_list, topic_list, speaker_list, top_topic_list, tail_topic_list):
                new_dialogue = ("Topic of Summary: {}. Speaker: {}. Dialogue: ".format(top_topic, speaker) + dialogue)
                new_top_topic_list.append(new_dialogue)
                
                new_dialogue = ("Topic of Summary: {}. Speaker: {}. Dialogue: ".format(tail_topic, speaker) + dialogue)
                new_tail_topic_list.append(new_dialogue)
                
        elif args.len_input == "topic-speaker-length":
            new_top_topic_list = []
            new_tail_topic_list = []
            for dialogue, topic, speaker, top_topic, tail_topic in zip(dialogue_list, topic_list, speaker_list, top_topic_list, tail_topic_list):
                sum_len = len(summary.split(" "))
                new_dialogue = ("Topic of Summary: {}. Speaker: {}. Length of Summary: {}. Dialogue: ".format(top_topic, speaker, sum_len) + dialogue)
                new_top_topic_list.append(new_dialogue)
                
                new_dialogue = ("Topic of Summary: {}. Speaker: {}. Length of Summary: {}. Dialogue: ".format(tail_topic, speaker, sum_len) + dialogue)
                new_tail_topic_list.append(new_dialogue)


    if args.len_output == "no" or split_type == "val" or split_type == "test":
        new_summary_list = summary_list
        

    split_dict = {
        "id": id_list,
        "dialogue": new_dialogue_list,
        "summary": new_summary_list,
    }

    if args.contrastive != "no":
        split_dict["top_topic_dialogue"] = new_top_topic_list
        split_dict["tail_topic_dialogue"] = new_tail_topic_list

    split_dict = Dataset.from_dict(split_dict)

    return split_dict


def cosine_embedding_loss(pos, neg, contrastive, margin=0.5):
    cs_loss = nn.CosineEmbeddingLoss(margin)
    loss_cosine_embedding = cs_loss(pos, neg, contrastive)
    return loss_cosine_embedding
