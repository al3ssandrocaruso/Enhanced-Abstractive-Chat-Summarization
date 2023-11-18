import os
os.environ['WANDB_SILENT']="true"

import sys
sys.path.append('../')
import argparse
import random
import json
import nltk
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
#from datasets import load_metric
from datasets import load_metric
import wandb
import json
import pickle
from tkinter import dialog
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from datasets import load_dataset
import os
import spacy
import re
import random

# replicated dataset classes here for problems of import in colab
class SamsumDataset(Dataset):
    def __init__(self, encoder_max_len, decoder_max_len, split_type,
                 tokenizer, extra_context=False, extra_supervision=False,
                 paracomet=False, relation="xReason", supervision_relation="xIntent",
                 roberta=False, sentence_transformer=False):
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len
        self.split_type = split_type
        self.tokenizer = tokenizer

        self.extra_context = extra_context
        self.extra_supervision = extra_supervision

        self.relation = relation
        self.paracomet = paracomet
        if self.paracomet and (self.relation[0] != "<"):
            self.relation = f"<|{self.relation}|>"

        self.supervision_relation = supervision_relation
        if self.paracomet and (self.supervision_relation[0] != "<"):
            self.supervision_relation = f"<|{self.supervision_relation}|>"

        self.roberta = roberta
        self.sentence_transformer = sentence_transformer
        print(self.relation)
        ##################################################

        self.data = load_dataset('samsum', split=split_type)
        self.dialogue = self.data['dialogue']
        self.summary = self.data['summary']
        self.id = self.data['id']

        self.nlp = spacy.load('en_core_web_sm')

        ###########################
        #   LOAD .json dataset    #
        ###########################
        if self.extra_context == True:
            if self.paracomet == False:
                ##### COMET #####
                with open(f"/content/SICK_Summarization/data/COMET_data/comet/dialogue/samsum/comet_{self.split_type}.json") as f:
                    self.dialogue_comet_inference = json.load(f)

                if self.roberta:
                    print('ROBERTA ON!')
                    with open(
                            f"/content/SICK_Summarization/data/COMET_data/comet/dialogue/samsum/roberta_nli/roberta_classified_top1_{self.split_type}.json") as f:
                        self.roberta_classified_z = json.load(f)
                if self.sentence_transformer:
                    with open(
                            f"/content/SICK_Summarization/data/COMET_data/comet/dialogue/samsum/sentence_transformer/comet_{self.split_type}_z.json") as f:
                        self.sentence_transformer_classified_z = json.load(f)


            else:

                with open(
                        f"/content/SICK_Summarization/data/COMET_data/paracomet/dialogue/samsum/dialog_{self.split_type}_split5_collated.json") as f:
                    self.dialogue_comet_inference = json.load(f)
                if self.roberta:
                    print('ROBERTA ON!')
                    with open(
                            f"/content/SICK_Summarization/data/COMET_data/paracomet/dialogue/samsum/roberta_nli/paracomet_samsum_roberta_classified_top1_{self.split_type}.json") as f:
                        self.roberta_classified_z = json.load(f)
                if self.sentence_transformer:
                    with open(
                            f"/content/SICK_Summarization/data/COMET_data/paracomet/dialogue/samsum/sentence_transformer/paracomet_{self.split_type}_z.json") as f:
                        self.sentence_transformer_classified_z = json.load(f)

        if self.extra_supervision == True:  # use commonsense w
            if self.split_type == 'train':
                if self.paracomet == False:  # plain COMET
                    with open(f"/content/SICK_Summarization/data/COMET_data/comet/summary/samsum/comet_train_w.json") as f:
                        self.summary_comet_inference = json.load(f)

                    if self.roberta:
                        print('ROBERTA ON!')
                        with open(
                                f"/content/SICK_Summarization/data/COMET_data/comet/summary/samsum/roberta_nli/roberta_classified_top1_w.json") as f:
                            self.roberta_classified_w = json.load(f)
                    if self.sentence_transformer:
                        with open(
                                f"/content/SICK_Summarization/data/COMET_data/comet/summary/samsum/sentence_transformer/comet_train_w.json") as f:
                            self.sentence_transformer_classified_w = json.load(f)
                else:
                    with open(f"/content/SICK_Summarization/data/COMET_data/paracomet/summary/samsum/summary_train_split5_collated.json") as f:
                        self.summary_comet_inference = json.load(f)
                    if self.roberta:
                        print('ROBERTA ON!')
                        with open(
                                f"/content/SICK_Summarization/data/COMET_data/paracomet/summary/samsum/roberta_nli/roberta_classified_top1_w.json") as f:
                            self.roberta_classified_w = json.load(f)

                    if self.sentence_transformer:
                        with open(
                                f"/content/SICK_Summarization/data/COMET_data/paracomet/summary/samsum/sentence_transformer/paracomet_train_w.json") as f:
                            self.sentence_transformer_classified_w = json.load(f)

        self.data_len = len(self.data)

        # total = [i for i in range(self.data_len)]
        # self.low_res = random.sample(total,self.data_len/10)
        # print(self.low_res)

    def process_media_msg(self, sentence, person, commonsense):
        # print(person)
        if ('<file_photo>' in sentence) or ('<photo_file>' in sentence) or ('<file_picture>' in sentence):
            return "<I> " + person + " sent a photo. </I>" + '\n'
        elif ('<video>' in sentence) or ('<file_video>' in sentence):
            return "<I> " + person + " sent a video. </I>" + '\n'
        elif '<file_gif>' in sentence:
            return "<I> " + person + " sent a file. </I>" + '\n'
        elif ('<file_other>' in sentence) or ('<file_others>' in sentence):
            return "<I> " + person + " sent a file. </I>" + '\n'
        elif ('<link>' in sentence) or ('<file_link>' in sentence):
            return "<I> " + person + " sent a link. </I>" + '\n'
        elif '<location>' in sentence:
            return "<I> " + person + " sent a location. </I>" + '\n'
        else:
            if commonsense.strip() != 'none':
                return "<I> " + commonsense.strip() + ". </I>" + '\n'
            else:
                return ""

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        if self.extra_context == False:
            # (1, sequence_length)
            encoded_dialogue = self.tokenizer(self.dialogue[index],
                                              padding='max_length',
                                              truncation=True,
                                              max_length=self.encoder_max_len,
                                              return_tensors='pt')
        else:
            if self.paracomet == False:  # plain COMET
                try:

                    dia = self.dialogue_comet_inference[self.id[index]]

                    dialogue = ""
                    for sent_idx, sent in enumerate(dia):
                        person = sent['speaker'].replace(": ", "").replace(":", "").strip()
                        sentence = sent['sentence'].strip()
                        if self.roberta:
                            commonsense = self.roberta_classified_z[self.id[index]][str(sent_idx)]["out"]

                        elif self.sentence_transformer:
                            commonsense = self.sentence_transformer_classified_z[self.id[index]][str(sent_idx)]["out"]
                            # print(commonsense)
                        else:
                            # print(self.relation)
                            commonsense = sent[self.relation][0].strip()
                            # print(commonsense)

                        commonsense = commonsense.replace("PersonX", "Person").replace("PersonY", "Person")
                        dialogue += person + " said \"" + sentence + ".\"" + '\n'
                        if sent['speaker'] + sentence != commonsense:
                            # print(self.process_media_msg(sentence, person, commonsense))
                            dialogue += self.process_media_msg(sentence, person, commonsense)
                except KeyError:
                    print("key error")
                    dialogue = self.dialogue[index]



            else:  # use PARACOMETd
                try:
                    dia = self.dialogue_comet_inference[self.id[index]]
                    dialogue = ""
                    for sent_idx, sent in dia.items():
                        sentence = sent['sentence'].strip()

                        person = sentence.split()[0]

                        if self.roberta:
                            commonsense = self.roberta_classified_z[self.id[index]][str(sent_idx)]["out"]

                        elif self.sentence_transformer:
                            commonsense = self.sentence_transformer_classified_z[self.id[index]][str(sent_idx)]["out"]

                        else:
                            commonsense = sent[self.relation][0].strip()

                        dialogue += sentence + '\n'

                        if sentence != commonsense:
                            dialogue += self.process_media_msg(sentence, person, commonsense)

                except KeyError:  # when an error occurred while processing commonsense, just give plain utterance as output
                    print("key error")
                    # print(index)
                    # print(self.id[index])
                    # print(self.dialogue_comet_inference.keys())
                    dialogue = self.dialogue[index]

            encoded_dialogue = self.tokenizer(dialogue,
                                              padding='max_length',
                                              truncation=True,
                                              max_length=self.encoder_max_len,
                                              return_tensors='pt')

        # (1, sequence_length)
        with self.tokenizer.as_target_tokenizer():
            encoded_summary = self.tokenizer(self.summary[index],
                                             padding='max_length',
                                             truncation=True,
                                             max_length=self.decoder_max_len,
                                             return_tensors='pt')

        model_inputs = encoded_dialogue
        model_inputs['input_ids'] = model_inputs['input_ids'].squeeze(0)
        model_inputs['attention_mask'] = model_inputs['attention_mask'].squeeze(0)
        model_inputs['labels'] = encoded_summary['input_ids'].squeeze(0)

        if self.extra_supervision == True:
            if self.split_type == 'train':
                def split_sentences(text, speaker):
                    doc = self.nlp(text)
                    sents = [speaker.replace(":", "") + ' said "' + sent.text + '"' for sent in doc.sents]
                    return sents

                if self.paracomet == False:  # plain COMET
                    summary_commonsense = ""
                    if self.roberta:
                        for _, summ in self.roberta_classified_w[self.id[index]].items():
                            commonsense = summ["out"].strip() + ". "
                            commonsense = commonsense.replace("PersonX", "Person").replace("PersonY", "Person")
                            summary_commonsense += commonsense
                    elif self.sentence_transformer:
                        for _, summ in self.sentence_transformer_classified_w[self.id[index]].items():
                            commonsense = summ["out"].strip() + ". "
                            commonsense = commonsense.replace("PersonX", "Person").replace("PersonY", "Person")
                            summary_commonsense += commonsense
                    else:
                        for summ in self.summary_comet_inference[self.id[index]]:
                            commonsense = summ[self.supervision_relation][0].strip() + '. '
                            commonsense = commonsense.replace("PersonX", "Person").replace("PersonY", "Person")
                            summary_commonsense += commonsense

                    with self.tokenizer.as_target_tokenizer():
                        encoded_extra_supervision = self.tokenizer(summary_commonsense,
                                                                   padding='max_length',
                                                                   truncation=True,
                                                                   max_length=self.decoder_max_len,
                                                                   return_tensors='pt')

                    model_inputs['extra_labels'] = encoded_extra_supervision['input_ids'].squeeze(0)
                else:
                    if index == 6054:
                        summary_commonsense = "problem with presentation."
                    elif self.roberta:
                        summary_commonsense = ""
                        for _, summ in self.roberta_classified_w[self.id[index]].items():
                            commonsense = summ["out"].strip() + ". "
                            commonsense = commonsense.replace("PersonX", "Person").replace("PersonY", "Person")
                            summary_commonsense += commonsense
                    elif self.sentence_transformer:
                        summary_commonsense = ""
                        for _, summ in self.sentence_transformer_classified_w[self.id[index]].items():
                            commonsense = summ["out"].strip().strip(".") + ". "
                            commonsense = commonsense.replace("PersonX", "Person").replace("PersonY", "Person")
                            summary_commonsense += commonsense
                    else:
                        summary_commonsense = ""
                        for _, summ in self.summary_comet_inference[self.id[index]].items():
                            try:
                                summary_commonsense += summ[self.supervision_relation][0].strip() + '. '
                            except KeyError:
                                print("key error in supervision")
                                summary_commonsense = ""

                    with self.tokenizer.as_target_tokenizer():
                        encoded_extra_supervision = self.tokenizer(summary_commonsense,
                                                                   padding='max_length',
                                                                   truncation=True,
                                                                   max_length=self.decoder_max_len,
                                                                   return_tensors='pt')

                    model_inputs['extra_labels'] = encoded_extra_supervision['input_ids'].squeeze(0)
                # print(summary_commonsense)

        return model_inputs
class SamsumDataset_total:
    def __init__(self, encoder_max_len, decoder_max_len, tokenizer,
                 extra_context=False, extra_supervision=False, paracomet=False,
                 relation="xReason", supervision_relation='isAfter',
                 roberta=False, sentence_transformer=False):
        self.train_dataset = SamsumDataset(encoder_max_len, decoder_max_len, 'train', tokenizer,
                                           extra_context=extra_context, extra_supervision=extra_supervision,
                                           paracomet=paracomet, relation=relation,
                                           supervision_relation=supervision_relation, roberta=roberta,
                                           sentence_transformer=sentence_transformer)
        self.eval_dataset = SamsumDataset(encoder_max_len, decoder_max_len, 'validation', tokenizer,
                                          extra_context=extra_context, extra_supervision=extra_supervision,
                                          paracomet=paracomet, relation=relation,
                                          supervision_relation=supervision_relation, roberta=roberta,
                                          sentence_transformer=sentence_transformer)
        self.test_dataset = SamsumDataset(encoder_max_len, decoder_max_len, 'test', tokenizer,
                                          extra_context=extra_context, extra_supervision=extra_supervision,
                                          paracomet=paracomet, relation=relation,
                                          supervision_relation=supervision_relation, roberta=roberta,
                                          sentence_transformer=sentence_transformer)

    def getTrainData(self):
        return self.train_dataset

    def getEvalData(self):
        return self.eval_dataset

    def getTestData(self):
        return self.test_dataset
def custom_load_dataset(type, split):
    if type == "dialogsum":
        dir = f"./DialogSum_Data/dialogsum.{split}.jsonl"
        data = {'dialogue': [], 'summary': [], 'id': []}
        with open(dir, 'r') as json_file:
            json_list = list(json_file)
        if split == "train":
            for json_str in json_list:
                result = json.loads(json_str)
                data['dialogue'].append(result['dialogue'])
                data['summary'].append((result['summary']))
                data['id'].append((result['fname'][6:]))
        elif split == "validation":
            for json_str in json_list:
                result = json.loads(json_str)
                data['dialogue'].append(result['dialogue'])
                data['summary'].append((result['summary']))
                data['id'].append((result['fname'][4:]))
        elif split == "test":
            data = {'dialogue': [], 'summary': [], 'id': [], 'summary2': [], 'summary3': []}
            for json_str in json_list:
                result = json.loads(json_str)
                data['dialogue'].append(result['dialogue'])
                data['summary'].append((result['summary1']))
                data['summary2'].append((result['summary2']))
                data['summary3'].append((result['summary3']))
                data['id'].append((result['fname'][5:]))
        else:
            print("non-existing")
            os.exit()
        return data
class DialogsumDataset(Dataset):
    def __init__(self, encoder_max_len, decoder_max_len, split_type, tokenizer, extra_context=False,
                 extra_supervision=False, paracomet=False, relation="xReason", supervision_relation="isAfter",
                 roberta=False, sentence_transformer=False):
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len
        self.split_type = split_type
        self.tokenizer = tokenizer

        self.extra_context = extra_context
        self.extra_supervision = extra_supervision

        self.relation = relation
        self.paracomet = paracomet

        self.roberta = roberta
        self.sentence_transformer = sentence_transformer

        if (self.paracomet) and ("<" != self.relation[0]):
            self.relation = f"<|{self.relation}|>"

        self.supervision_relation = supervision_relation
        if not self.sentence_transformer:
            print(self.relation)

        else:
            if self.paracomet:
                print("PARACOMET sentence-transformer")
            else:
                print("COMET sentence-transformer")

        ##################################################

        self.data = custom_load_dataset('dialogsum', split=split_type)
        self.dialogue = self.data['dialogue']
        self.summary = self.data['summary']
        if split_type == "test":
            self.summary2 = self.data['summary2']
            self.summary3 = self.data['summary3']
        self.id = self.data['id']

        self.nlp = spacy.load('en_core_web_sm')

        if self.extra_context == True:
            if self.paracomet == False:
                ###########################
                # CODE FOR COMET
                ###########################

                with open(f"/content/SICK_Summarization/data/COMET_data/comet/dialogue/dialogsum/comet_{self.split_type}.json") as f:
                    self.dialogue_comet_inference = json.load(f)

                if self.roberta:
                    with open(
                            f"/content/SICK_Summarization/data/COMET_data/comet/dialogue/dialogsum/roberta_nli/roberta_classified_top1_{self.split_type}.json") as f:
                        self.roberta_classified_z = json.load(f)

                if self.sentence_transformer:
                    with open(
                            f"/content/SICK_Summarization/data/COMET_data/comet/dialogue/dialogsum/sentence_transformer/comet_{self.split_type}_z.json",
                            "r") as f:
                        self.sentence_transformer_classified_z = json.load(f)


            else:
                ###########################
                # CODE FOR PARACOMET
                ###########################

                with open(
                        f"/content/SICK_Summarization/data/COMET_data/paracomet/dialogue/dialogsum/dialog_{self.split_type}_split5_collated.json") as f:
                    self.dialogue_comet_inference = json.load(f)

                if self.roberta:
                    with open(
                            f"/content/SICK_Summarization/data/COMET_data/paracomet/dialogue/dialogsum/roberta_nli/paracomet_dialogsum_roberta_classified_top1_{self.split_type}.json") as f:
                        self.roberta_classified_z = json.load(f)

                if self.sentence_transformer:
                    with open(
                            f"/content/SICK_Summarization/data/COMET_data/paracomet/dialogue/dialogsum/sentence_transformer/paracomet_{self.split_type}_z.json",
                            "r") as f:
                        self.sentence_transformer_classified_z = json.load(f)

        if self.extra_supervision == True:
            if self.split_type == 'train':
                if self.paracomet == False:
                    ######################
                    # CODE FOR COMET
                    ######################
                    with open(f"/content/SICK_Summarization/data/COMET_data/comet/summary/dialogsum/comet_train_w.json") as f:
                        self.summary_comet_inference = json.load(f)

                    if self.roberta:
                        with open(
                                f"/content/SICK_Summarization/data/COMET_data/comet/dialogue/dialogsum/roberta_nli/roberta_classified_top1_w.json") as f:
                            self.roberta_classified_w = json.load(f)

                    if sentence_transformer:
                        with open(f"/content/SICK_Summarization/data/COMET_data/comet/summary/dialogsum/sentence_transformer/comet_train_w.json",
                                  "r") as f:
                            self.sentence_transformer_classified_w = json.load(f)

                else:
                    ########################
                    # CODE FOR PARACOMET
                    ########################
                    with open("/content/SICK_Summarization/data/COMET_data/paracomet/summary/dialogsum/summary_train_split5_collated.json") as f:
                        self.summary_comet_inference = json.load(f)

                    if self.roberta:
                        with open(
                                "/content/SICK_Summarization/data/COMET_data/paracomet/summary/dialogsum/roberta_nli/roberta_classified_top1_w.json") as f:
                            self.roberta_classified_w = json.load(f)

                    if sentence_transformer:
                        with open(
                                "/content/SICK_Summarization/data/COMET_data/paracomet/summary/dialogsum/sentence_transformer/paracomet_train_w.json",
                                "r") as f:
                            self.sentence_transformer_classified_w = json.load(f)

        self.data_len = len(self.id)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        if self.extra_context == False:
            # (1, sequence_length)
            encoded_dialogue = self.tokenizer(self.dialogue[index],
                                              padding='max_length',
                                              truncation=True,
                                              max_length=self.encoder_max_len,
                                              return_tensors='pt')
        else:
            if self.split_type == "validation":
                dialog_id = f"dev_{self.id[index]}"

            else:
                dialog_id = f"{self.split_type}_{self.id[index]}"
            if self.sentence_transformer:
                cur_dialog_data = self.sentence_transformer_classified_z[dialog_id]
                dialogue = ""
                for sentence_idx in range(len(cur_dialog_data.keys())):
                    sentence = cur_dialog_data[str(sentence_idx)]["sentence"]
                    relation = cur_dialog_data[str(sentence_idx)]["relation"]
                    commonsense = cur_dialog_data[str(sentence_idx)]["out"]

                    dialogue += sentence + "\n"
                    dialogue += '<I> '
                    dialogue += commonsense + '.'
                    dialogue += ' </I>' + '\n'

            elif self.roberta:
                cur_dialog_data = self.roberta_classified_z[dialog_id]
                dialogue = ""
                for sentence_idx in range(len(cur_dialog_data.keys())):
                    try:
                        sentence = cur_dialog_data[str(sentence_idx)]["sentence"]
                        relation = cur_dialog_data[str(sentence_idx)]["relation"]
                        commonsense = cur_dialog_data[str(sentence_idx)]["out"]

                        dialogue += sentence + "\n"
                        dialogue += "<I> "
                        dialogue += commonsense + "."
                        dialogue += " </I>" + "\n"
                    except KeyError:
                        continue


            elif self.paracomet == False:
                #######################
                # CODE FOR COMET
                #######################
                # extra context exist
                # z is available
                splitted_dialogue = self.dialogue[index].replace('\r\n', '\n').split('\n')

                def split_sentences(text, speaker):
                    doc = self.nlp(text)
                    sents = [speaker.replace(":", "") + ' said "' + sent.text + '"' for sent in doc.sents]
                    return sents

                splitted_sentences = []
                for idx, utterance in enumerate(splitted_dialogue):
                    speaker = re.search(".*?\:", utterance)[0]
                    utterance = utterance.replace(speaker, "").strip()
                    utterance = split_sentences(utterance, speaker)
                    splitted_sentences.extend(utterance)

                dialogue = ""
                idx = 0
                for utterance in splitted_sentences:
                    dialogue += utterance + '\n'
                    if self.split_type == 'train':
                        try:
                            while True:
                                if self.dialogue_comet_inference['train_' + self.id[index]][idx]['sentence'] not in (
                                "#Person1#:", "#Person2#:"):
                                    commonsense = \
                                    self.dialogue_comet_inference['train_' + self.id[index]][idx][self.relation][
                                        0].strip()
                                    # commonsense = commonsense.replace("PersonX","Person").replace("PersonY","Person")
                                    break
                                else:
                                    idx += 1
                                continue
                        except:
                            continue
                    elif self.split_type == 'validation':
                        try:
                            while True:
                                if self.dialogue_comet_inference['dev_' + self.id[index]][idx]['sentence'] not in (
                                "#Person1#:", "#Person2#:"):
                                    commonsense = \
                                    self.dialogue_comet_inference['dev_' + self.id[index]][idx][self.relation][
                                        0].strip()
                                    commonsense = commonsense.replace("PersonX", "Person").replace("PersonY", "Person")
                                    break
                                else:
                                    idx += 1
                                continue
                        except:
                            continue
                    else:  # self.split_type=='test':
                        try:
                            while True:
                                if self.dialogue_comet_inference['test_' + self.id[index]][idx]['sentence'] not in (
                                "#Person1#:", "#Person2#:"):
                                    commonsense = \
                                    self.dialogue_comet_inference['test_' + self.id[index]][idx][self.relation][
                                        0].strip()
                                    # commonsense = commonsense.replace("PersonX","Person").replace("PersonY","Person")
                                    break
                                else:
                                    idx += 1
                                continue

                        except:
                            continue
                    if 'none' not in commonsense:
                        dialogue += '<I> '
                        dialogue += commonsense + '.'
                        dialogue += ' </I>' + '\n'
                    idx += 1
            ############################### PARACOMET START #######################################################
            else:
                if self.split_type == 'validation':
                    dia = self.dialogue_comet_inference['dev' + '_' + self.id[index]]
                else:
                    dia = self.dialogue_comet_inference[self.split_type + '_' + self.id[index]]
                dialogue = ""
                for _, sent in dia.items():
                    sentence = sent['sentence'].strip()
                    person = sentence.split()[0]
                    commonsense = sent[self.relation][0].strip()

                    dialogue += sentence + '\n'

                    if sentence != commonsense:
                        if ('<file_photo>' in sentence) or ('<photo_file>' in sentence) or (
                                '<file_picture>' in sentence):
                            dialogue += "<I> " + person + " sent a photo. </I>" + '\n'
                        elif ('<video>' in sentence) or ('<file_video>' in sentence):
                            dialogue += "<I> " + person + " sent a video. </I>" + '\n'
                        elif '<file_gif>' in sentence:
                            dialogue += "<I> " + person + " sent a file. </I>" + '\n'
                        elif ('<file_other>' in sentence) or ('<file_others>' in sentence):
                            dialogue += "<I> " + person + " sent a file. </I>" + '\n'
                        elif ('<link>' in sentence) or ('<file_link>' in sentence):
                            dialogue += "<I> " + person + " sent a link. </I>" + '\n'
                        elif '<location>' in sentence:
                            dialogue += "<I> " + person + " sent a location. </I>" + '\n'
                        else:
                            if commonsense.strip() != 'none':
                                dialogue += "<I> " + commonsense.strip() + ". </I>" + '\n'

            encoded_dialogue = self.tokenizer(dialogue,
                                              padding='max_length',
                                              truncation=True,
                                              max_length=self.encoder_max_len,
                                              add_special_tokens=True,
                                              return_tensors='pt')

        # (1, sequence_length)
        # with self.tokenizer.as_target_tokenizer():
        encoded_summary = self.tokenizer(self.summary[index],
                                         padding='max_length',
                                         truncation=True,
                                         max_length=self.decoder_max_len,
                                         add_special_tokens=True,
                                         return_tensors='pt')

        model_inputs = encoded_dialogue
        model_inputs['input_ids'] = model_inputs['input_ids'].squeeze(0)
        model_inputs['attention_mask'] = model_inputs['attention_mask'].squeeze(0)
        model_inputs['labels'] = encoded_summary['input_ids']

        def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
            """
            Shift input ids one token to the right.
            """
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)

            shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
            shifted_input_ids[:, 0] = decoder_start_token_id

            if pad_token_id is None:
                raise ValueError("self.model.config.pad_token_id has to be defined.")
            # replace possible -100 values in labels by `pad_token_id`
            shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

            return shifted_input_ids

        # model_inputs['decoder_input_ids'] = shift_tokens_right(model_inputs['labels'].clone(),self.tokenizer.pad_token_id,0).squeeze(0)
        model_inputs['labels'] = model_inputs['labels'].squeeze(0)
        # print('#####')
        # print(model_inputs['decoder_input_ids'])
        # print()
        # print(model_inputs['labels'])
        # print('#####')
        # model_inputs['decoder_attention_mask'] = encoded_summary['attention_mask'].squeeze(0)

        if self.split_type == "test":
            encoded_summary2 = self.tokenizer(self.summary2[index],
                                              padding='max_length',
                                              truncation=True,
                                              max_length=self.decoder_max_len,
                                              return_tensors='pt')
            model_inputs['labels2'] = encoded_summary2['input_ids'].squeeze(0)

            encoded_summary3 = self.tokenizer(self.summary3[index],
                                              padding='max_length',
                                              truncation=True,
                                              max_length=self.decoder_max_len,
                                              return_tensors='pt')
            model_inputs['labels3'] = encoded_summary3['input_ids'].squeeze(0)

        if self.extra_supervision == True:
            if self.split_type == 'train':
                if self.sentence_transformer:
                    cur_summary_commonsense_data = self.sentence_transformer_classified_w[f"train_{self.id[index]}"]
                    summary_commonsense = ""
                    for summary_sentence_idx in range(len(cur_summary_commonsense_data.keys())):
                        commonsense = cur_summary_commonsense_data[str(summary_sentence_idx)]["out"].strip() + " ."
                        summary_commonsense += commonsense


                elif self.roberta:
                    cur_summary_commonsense_data = self.roberta_classified_w[f"train_{self.id[index]}"]
                    summary_commonsense = ""
                    for summary_sentence_idx in range(len(cur_summary_commonsense_data.keys())):
                        commonsense = cur_summary_commonsense_data[str(summary_sentence_idx)]["out"].strip() + " ."
                        summary_commonsense += commonsense



                elif self.paracomet == False:
                    summary_commonsense = ""
                    for summ in self.summary_comet_inference["train_" + self.id[index]]:
                        commonsense = summ[self.supervision_relation][0].strip() + '. '
                        commonsense = commonsense.replace('PersonX', 'Person').replace('PersonY', 'Person')
                        summary_commonsense += commonsense

                ####################################### PARACOMET START ###########################################
                else:
                    summary_commonsense = ""
                    if self.split_type == 'validation':
                        for _, summ in self.summary_comet_inference['dev' + '_' + self.id[index]].items():
                            summary_commonsense += summ[self.supervision_relation][0].strip() + '. '
                    else:
                        for _, summ in self.summary_comet_inference[self.split_type + '_' + self.id[index]].items():
                            summary_commonsense += summ[self.supervision_relation][0].strip() + '. '

                with self.tokenizer.as_target_tokenizer():
                    encoded_extra_supervision = self.tokenizer(summary_commonsense,
                                                               padding='max_length',
                                                               truncation=True,
                                                               max_length=self.decoder_max_len,
                                                               return_tensors='pt')

                model_inputs['extra_labels'] = encoded_extra_supervision['input_ids'].squeeze(0)

        return model_inputs
class DialogsumDataset_total:
    def __init__(self, encoder_max_len, decoder_max_len, tokenizer,
                 extra_context=False, extra_supervision=False, paracomet=False,
                 relation="xReason", roberta=False, supervision_relation='isAfter',
                 sentence_transformer=False):
        self.train_dataset = DialogsumDataset(encoder_max_len, decoder_max_len, 'train', tokenizer, extra_context,
                                              extra_supervision, paracomet=paracomet, relation=relation,
                                              roberta=roberta, supervision_relation=supervision_relation,
                                              sentence_transformer=sentence_transformer)
        self.eval_dataset = DialogsumDataset(encoder_max_len, decoder_max_len, 'validation', tokenizer, extra_context,
                                             extra_supervision, paracomet=paracomet, relation=relation, roberta=roberta,
                                             supervision_relation=supervision_relation,
                                             sentence_transformer=sentence_transformer)
        self.test_dataset = DialogsumDataset(encoder_max_len, decoder_max_len, 'test', tokenizer, extra_context,
                                             extra_supervision, paracomet=paracomet, relation=relation, roberta=roberta,
                                             supervision_relation=supervision_relation,
                                             sentence_transformer=sentence_transformer)
        print(self.train_dataset.data_len)

    def getTrainData(self):
        return self.train_dataset

    def getEvalData(self):
        return self.eval_dataset

    def getTestData(self):
        return self.test_dataset
class MediasumDataset(Dataset):
    pass
class MediasumDataset_total:
    pass
class TweetsummDataset(Dataset):
    pass
class TweetsummDataset_total:
    pass
class SamsumDataset_low(Dataset):
    def __init__(self, encoder_max_len, decoder_max_len, split_type, tokenizer, extra_context=False,
                 extra_supervision=False, paracomet=False, relation="xReason", supervision_relation="isAfter",
                 roberta=False):
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len
        self.split_type = split_type
        self.tokenizer = tokenizer

        self.extra_context = extra_context
        self.extra_supervision = extra_supervision
        ####### THIS WILL BE ALTERED IN THE FUTURE #######
        # self.relation = 'xReason'
        self.relation = relation
        self.paracomet = paracomet
        if self.paracomet and (self.relation[0] != "<"):
            self.relation = f"<|{self.relation}|>"

        self.supervision_relation = supervision_relation
        if self.paracomet and (self.supervision_relation[0] != "<"):
            self.supervision_relation = f"<|{self.supervision_relation}|>"

        self.roberta = roberta
        print(self.relation)
        ##################################################

        self.data = load_dataset('samsum', split=split_type)
        total = [i for i in range(len(self.data))]
        low_res = random.sample(total, len(self.data) // 10)
        whole_dialogue = self.data['dialogue']
        whole_summary = self.data['summary']
        whole_id = self.data['id']

        self.dialogue = [whole_dialogue[i] for i in low_res]
        self.summary = [whole_summary[i] for i in low_res]
        self.id = [whole_id[i] for i in low_res]

        self.nlp = spacy.load('en_core_web_sm')

        ###########################
        #   LOAD .json dataset    #
        ###########################
        if self.extra_context == True:
            if self.paracomet == False:
                with open(os.path.join(DATA_DIR, f"preprocessed/samsum/comet_{self.split_type}.json")) as f:
                    self.dialogue_comet_inference = json.load(f)

                if self.roberta:
                    with open(os.path.join(DATA_DIR,
                                           f"RobertaClassifier/samsum/roberta_classified_top1_{self.split_type}.json")) as f:
                        self.roberta_classified_z = json.load(f)

            else:
                with open(os.path.join(DATA_DIR,
                                       f"narrative_inference_demo/samsum_preprocess/collated/dialog_{self.split_type}_split5_collated.json")) as f:
                    self.dialogue_comet_inference = json.load(f)

        if self.extra_supervision == True:  # use commonsense w
            if self.split_type == 'train':
                if self.paracomet == False:  # plain COMET
                    with open(os.path.join(DATA_DIR, "preprocessed/samsum/comet_train_w.json")) as f:
                        self.summary_comet_inference = json.load(f)

                    if self.roberta:
                        with open(os.path.join(DATA_DIR,
                                               f"RobertaClassifier/samsum/roberta_classified_top1_w.json")) as f:
                            self.roberta_classified_w = json.load(f)
                else:
                    with open(os.path.join(DATA_DIR,
                                           "narrative_inference_demo/samsum_preprocess/collated/summary_train_split5_collated.json")) as f:
                        self.summary_comet_inference = json.load(f)

        self.data_len = len(self.data)

        # total = [i for i in range(self.data_len)]
        # self.low_res = random.sample(total,self.data_len//10)
        # print(self.low_res)

    def process_media_msg(self, sentence, person, commonsense):
        # print(person)
        if ('<file_photo>' in sentence) or ('<photo_file>' in sentence) or ('<file_picture>' in sentence):
            return "<I> " + person + " sent a photo. </I>" + '\n'
        elif ('<video>' in sentence) or ('<file_video>' in sentence):
            return "<I> " + person + " sent a video. </I>" + '\n'
        elif '<file_gif>' in sentence:
            return "<I> " + person + " sent a file. </I>" + '\n'
        elif ('<file_other>' in sentence) or ('<file_others>' in sentence):
            return "<I> " + person + " sent a file. </I>" + '\n'
        elif ('<link>' in sentence) or ('<file_link>' in sentence):
            return "<I> " + person + " sent a link. </I>" + '\n'
        elif '<location>' in sentence:
            return "<I> " + person + " sent a location. </I>" + '\n'
        else:
            if commonsense.strip() != 'none':
                return "<I> " + commonsense.strip() + ". </I>" + '\n'
            else:
                return ""

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        if self.extra_context == False:
            # (1, sequence_length)
            encoded_dialogue = self.tokenizer(self.dialogue[index],
                                              padding='max_length',
                                              truncation=True,
                                              max_length=self.encoder_max_len,
                                              return_tensors='pt')
        else:
            if self.paracomet == False:  # plain COMET
                try:

                    dia = self.dialogue_comet_inference[self.id[index]]

                    dialogue = ""
                    for sent_idx, sent in enumerate(dia):
                        person = sent['speaker'].replace(": ", "").replace(":", "").strip()
                        sentence = sent['sentence'].strip()
                        if self.roberta:
                            commonsense = self.roberta_classified_z[self.id[index]][str(sent_idx)]["out"]
                            # print(commonsense)
                        else:
                            # print(self.relation)
                            commonsense = sent[self.relation][0].strip()
                            # print(commonsense)

                        commonsense = commonsense.replace("PersonX", "Person").replace("PersonY", "Person")
                        dialogue += person + " said \"" + sentence + ".\"" + '\n'
                        if sent['speaker'] + sentence != commonsense:
                            # print(self.process_media_msg(sentence, person, commonsense))
                            dialogue += self.process_media_msg(sentence, person, commonsense)
                except KeyError:
                    print("key error")
                    dialogue = self.dialogue[index]



            else:  # use PARACOMETd
                try:
                    dia = self.dialogue_comet_inference[self.id[index]]
                    dialogue = ""
                    for _, sent in dia.items():
                        sentence = sent['sentence'].strip()
                        person = sentence.split()[0]
                        commonsense = sent[self.relation][0].strip()

                        dialogue += sentence + '\n'

                        if sentence != commonsense:
                            dialogue += self.process_media_msg(sentence, person, commonsense)
                except KeyError:  # when an error occurred while processing commonsense, just give plain utterance as output
                    print("key error")
                    # print(index)
                    # print(self.id[index])
                    # print(self.dialogue_comet_inference.keys())
                    dialogue = self.dialogue[index]

            encoded_dialogue = self.tokenizer(dialogue,
                                              padding='max_length',
                                              truncation=True,
                                              max_length=self.encoder_max_len,
                                              return_tensors='pt')

        # (1, sequence_length)
        with self.tokenizer.as_target_tokenizer():
            encoded_summary = self.tokenizer(self.summary[index],
                                             padding='max_length',
                                             truncation=True,
                                             max_length=self.decoder_max_len,
                                             return_tensors='pt')

        model_inputs = encoded_dialogue
        model_inputs['input_ids'] = model_inputs['input_ids'].squeeze(0)
        model_inputs['attention_mask'] = model_inputs['attention_mask'].squeeze(0)
        model_inputs['labels'] = encoded_summary['input_ids'].squeeze(0)

        if self.extra_supervision == True:
            if self.split_type == 'train':
                def split_sentences(text, speaker):
                    doc = self.nlp(text)
                    sents = [speaker.replace(":", "") + ' said "' + sent.text + '"' for sent in doc.sents]
                    return sents

                if self.paracomet == False:  # plain COMET
                    summary_commonsense = ""
                    if self.roberta:
                        for _, summ in self.roberta_classified_w[self.id[index]].items():
                            commonsense = summ["out"].strip() + ". "
                            commonsense = commonsense.replace("PersonX", "Person").replace("PersonY", "Person")
                            summary_commonsense += commonsense

                    else:
                        for summ in self.summary_comet_inference[self.id[index]]:
                            commonsense = summ[self.supervision_relation][0].strip() + '. '
                            commonsense = commonsense.replace("PersonX", "Person").replace("PersonY", "Person")
                            summary_commonsense += commonsense

                    with self.tokenizer.as_target_tokenizer():
                        encoded_extra_supervision = self.tokenizer(summary_commonsense,
                                                                   padding='max_length',
                                                                   truncation=True,
                                                                   max_length=self.decoder_max_len,
                                                                   return_tensors='pt')

                    model_inputs['extra_labels'] = encoded_extra_supervision['input_ids'].squeeze(0)
                else:
                    if index == 6054:
                        summary_commonsense = "problem with presentation."
                    else:
                        summary_commonsense = ""
                        for _, summ in self.summary_comet_inference[self.id[index]].items():
                            try:
                                summary_commonsense += summ[self.supervision_relation][0].strip() + '. '
                            except KeyError:
                                print("key error in supervision")
                                summary_commonsense = ""

                    with self.tokenizer.as_target_tokenizer():
                        encoded_extra_supervision = self.tokenizer(summary_commonsense,
                                                                   padding='max_length',
                                                                   truncation=True,
                                                                   max_length=self.decoder_max_len,
                                                                   return_tensors='pt')

                    model_inputs['extra_labels'] = encoded_extra_supervision['input_ids'].squeeze(0)
                # print(summary_commonsense)

        return model_inputs
class SamsumDataset_low_total:
    def __init__(self, encoder_max_len, decoder_max_len, tokenizer, extra_context=False, extra_supervision=False,
                 paracomet=False, relation="xReason", supervision_relation='isAfter', roberta=False):
        self.train_dataset = SamsumDataset_low(encoder_max_len, decoder_max_len, 'train', tokenizer,
                                               extra_context=extra_context, extra_supervision=extra_supervision,
                                               paracomet=paracomet, relation=relation,
                                               supervision_relation=supervision_relation, roberta=roberta)
        self.eval_dataset = SamsumDataset_low(encoder_max_len, decoder_max_len, 'validation', tokenizer,
                                              extra_context=extra_context, extra_supervision=extra_supervision,
                                              paracomet=paracomet, relation=relation,
                                              supervision_relation=supervision_relation, roberta=roberta)
        self.test_dataset = SamsumDataset_low(encoder_max_len, decoder_max_len, 'test', tokenizer,
                                              extra_context=extra_context, extra_supervision=extra_supervision,
                                              paracomet=paracomet, relation=relation,
                                              supervision_relation=supervision_relation, roberta=roberta)

    def getTrainData(self):
        return self.train_dataset

    def getEvalData(self):
        return self.eval_dataset

    def getTestData(self):
        return self.test_dataset


# Set Argument Parser
parser = argparse.ArgumentParser()
# Training hyperparameters
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--train_batch_size', type=int, default=16)
#parser.add_argument('--display_step',type=int, default=2000)
parser.add_argument('--val_batch_size',type=int, default=4)
parser.add_argument('--test_batch_size',type=int,default=1)
# Model hyperparameters
parser.add_argument('--model_name',type=str, default='facebook/bart-large')
# Optimizer hyperparameters
parser.add_argument('--init_lr',type=float, default=3e-6)
parser.add_argument('--warm_up',type=int, default=600)
parser.add_argument('--weight_decay',type=float, default=1e-2)
parser.add_argument('--decay_epoch',type=int, default=0)
parser.add_argument('--adam_beta1',type=float, default=0.9)
parser.add_argument('--adam_beta2',type=float, default=0.999)
parser.add_argument('--adam_eps',type=float, default=1e-12)
parser.add_argument('--dropout_rate',type=float, default=0.1)
# Tokenizer hyperparameters
parser.add_argument('--encoder_max_len', type=int, default=1024)
parser.add_argument('--decoder_max_len', type=int, default=100)
parser.add_argument('--vocab_size',type=int, default=51201)
parser.add_argument('--eos_idx',type=int, default=51200)
parser.add_argument('--tokenizer_name',type=str, default='RobertaTokenizer')
# Checkpoint directory hyperparameters
parser.add_argument('--pretrained_weight_path',type=str, default='pretrained_weights')
parser.add_argument('--finetune_weight_path', type=str, default="./context_BART_weights_Samsum_5epoch")
parser.add_argument('--best_finetune_weight_path',type=str, default='context_final_BART_weights_Samsum_5epoch')
# Dataset hyperparameters
parser.add_argument('--dataset_name',type=str, default='samsum')
parser.add_argument('--use_paracomet',type=bool,default=False)
parser.add_argument('--use_roberta',type=bool,default=False)
parser.add_argument('--use_sentence_transformer',type=bool,default=False)
parser.add_argument('--dataset_directory',type=str, default='./data')
parser.add_argument('--test_output_file_name',type=str, default='samsum_context_trial2.txt')
parser.add_argument('--relation',type=str,default="xReason")
parser.add_argument('--supervision_relation',type=str,default='isAfter')
args = parser.parse_args()


# Set GPU
print('######################################################################')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
print(torch.cuda.get_device_name())
print('######################################################################')


# Start WANDB Log (Set Logging API)
wandb.init()
if args.use_paracomet:
    cs = "para"
    if args.use_roberta:
        cs+= "_roberta"
else:
    cs = "comet"
    if args.use_roberta:
        cs+= "_roberta"

if args.use_sentence_transformer:
    if args.use_paracomet:
        cs = "paracomet_sentence_transformer"
    else:
        cs = "comet_sentence_transformer"

print("#"*50)
print(cs)
print("#"*50)
wandb.run.name = f"context_{args.dataset_name}_{args.relation}_{cs}_lr{str(args.init_lr)}"


# Define Global Values
model_checkpoint_list = [
    "facebook/bart-large", 
    "facebook/bart-large-xsum",
    "google/pegasus-large",
    "google/peagsus-xsum",
    "google/t5-large-lm-adapt", 
    "google/t5-v1_1-large"
]
tokenizer_list = {
    "facebook/bart-large":"RobertaTokenizer",
    "facebook/bart-large-xsum":"RobertaTokenizer",
    "google/pegasus-large":"PegasusTokenizer",
    "google/peagsus-xsum":"PegasusTokenizer",
    "google/t5-large-lm-adapt":"T5Tokenizer", 
    "google/t5-v1_1-large":"T5Tokenizer"
}
max_len_list ={
    "facebook/bart-large":1024,
    "facebook/bart-large-xsum":1024,
    "google/pegasus-large":1024,
    "google/peagsus-xsum":512,
    "google/t5-large-lm-adapt":512, 
    "google/t5-v1_1-large":512
}
vocab_size_list={
    "facebook/bart-large":50265,
    "facebook/bart-large-xsum":50264,
    "google/pegasus-large":96103,
    "google/peagsus-xsum":96103,
    "google/t5-large-lm-adapt":32128, 
    "google/t5-v1_1-large":32128
}
dataset_list = [
    "samsum","dialogsum"
]

# The model names is passed as argument of the program (in our example, facebook/bart-large-xsum)
# Refine arguments based on global values
if args.model_name not in model_checkpoint_list:
    assert "Your Model checkpoint name is not valid"
args.tokenizer_name = tokenizer_list[args.model_name]
#args.max_len = max_len_list[args.model_name]
#args.max_len = 1024
args.vocab_size = vocab_size_list[args.model_name]
if args.dataset_name not in dataset_list:
    assert "Your Dataset name is not valid"


# Set metric
metric = load_metric("rouge")
#metric = load_metric("../utils/rouge.py")

# Load Tokenizer associated to the model
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# Add special token 
special_tokens_dict = {'additional_special_tokens':['<I>','</I>']}
tokenizer.add_special_tokens(special_tokens_dict)


# Set dataset
if args.dataset_name=='samsum':
    total_dataset = SamsumDataset_total(args.encoder_max_len,args.decoder_max_len,tokenizer,extra_context=True,paracomet=args.use_paracomet,relation=args.relation,supervision_relation=args.supervision_relation,roberta=args.use_roberta, sentence_transformer=args.use_sentence_transformer)
    train_dataset = total_dataset.getTrainData()
    eval_dataset = total_dataset.getEvalData()
    test_dataset = total_dataset.getTestData()
elif args.dataset_name=='dialogsum':
    total_dataset = DialogsumDataset_total(args.encoder_max_len,args.decoder_max_len,tokenizer,extra_context=True,paracomet=args.use_paracomet,relation=args.relation,supervision_relation=args.supervision_relation, sentence_transformer=args.use_sentence_transformer, roberta=args.use_roberta)
    train_dataset = total_dataset.getTrainData()
    eval_dataset = total_dataset.getEvalData()
    test_dataset = total_dataset.getTestData()
print('######################################################################')
print('Training Dataset Size is : ')
print(len(train_dataset))
print('Validation Dataset Size is : ')
print(len(eval_dataset))
print('Test Dataset Size is : ')
print(len(test_dataset))
print('######################################################################')


# Loading checkpoint of model
config = AutoConfig.from_pretrained(args.model_name)
finetune_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
print('######################################################################')
print("Number of Model Parameters are : ",finetune_model.num_parameters())
print('######################################################################')


# Set extra Configuration for Finetuning on Summarization Dataset
finetune_model.resize_token_embeddings(len(tokenizer))
finetune_model.gradient_checkpointing_enable()
finetune_model = finetune_model.to(device)


# Set Training Arguments (& Connect to WANDB)
finetune_args = Seq2SeqTrainingArguments(
    output_dir = args.finetune_weight_path,
    overwrite_output_dir = True,
    do_train=True,
    do_eval=True,
    do_predict=True,
    evaluation_strategy='epoch',
    logging_strategy="epoch",
    save_strategy= "epoch",
    # eval_steps=1,
    # logging_steps=1,
    # save_steps=1,
    per_device_train_batch_size = args.train_batch_size,
    per_device_eval_batch_size = args.val_batch_size,
    learning_rate=args.init_lr,
    weight_decay=args.weight_decay,
    adam_beta1=args.adam_beta1,
    adam_beta2=args.adam_beta2,
    adam_epsilon=args.adam_eps,
    num_train_epochs=args.epoch,
    max_grad_norm=0.1,
    #label_smoothing_factor=0.1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    # max_steps= ,
    lr_scheduler_type='polynomial',
    #warmup_ratio= ,
    warmup_steps= args.warm_up,
    save_total_limit=1,
    fp16=True,
    seed = 516,
    load_best_model_at_end=True,
    predict_with_generate=True,
    prediction_loss_only=False,
    generation_max_length=100,
    generation_num_beams=5,
    metric_for_best_model='eval_rouge1',
    greater_is_better=True,
    report_to = 'wandb',
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

def preprocess_logits_for_metrics(logits, labels):
    print(logits.size())
    logits,_ = logits
    logits_device = logits.device
    logits_reduced = np.argmax(logits.cpu(),axis=-1)
    logits_reduced = logits_reduced.to(logits_device)

    return logits_reduced

finetune_trainer = Seq2SeqTrainer(
    model = finetune_model,
    args = finetune_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    tokenizer = tokenizer,
    compute_metrics=compute_metrics,
    # preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

# Run Training (Finetuning)
finetune_trainer.train()


# Save final weights
finetune_trainer.save_model(args.best_finetune_weight_path)

"""
# Run Evaluation on Test Data
results = finetune_trainer.predict(
    test_dataset=test_dataset,
    max_length= 60,
    num_beams = 5   #1,3,5,10
)
predictions, labels, metrics = results
print('######################################################################')
print("Final Rouge Results are : ",metrics)
print('######################################################################')


# Write evaluation predictions on txt file
decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after e ach sentence
decoded_preds = [" ".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
decoded_labels = [" ".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]


# output summaries on test set
with open(args.test_output_file_name,"w") as f: 
    f.write(metrics)
    for i in decoded_preds:
        f.write(i.replace("\n","")+"\n")
"""
# END WANDB log
wandb.finish()

