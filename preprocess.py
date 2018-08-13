import re
import string
import os
from unicodedata import normalize
import numpy as np

import torch
from torchtext import data, datasets

from utils import *


class DataProcessor(object):
    def __init__(self, src_lng, trg_lng):
        self.src_field, self.trg_field = self.generate_fields()
        self.src_lng = src_lng
        self.trg_lng = trg_lng

    @staticmethod
    def load_document(filename):
        with open(filename, mode='rt', encoding='utf-8') as file:
            text = file.read()
            lines = text.strip().split('\n')
            pairs = [line.split('\t') for line in lines]

            return pairs[0:1000]

    @staticmethod
    def clean_pairs(lines):
        cleaned = list()
        # prepare regex for char filtering
        re_print = re.compile('[^%s]' % re.escape(string.printable))
        # prepare translation table for removing punctuation
        table = str.maketrans('', '', string.punctuation)
        for pair in lines:
            clean_pair = list()
            for line in pair:
                # normalize unicode characters
                line = normalize('NFD', line).encode('ascii', 'ignore')
                line = line.decode('UTF-8')
                # tokenize on white space
                line = line.split()
                # convert to lowercase
                line = [word.lower() for word in line]
                # remove punctuation from each token
                line = [word.translate(table) for word in line]
                # remove non-printable chars form each token
                line = [re_print.sub('', w) for w in line]
                # remove tokens with numbers in them
                line = [word for word in line if word.isalpha()]
                # store as string
                clean_pair.append(' '.join(line))
            cleaned.append(clean_pair)
        return np.array(cleaned)

    @staticmethod
    def write_files(data, prefix, src_lang='en', dst_lang='fr'):
        with open(f'out/{prefix}.{src_lang}', 'wt') as f:
            f.writelines([line + '\n' for line in data[:, 0]])
        with open(f'out/{prefix}.{dst_lang}', 'wt') as f:
            f.writelines([line + '\n' for line in data[:, 1]])

    def load_data(self, input_filename, output_filename):
        out_file = f'out/{output_filename}'

        if not os.path.exists('out'):
            os.mkdir('out')

        if not os.path.exists(out_file):
            print("Preprocessing train dataset...")

            input_filepath = os.path.join('data', input_filename)
            pairs = DataProcessor.load_document(input_filepath)
            sentences = DataProcessor.clean_pairs(pairs)

            np.random.shuffle(sentences)

            m = len(sentences)

            train, test = sentences[:int(m * 0.9)], sentences[int(m * 0.9):]
            valid, test = test[:len(test) // 2], test[len(test) // 2:]

            print("Saving dataset...")

            data = {'train': train, 'valid': valid, 'test': test}

            torch.save(data, out_file)

            for d, p in [(train, 'train'), (valid, 'val'), (test, 'test')]:
                DataProcessor.write_files(d, p)

        print('Loading datasets...')
        train_dataset, valid_dataset, test_dataset = datasets.TranslationDataset.splits(path='out',
                                                                                        exts=('.' + self.src_lng,
                                                                                              '.' + self.trg_lng),
                                                                                        fields=(self.src_field, self.trg_field))

        self.src_field.build_vocab(train_dataset, max_size=30000)
        self.trg_field.build_vocab(train_dataset, max_size=30000)

        src_vocab = self.src_field.vocab.stoi
        trg_vocab = self.trg_field.vocab.stoi

        # Define index to string vocabs
        src_inv_vocab = self.src_field.vocab.itos
        trg_inv_vocab = self.trg_field.vocab.itos

        vocabs = {'src_vocab': src_vocab, 'trg_vocab': trg_vocab,
                  'src_inv_vocab': src_inv_vocab, 'trg_inv_vocab': trg_inv_vocab}

        return train_dataset, valid_dataset, test_dataset, vocabs

    def generate_fields(self):
        src_field = data.Field(tokenize=data.get_tokenizer('spacy'),
                               init_token=SOS_WORD,
                               eos_token=EOS_WORD,
                               pad_token=PAD_WORD,
                               include_lengths=True,
                               batch_first=True)
        trg_field = data.Field(tokenize=data.get_tokenizer('spacy'),
                               init_token=SOS_WORD,
                               eos_token=EOS_WORD,
                               pad_token=PAD_WORD,
                               include_lengths=True,
                               batch_first=True)
        return src_field, trg_field
