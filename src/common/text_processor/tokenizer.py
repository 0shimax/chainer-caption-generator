from gensim import corpora
import os
import h5py
from chainer import serializers
import numpy as np


class Tokenizer(object):
    def __init__(self, args):
        if args.lang=='ja':
            self.parser = self.parse_japanese_words
        elif args.lang=='en':
            self.parser = self.parse_english_words
        self.args = args
        self.tokens = None

    def parse_japanese_words(self, text):
        import MeCab
        tagger_owakati = MeCab.Tagger(self.args.tagger)
        str_owakati = tagger_owakati.parse(text)
        tokens = str_owakati.rstrip().split(' ')
        tokens += ['<EOS>']+['<ignore>']*(self.args.max_len-len(tokens)-1)
        return np.array(tokens)

    def parse_english_words(self, text):
        tokens = text.rstrip().split(' ')
        tokens += ['<EOS>']+['<ignore>']*(self.args.max_len-len(tokens)-1)
        return  np.array(tokens)

    def parse(self, texts_path):
        self.tokens = []
        with open(texts_path, 'r') as f:
            for text in f:
                words = self.parser(text.rstrip())
                self.tokens.append(words)

    def token2id(self):
        # tokens: [[word11, word12,...,EOS],..., [word n1, word n2,...,EOS]]
        if os.path.exists(self.args.tokens_path):
            self.__load_token()
            dictionary = self.__loda_dictionary()
        else:
            if os.path.exists(self.args.dic_load_path):
                self.parse(self.args.texts_path)
                dictionary = self.__loda_dictionary()
            else:
                self.parse(self.args.texts_path)
                dictionary = corpora.Dictionary(self.tokens)
                self.__save_dictionary(dictionary)
            self.__save_token()

        dic_ids = dictionary.token2id
        token_ids = []
        for words in self.tokens:
            token_ids.append([dic_ids[word] for word in words])
        return np.array(token_ids, dtype=np.int32)

    def __loda_dictionary(self):
        print('loading dictionary from:', self.args.dic_load_path)
        return corpora.Dictionary.load(self.args.dic_load_path)

    def __load_token(self):
        print('loading tokens from:', self.args.tokens_path)
        infnpz = np.load(self.args.tokens_path)
        self.tokens = infnpz['tokens']

    def __save_dictionary(self, dictionary):
        if self.args.dic_save_path is not None:
            print('save dictionary from:', self.args.dic_save_path)
            dictionary.token2id['<ignore>'] = -1
            dictionary.save(self.args.dic_save_path)

    def __save_token(self):
        print('saving tokens from:', self.args.tokens_path)
        np.savez(self.args.tokens_path, tokens=self.tokens)
