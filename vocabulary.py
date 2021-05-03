import numpy as np
import torch


def pad_list(l, max_len, pad_value=None):
    l += [pad_value] * (max_len - len(l))
    return l


class Vocabulary(object):
    def __init__(self, start_word=None, end_word=None, unknown_word=None, pad_word=None) -> None:
        super().__init__()
        self._word_to_index = {}
        self._index_to_word = []
        self._special_words = {}  # {type: (word, index)}
        self.add_special_word(pad_word or '<pad>', 'P')
        self.add_special_word(start_word or '<start>', 'S')
        self.add_special_word(end_word or '<end>', 'E')
        self.add_special_word(unknown_word or '<unk>', 'U')

    def to_dataframe(self):
        import pandas as pd
        l = []
        specials = {v[1]: k for k, v in self._special_words.items()}
        for i, w in enumerate(self._index_to_word):
            if i in specials:
                l.append((w, specials[i]))
            else:
                l.append((w, ''))
        return pd.DataFrame(l, columns=['Word', 'SpecialType'])

    def from_dataframe(self, df):
        self._word_to_index = {}
        self._index_to_word = []
        self._special_words = {} 
        for row in df.itertuples():
            if row.SpecialType:
                self.add_special_word(row.Word, row.SpecialType)
            else:
                self.add_word(row.Word)

    def add_special_word(self, word, special_type):
        if word in self._word_to_index:
            return None
        ix = self.add_word(word)
        self._special_words[special_type] = word, ix
        return ix

    def get_all_words(self):
        return self._index_to_word

    def add_word(self, word):
        if word is None:
            raise Exception('word cannot be null')
        idx = self._word_to_index.get(word, None)
        if idx is None:
            idx = len(self._index_to_word)
            self._index_to_word.append(word)
            self._word_to_index[word] = idx
        return idx

    def decode_word(self, index):
        if index < len(self._index_to_word):
            return self._index_to_word[index]
        return self.get_unk_word(False)

    def encode_word(self, word):
        return self._word_to_index.get(word, None) or self.get_unk_word()

    def get_start_word(self, index=True):
        return self._special_words['S'][1 if index else 0]

    def get_end_word(self, index=True):
        return self._special_words['E'][1 if index else 0]

    def get_unk_word(self, index=True):
        return self._special_words['U'][1 if index else 0]

    def get_pad_word(self, index=True):
        return self._special_words['P'][1 if index else 0]

    def encode_sentence(self, sentence, min_len=None):
        start_ix = self.get_start_word()
        pad_ix = self.get_pad_word()
        end_ix = self.get_end_word()
        ret = [start_ix]
        ret.extend(map(self.encode_word, sentence))
        ret.append(end_ix)
        if min_len:
            ret = pad_list(ret, min_len + 2, pad_ix)
        return ret

    def decode_sentence(self, indexes):
        return [self.decode_word(idx) for idx in indexes]

    def add_padding(self, indexes, target_len):
        return pad_list(list(indexes), target_len, self.get_pad_word())

    def trim_padding(self, indexes):
        a = np.array(indexes)
        pad_index = self.get_pad_word()
        return list(a[:len(a) - (np.flip(a) == pad_index).argmin()])
