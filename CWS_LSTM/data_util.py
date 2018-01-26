import numpy as np
import re
import sys
from tqdm import tqdm
import pandas as pd
from itertools import chain
import pickle


class BatchLoader(object):
    def __init__(self, X, y, shuffle=False):
        self._X = X
        self._y = y
        self._shuffle = shuffle
        self._cont_in_epoch = 0
        self._epoch_completed = 0
        self._sample_number = self._X.shape[0]
        if self._shuffle:
            index = np.random.permutation(self._sample_number)
            self._X = self._X[index]
            self._y = self._y[index]
            
    @property
    def X(self):
        return self._X
    
    @property
    def y(self):
        return self._y
    
    @property
    def sample_number(self):
        return self._sample_number

    @property
    def epoches_completed(self):
        return self._epoch_completed

    @property
    def cont_in_epoch(self):
        return self._cont_in_epoch

    def next_batch(self, batch_size=64):
        start = self._cont_in_epoch
        end = self._cont_in_epoch + batch_size
        if end > self._sample_number:
            self._epoch_completed += 1
            self._cont_in_epoch = 0
            start = 0
            end = batch_size
            if self._shuffle:
                index = np.random.permutation(range(self._sample_number))
                self._X = self._X[index]
                self._y = self._y[index]
        self._cont_in_epoch = end
        return self._X[start:end], self._y[start:end]


def compute_trans_matrix(labels):
    A = {
        'sb': 0,
        'ss': 0,
        'be': 0,
        'bm': 0,
        'me': 0,
        'mm': 0,
        'eb': 0,
        'es': 0
    }
    # zy 表示转移概率矩阵
    zy = dict()
    for label in labels:
        for t in range(len(label) - 1):
            key = label[t] + label[t + 1]
            A[key] += 1.0

    zy['sb'] = A['sb'] / (A['sb'] + A['ss'])
    zy['ss'] = 1.0 - zy['sb']
    zy['be'] = A['be'] / (A['be'] + A['bm'])
    zy['bm'] = 1.0 - zy['be']
    zy['me'] = A['me'] / (A['me'] + A['mm'])
    zy['mm'] = 1.0 - zy['me']
    zy['eb'] = A['eb'] / (A['eb'] + A['es'])
    zy['es'] = 1.0 - zy['eb']
    return zy


def process_raw_txt(file_name, patch_len=32):
    with open(file_name, 'rb') as fp:
        texts = fp.read().decode('gbk')
    sentences = texts.split('\n')
    texts = ''.join(sentences)
    sentences = re.split('[，。！？、‘’“”]/[bems]', texts)
    print('Total sentence number {}'.format(len(sentences)))
    print('Separating words and tags...')
    words = list()
    tags = list()
    for sentence in tqdm(sentences):
        word, tag = sep_word_tag(sentence)
        if word is not None:
            words.append(word)
            tags.append(tag)

    trans_prob = compute_trans_matrix(tags)

    print('Saving to pandas DataFrame...')
    df = pd.DataFrame({'words': words, 'tags': tags}, index=range(len(words)))
    df['sentenceLen'] = df['words'].apply(lambda sentence: len(sentence))

    all_words = list(chain(*df['words'].values))
    all_words = pd.Series(all_words)
    all_words_cont = all_words.value_counts()
    all_words_set = all_words_cont.index
    print('Training set have {} different words'.format(len(all_words_set)))
    words_id = range(1, len(all_words_set) + 1)
    tags_set = ['x', 's', 'b', 'm', 'e']
    tags_id = range(len(tags_set))
    word2id = pd.Series(words_id, index=all_words_set)
    id2word = pd.Series(all_words_set, index=words_id)
    tag2id = pd.Series(tags_id, index=tags_set)
    id2tag = pd.Series(tags_set, index=tags_id)
    print('Converting words to ids, this may take several minutes, please wait...')
    df['X'] = df['words'].apply(wordtag2Xy, args=(word2id, patch_len,))
    df['y'] = df['tags'].apply(wordtag2Xy, args=(tag2id, patch_len,))
    X = np.asarray(list(df['X'].values))
    y = np.asarray(list(df['y'].values))

    print('Saving to pickle...')
    with open('{}.pkl'.format(file_name.split('.')[0]), 'wb') as pk:
        pickle.dump(X, pk)
        pickle.dump(y, pk)
        pickle.dump(word2id, pk)
        pickle.dump(id2word, pk)
        pickle.dump(tag2id, pk)
        pickle.dump(id2tag, pk)
        pickle.dump(trans_prob, pk)
    print('Finished processing raw data!')


def wordtag2Xy(words, wordtag2id, patch_len=32):
    ids = list(wordtag2id[words])
    if len(ids) >= patch_len:
        return ids[:patch_len]
    else:
        ids.extend([0] * (patch_len - len(ids)))
        return ids


def sep_word_tag(sentence):
    sep = re.findall('(.)/(.)', sentence)
    if sep:
        sep = np.asarray(sep)
        word = sep[:, 0]
        tag = sep[:, 1]
    else:
        word = tag = None
    return word, tag


def clean(s):
    if u'“/s' not in s:
        return s.replace(u' ”/s', '')
    elif u'”/s' not in s:
        return s.replace(u'“/s ', '')
    elif u'‘/s' not in s:
        return s.replace(u' ’/s', '')
    elif u'’/s' not in s:
        return s.replace(u'‘/s ', '')
    else:
        return s


if __name__ == '__main__':
    defaultFile = 'data/msr_train.txt'
    if len(sys.argv) == 2:
        trainData = sys.argv[1]
    else:
        print('Using default training data: msr_train.txt ...')
        trainData = defaultFile
    process_raw_txt(trainData)
