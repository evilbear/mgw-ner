import argparse, logging, os
import numpy as np

def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger

def get_vocab(filename):
    vocab = dict()
    with open(filename) as f:
        for idx, word in enumerate(f):
            word = word.strip('\n')
            vocab[word] = idx
    return vocab    

def deal_word(word, vocab_words, vocab_chars, use_chars):
    if vocab_chars is not None and use_chars == True:
        char_ids = []
        for char in word:
            if char.isdigit():
                char_ids.append(vocab_chars["$NUM$"])
            elif char in vocab_chars:
                char_ids.append(vocab_chars[char])
            else:
                char_ids.append(vocab_chars["$UNK$"])
    if word.isdigit():
        word = vocab_words["$NUM$"]
    elif word in vocab_words:
        word = vocab_words[word]
    else:
        word = vocab_words["$UNK$"]
    if vocab_chars is not None and use_chars == True:
        return char_ids, word
    else:
        return word

def deal_tags(tag, vocab_tags):  
    tag = vocab_tags[tag]
    return tag

def get_corpus(filename, vocab_words, vocab_tags, vocab_chars=None, use_chars=False):
    with open(filename) as f:
        words, tags, corpus= [], [], []
        for line in f:
            line = line.strip('\n')
            if len(line) == 0:
                corpus.append([words, tags])
                words, tags = [], []
            else:
                ls = line.split(' ')
                word, tag = ls[0],ls[-1]
                word = deal_word(word, vocab_words, vocab_chars, use_chars)
                tag = deal_tags(tag, vocab_tags)
                words += [word]
                tags += [tag]
    return corpus  

def batch_pre(data, batch_size):
    seqs, labels, corpus_callback = [], [], []
    for (seq, tag) in data:
        if (len(seqs) == batch_size):
            corpus_callback.append([seqs, labels])
            seqs, labels = [], []
        try:
            if type(seq[0]) == tuple:
                seq = zip(*seq)
        except IndexError:
            print(seq)
            print(tag)
        seqs.append(seq)
        labels.append(tag)
    if len(seqs) != 0:
        corpus_callback.append([seqs, labels])
    return corpus_callback

def _pad_sequences(seqs, pad_tok, max_length):
    sequence_padded, sequence_length = [], []
    for seq in seqs:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        sequence_length.append(min(len(seq), max_length))
    return sequence_padded, sequence_length

def pad_sequences(seqs, nlevels=1):
    if nlevels == 1:
        max_length = max(map(lambda x: len(x), seqs))
        sequence_padded, sequence_length = _pad_sequences(seqs, 0, max_length)
    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in seqs])
        sequence_padded, sequence_length = [], []
        for seq in seqs:
            sp, sl = _pad_sequences(seq, 0, max_length_word)
            sequence_padded.append(sp)
            sequence_length.append(sl)
        max_length_sentence = max(map(lambda x : len(x), seqs))
        sequence_padded, _ = _pad_sequences(sequence_padded, [0]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    return sequence_padded, sequence_length

def get_tag(str1):
    if str1 == 1:
        return('B-LOC')
    elif str1 == 2:
        return('I-LOC')
    elif str1 == 3:
        return('B-ORG')
    elif str1 == 4:
        return('I-ORG')
    elif str1 == 5:
        return('B-PER')
    elif str1 == 6:
        return('I-PER')
    else:
        return('O')

def get_lm_embeddings(mode, parm, step):
    embedding_save_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/'+ mode+'_'+ parm+'/'+ mode+'_'+str(step)+'.npz'
    embedding_txt_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/lm/'+ mode+'/'+mode+'_'+str(step)+'.txt'
    line_list = []
    with open(embedding_txt_path, 'r') as f:
        for i in f:
            i = i.strip().split(' ')
            line_list += [len(i)]
    with np.load(embedding_save_path) as data:
        embeddings = data["embeddings"]
    embeddings = embeddings.tolist()
    lm_embeddings = []
    max_length = max(line_list)
    num = 0
    for i in line_list:
        pad_len = max(max_length - i, 0)
        lm_embeddings += [embeddings[num:num+i] + [[0.]*300]*pad_len]
        num += i + 1
    lm_embeddings = np.array(lm_embeddings, dtype='float32')
    return lm_embeddings
    
