import os, argparse
import numpy as np
from ner_model import BILSTM_CRF
from ner_utils import str2bool, get_logger, get_vocab, get_corpus

# hyperparameters
parser = argparse.ArgumentParser(description='MGW NER task')
parser.add_argument('--train_data', type=str, default='train.txt', help='train data source')
parser.add_argument('--dev_data', type=str, default='dev.txt', help='dev data source')
parser.add_argument('--test_data', type=str, default='test.txt', help='test data source')
parser.add_argument('--output_path', type=str, default='/result/', help='output path')
parser.add_argument('--batch_size', type=int, default=20, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=100, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--use_chars', type=str2bool, default=True, help='whether use char embedding')
parser.add_argument('--use_model', type=str, default='lms_att', help='baseline, lms_concat, lms_att, lms_concat_blstm or lms_att_blstm')
parser.add_argument('--mode', type=str, default='train', help='train or test')
args = parser.parse_args()


#get word embeddings
embedding_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/vectors.npz'
with np.load(embedding_path) as data:
    embeddings = data["embeddings"]
embeddings = np.array(embeddings, dtype='float32')

#get words tags chars vocab
words_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/words.txt'
tags_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/tags.txt'
chars_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/chars.txt'
vocab_words = get_vocab(words_path)
vocab_tags = get_vocab(tags_path)
vocab_chars = get_vocab(chars_path)

# paths setting
output_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + args.output_path
if not os.path.exists(output_path): os.makedirs(output_path)
log_path = output_path + 'log.txt'
logger = get_logger(log_path)
logger.info(str(args))

#Model controlled by parameters
use_lm, use_att, use_extra_blstm = False, False, False
if args.use_model == 'lms_concat':
    use_lm = True
elif args.use_model == 'lms_att':
    use_lm, use_att = True, True
elif args.use_model == 'lms_concat_blstm':
    use_lm, use_extra_blstm = True, True
elif args.use_model == 'lms_att_blstm':
    use_lm, use_att, use_extra_blstm = True, True, True
else:
    pass

#training model
if args.mode == 'train':
    train_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/' + args.train_data
    dev_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/' + args.dev_data
    train_data = get_corpus(train_path, vocab_words, vocab_tags, vocab_chars, args.use_chars)
    dev_data = get_corpus(dev_path, vocab_words, vocab_tags, vocab_chars, args.use_chars)
    model = BILSTM_CRF(batch_size=args.batch_size, epoch_num=args.epoch, hidden_dim=args.hidden_dim, embeddings=embeddings,
                dropout_keep=args.dropout, optimizer=args.optimizer, lr=args.lr, clip_grad=args.clip, vocab_words=vocab_words,
                vocab_tags=vocab_tags, vocab_chars=vocab_chars, output_path=output_path, logger=logger,update_embedding=args.update_embedding,
                use_chars=args.use_chars, use_lm=use_lm, use_att=use_att, use_extra_blstm=use_extra_blstm)
    model.build_graph()
    model.train(train_data, dev_data)
elif args.mode == 'test':
    test_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/' + args.test_data
    test_data = get_corpus(test_path, vocab_words, vocab_tags, vocab_chars, args.use_chars)
    model = BILSTM_CRF(batch_size=args.batch_size, epoch_num=args.epoch, hidden_dim=args.hidden_dim, embeddings=embeddings,
                dropout_keep=args.dropout, optimizer=args.optimizer, lr=args.lr, clip_grad=args.clip, vocab_words=vocab_words,
                vocab_tags=vocab_tags, vocab_chars=vocab_chars, output_path=output_path, logger=logger,update_embedding=args.update_embedding,
                use_chars=args.use_chars, use_lm=use_lm, use_att=use_att, use_extra_blstm=use_extra_blstm)
    model.build_graph()
    model.restore_session(output_path)
    model.test(test_data)
else:
    pass