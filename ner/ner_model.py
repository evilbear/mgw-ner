import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode
from attention import attention
from ner_utils import batch_pre, pad_sequences, get_tag, get_lm_embeddings

class BILSTM_CRF(object):
    def __init__(self, batch_size, epoch_num, hidden_dim, embeddings, dropout_keep, optimizer, lr, clip_grad, vocab_words,
                                                    vocab_tags, vocab_chars, output_path, logger, update_embedding=True,
                                                    use_chars=True,use_lm=False, use_att=False, use_extra_blstm=False):
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.hidden_dim = hidden_dim
        self.embeddings = embeddings
        self.dropout_keep = dropout_keep
        self.optimizer = optimizer
        self.lr = lr
        self.clip_grad = clip_grad
        self.vocab_words = vocab_words
        self.vocab_tags = vocab_tags
        self.vocab_chars = vocab_chars
        self.num_words = len(vocab_words)
        self.num_chars = len(vocab_chars)
        self.num_tags = len(vocab_tags)
        self.output_path = output_path
        self.logger = logger
        self.update_embedding = update_embedding
        self.use_chars = use_chars
        self.use_lm = use_lm
        self.use_att = use_att
        self.use_extra_blstm = use_extra_blstm
        
    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()
        
    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name='word_ids')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None], name="char_ids")
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None], name="word_lengths")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name='dropout')
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name='lr')
        self.fw_lm_embeddings = tf.placeholder(tf.float32, shape=[None, None, self.hidden_dim], name="fw_lm_embeddings")
        self.bw_lm_embeddings = tf.placeholder(tf.float32, shape=[None, None, self.hidden_dim], name="bw_lm_embeddings")
        
    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            self._word_embeddings = tf.Variable(self.embeddings,dtype=tf.float32,trainable=self.update_embedding,name="_word_embeddings")
            # self._word_embeddings = tf.Variable(tf.random_uniform([self.num_words, self.hidden_dim], -1.0, 1.0)) 
            word_embeddings = tf.nn.embedding_lookup(self._word_embeddings, self.word_ids, name='word_embeddings')
        with tf.variable_scope("chars"):
            if self.use_chars:
                char_hidden_dim = self.hidden_dim // 2
                _char_embeddings = tf.get_variable(dtype=tf.float32,shape=[self.num_chars, char_hidden_dim],name="_char_embeddings")
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.char_ids, name="char_embeddings")
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings, shape=[s[0]*s[1], s[-2], char_hidden_dim])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])
                #LSTM
                cell_fw = LSTMCell(char_hidden_dim, state_is_tuple=True)
                cell_bw = LSTMCell(char_hidden_dim, state_is_tuple=True)
                _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, char_embeddings,
                                                                        sequence_length=word_lengths, dtype=tf.float32)
                #GRU
                #cell_fw = GRUCell(char_hidden_dim)
                #cell_bw = GRUCell(char_hidden_dim)
                #_, (output_fw, output_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, char_embeddings,
                #                                            sequence_length=word_lengths, dtype=tf.float32)
                output = tf.concat([output_fw, output_bw], axis=-1)
                output = tf.reshape(output, shape=[s[0], s[1], self.hidden_dim])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)
    
    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            #LSTM
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            #GRU
            #cell_fw = GRUCell(self.hidden_dim)
            #cell_bw = GRUCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                                                                cell_fw=cell_fw,
                                                                cell_bw=cell_bw,
                                                                inputs=self.word_embeddings,
                                                                sequence_length=self.sequence_lengths,
                                                                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            if self.use_lm:
                lm_emb = tf.concat([self.fw_lm_embeddings, self.bw_lm_embeddings], axis=-1)
                if self.use_att:
                    output = attention(output, lm_emb, 2*self.hidden_dim)
                else:
                    output = tf.concat([output, lm_emb], axis=-1)
            # output = tf.concat([output, self.fw_lm_embeddings], axis=-1)
            # output = tf.concat([output, self.bw_lm_embeddings], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)
        if self.use_extra_blstm:
            with tf.variable_scope("second-blstm"):
                #LSTM
                cell_fw = LSTMCell(self.hidden_dim)
                cell_bw = LSTMCell(self.hidden_dim)
                #GRU
                #cell_fw = GRUCell(self.hidden_dim)
                #cell_bw = GRUCell(self.hidden_dim)
                (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                                                                    cell_fw=cell_fw,
                                                                    cell_bw=cell_bw,
                                                                    inputs=output,
                                                                    sequence_length=self.sequence_lengths,
                                                                    dtype=tf.float32)
                output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
                output = tf.nn.dropout(output, self.dropout_pl)
        with tf.variable_scope("proj"):
            if (self.use_lm==True) and (self.use_att==False) and (self.use_extra_blstm==False):
                parameter_shape = 4 * self.hidden_dim
            else:
                parameter_shape = 2 * self.hidden_dim
            # parameter_shape = 3 * self.hidden_dim
            w = tf.get_variable(name="w",
                                shape=[parameter_shape, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)
            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, parameter_shape])
            pred = tf.matmul(output, w) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.num_tags])
            
    def loss_op(self):
        log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                    tag_indices=self.labels,
                                                                    sequence_lengths=self.sequence_lengths)
        self.loss = tf.reduce_mean(-log_likelihood)
        tf.summary.scalar("loss", self.loss)
        
    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()
        
    def add_summary(self, sess):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.output_path, sess.graph)
        
    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        if self.use_chars:
            char_ids, word_ids = zip(*seqs)
            word_ids, seq_len_list = pad_sequences(word_ids)
            char_ids, word_lengths = pad_sequences(char_ids, nlevels=2)
        else:
            word_ids, seq_len_list = pad_sequences(seqs)
        feed = {self.word_ids: word_ids, self.sequence_lengths: seq_len_list}
        if self.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths
        if labels is not None:
            labels_, _ = pad_sequences(labels)
            feed[self.labels] = labels_
        if lr is not None:
            feed[self.lr_pl] = lr
        if dropout is not None:
            feed[self.dropout_pl] = dropout
        return feed, seq_len_list
    
    def train(self, train, dev):
        best_score = 0
        nepoch_no_imprv = 0
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(self.init_op)
            self.add_summary(sess)
            for epoch in range(self.epoch_num):
                nbatches = (len(train) + self.batch_size - 1) // self.batch_size
                batches = batch_pre(train, self.batch_size)
                start_time = datetime.datetime.now()
                for step, (seqs, labels) in enumerate(batches):
                    step_num = epoch * nbatches + step + 1
                    fd, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep)
                    if self.use_lm:
                        fw_lm_embeddings = get_lm_embeddings('train', 'fw', step)
                        bw_lm_embeddings = get_lm_embeddings('train', 'bw', step)
                        fd[self.fw_lm_embeddings] = fw_lm_embeddings
                        fd[self.bw_lm_embeddings] = bw_lm_embeddings
                    # fd[self.fw_lm_embeddings] = get_lm_embeddings('train', 'fw', step)
                    # fd[self.bw_lm_embeddings] = get_lm_embeddings('train', 'fw', step)
                    # fd[self.fw_lm_embeddings] = get_lm_embeddings('train', 'bw', step)
                    # fd[self.bw_lm_embeddings] = get_lm_embeddings('train', 'bw', step)
                    _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step], feed_dict=fd)
                    if step + 1 == 1 or (step + 1) % 200 == 0 or step + 1 == nbatches:
                        now_time = datetime.datetime.now()
                        left_time = ((((now_time-start_time).seconds+1)/(step+1))*(nbatches-step-1))//60
                        self.logger.info(
                        '{} (need {:.0f}min) epoch {}, step {}/{}, loss: {:.4}, global_step: {}'.format(now_time.strftime("%H:%M"),
                                                                    left_time, epoch + 1, step + 1,nbatches,loss_train, step_num))
                    self.file_writer.add_summary(summary, step_num)
                self.logger.info('===========validation===========')
                label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev, 'dev')
                metrics = self.evaluate(label_list_dev, dev, epoch)
                for _ in metrics:
                    self.logger.info(_)
                self.lr *= 0.9
                #early stopping and saving best parameters
                score = float(metrics[1].strip('\n').split()[-1])
                if score >= best_score:
                    nepoch_no_imprv = 0
                    saver.save(sess, self.output_path+"save.ckpt")
                    best_score = score
                    self.logger.info("- new best score!")
                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= 10:
                        self.logger.info("- early stopping 10 epochs without improvement")
                        break
                    
    def test(self, test):
        self.logger.info("Testing model over test set")
        self.add_summary(self.sess)
        label_list_test, seq_len_list_test = self.dev_one_epoch(self.sess, test, 'test')
        metrics = self.evaluate(label_list_test, test, epoch=999)
        for _ in metrics:
            self.logger.info(_)
        self.sess.close()
                
    def dev_one_epoch(self, sess, dev, mode):
        label_list, seq_len_list = [], []
        step = -1
        for (seqs, labels) in batch_pre(dev, self.batch_size):
            feed_dict, seq_len_list_ = self.get_feed_dict(seqs, dropout=1.0)
            step += 1
            if self.use_lm:
                fw_lm_embeddings = get_lm_embeddings(mode, 'fw', step)
                bw_lm_embeddings = get_lm_embeddings(mode, 'bw', step)
                feed_dict[self.fw_lm_embeddings] = fw_lm_embeddings
                feed_dict[self.bw_lm_embeddings] = bw_lm_embeddings
            # feed_dict[self.fw_lm_embeddings] = get_lm_embeddings('train', 'fw', step)
            # feed_dict[self.bw_lm_embeddings] = get_lm_embeddings('train', 'fw', step)
            # feed_dict[self.fw_lm_embeddings] = get_lm_embeddings('train', 'bw', step)
            # feed_dict[self.bw_lm_embeddings] = get_lm_embeddings('train', 'bw', step)
            logits, transition_params = sess.run([self.logits, self.transition_params], feed_dict=feed_dict)
            label_list_ = []
            for logit, seq_len in zip(logits, seq_len_list_):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list_.append(viterbi_seq)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list
    
    def evaluate(self, label_list, data, epoch=0):
            cache_file = self.output_path + '/ner_result_'+str(epoch)+'.txt'
            cache_file_data = open(cache_file, 'w')
            for label_, (word_, tags) in zip(label_list, data):
                for x, y, z in zip(label_, tags, word_):
                    x = get_tag(x)
                    y = get_tag(y)
                    cache_file_data.write("{} {} {}\n".format(z[-1], y, x))
                cache_file_data.write("\n")
            cache_file_data.close()

            eval_perl = os.path.dirname(os.path.abspath(__file__)) + '/conlleval'
            output_file = self.output_path + '/output_'+str(epoch)+'.txt'
            os.system("perl {} < {} > {}".format(eval_perl, cache_file, output_file))

            with open(output_file) as fr:
                metrics = [line.strip() for line in fr]
            return metrics
                    
    def restore_session(self, dir_model):
        self.sess = tf.Session()
        self.sess.run(self.init_op)
        self.saver = tf.train.Saver()
        self.logger.info("Reloading the latest trained model...")
        self.saver.restore(self.sess, dir_model+"save.ckpt")
