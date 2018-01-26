import tensorflow as tf
import argparse
import pickle
from tqdm import tqdm
from tensorflow.contrib import rnn
from data_util import BatchLoader, wordtag2Xy
from sklearn.model_selection import train_test_split
from tools import AverageMeter, viterbi
import numpy as np
import os
import re

parser = argparse.ArgumentParser(description='CWS biLSTM tensorflow implementation')
parser.add_argument('--maybe-train', type=int, default=1,
                    help='maybe train the network, 1 means train net, 0 means '
                         'running the model')
parser.add_argument('--c-str', type=str, default='而不是把你前面看的内容全部抛弃了，忘记了，再去理解这个单词。',
                    help='if maybe_train set to False, pass this arg as sentence to do CWS')
parser.add_argument('--batch-size', type=int, default=2048, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--time-step', type=int, default=32, metavar='TS',
                    help='time step size or maximum length or patch size (default: 0.5)')
# parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                     help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                    help='learning rate (default: 0.01)')
# parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                     help='SGD momentum (default: 0.5)')
# parser.add_argument('--decay', type=float, default=0.85, metavar='D',
#                     help='Weight decay (default: 0.85)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--embedding-size', type=int, default=64, metavar='EB',
                    help='input word vector size aka embedding size')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

# Sleect GPU ID
if args.no_cuda is not True:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="2"

pickle_file = 'data/msr_train.pkl'
model_save_path = 'data/best_model.ckpt'
wordSetNum = 5159
classNum = 5
hiddenSize = 128
layerNum = 2
# keepProb = 0.5  # Dropout keeping rate
maxGrad = 5.0

with tf.variable_scope('embedding'):
    embedding = tf.get_variable("embedding", [wordSetNum, args.embedding_size], dtype=tf.float32)
keepProb = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def lstm_cell():
    cell = rnn.LSTMCell(hiddenSize)
    return rnn.DropoutWrapper(cell, output_keep_prob=keepProb)


def biLSTM(inputsX):
    input = tf.nn.embedding_lookup(embedding, inputsX)

    lstm_fw = rnn.MultiRNNCell([lstm_cell() for _ in range(layerNum)], state_is_tuple=True)
    lstm_bw = rnn.MultiRNNCell([lstm_cell() for _ in range(layerNum)], state_is_tuple=True)

    initial_state_fw = lstm_fw.zero_state(batch_size, tf.float32)
    initial_state_bw = lstm_bw.zero_state(batch_size, tf.float32)

    with tf.variable_scope('biLSTM_CWS'):
        outputs_fw = list()
        state_fw = initial_state_fw
        with tf.variable_scope('fw'):
            for timestep in range(args.time_step):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                output_fw, state_fw = lstm_fw(input[:, timestep, :], state_fw)
                outputs_fw.append(output_fw)

        outputs_bw = list()
        state_bw = initial_state_bw
        with tf.variable_scope('bw'):
            for timestep in range(args.time_step):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                output_bw, state_bw = lstm_bw(input[:, args.time_step - timestep - 1, :], state_bw)
                outputs_bw.append(output_bw)

        output = tf.concat([outputs_fw, outputs_bw], 2)
        output = tf.transpose(output, perm=[1, 0, 2])
        output = tf.reshape(output, [-1, hiddenSize*2])
    return output


# Load pickle file for word/tag dictionary and training data
with open(pickle_file, 'rb') as pk:
    X = pickle.load(pk)
    y = pickle.load(pk)
    word2id = pickle.load(pk)
    id2word = pickle.load(pk)
    tag2id = pickle.load(pk)
    id2tag = pickle.load(pk)
    trans_prob = pickle.load(pk)
trans_prob = {i:np.log(trans_prob[i]) for i in trans_prob.keys()}

# construct model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
with tf.variable_scope('inputs'):
    X_input = tf.placeholder(tf.int32, [None, args.time_step], name='X_input')
    y_input = tf.placeholder(tf.int32, [None, args.time_step], name='y_input')
biLSTMOut = biLSTM(X_input)
with tf.variable_scope('outputs'):
    fcW = weight_variable([hiddenSize*2, classNum])
    fcb = bias_variable([classNum])
    y_pred = tf.matmul(biLSTMOut, fcW) + fcb

correct_pred = tf.equal(tf.cast(tf.argmax(y_pred, 1), tf.int32),
                        tf.reshape(y_input, [-1]))
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=tf.reshape(y_input, [-1]), logits=y_pred))
trainable = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, trainable), maxGrad)
optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
train_op = optimizer.apply_gradients( zip(grads, trainable),
    global_step=tf.train.get_or_create_global_step())
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=10)


def test_epoch(dataset, keep_prob=1.0, batch_size=2048):
    _accs = AverageMeter()
    _losses = AverageMeter()
    sampleNum = dataset.sample_number
    for batch in range(sampleNum // batch_size):
        X_batch, y_batch = dataset.next_batch(batch_size=batch_size)
        feed_dict = {X_input: X_batch, y_input: y_batch, keepProb: keep_prob, batch_size: batch_size}
        fetches = [acc, loss]
        _acc, _loss = sess.run(fetches, feed_dict)
        _accs.update(_acc)
        _losses.update(_loss)
    return _accs, _losses


def train():
    print('Training...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=10)
    train_data = BatchLoader(X_train, y_train, shuffle=True)
    val_data = BatchLoader(X_val, y_val, shuffle=False)
    test_data = BatchLoader(X_test, y_test, shuffle=False)

    def train_epoch(dataset, keep_prob=0.5, batch_size=2048):
        _accs = AverageMeter()
        _losses = AverageMeter()
        sampleNum = dataset.sample_number
        for batch in tqdm(range(sampleNum // batch_size)):
            X_batch, y_batch = dataset.next_batch(batch_size=batch_size)
            feed_dict = {X_input: X_batch, y_input: y_batch, keepProb: keep_prob, batch_size: batch_size}
            fetches = [acc, loss, train_op]
            _acc, _loss, _ = sess.run(fetches, feed_dict)
            _accs.update(_acc)
            _losses.update(_loss)
        return _accs, _losses

    for epoch in (range(args.epochs)):
        print('Training epoch: {}'.format(epoch))
        _accs, _losses = train_epoch(train_data, keep_prob=0.5, batch_size=args.batch_size)
        val_acc, val_loss = test_epoch(val_data, keep_prob=1.0, batch_size=args.batch_size)
        print('Training accuracy: {},  loss: {}. Validation accuracy: {},  loss: {}'.format(
            _accs.avg, _losses.avg, val_acc.avg, val_loss.avg))
        if epoch % 3 == 0:
            save_path = saver.save(sess, model_save_path, global_step=(epoch + 1))
            print('Saving model checkpoint to {}'.format(save_path))

    # Test trained model on test set
    print('Testing net...')
    test_acc, test_loss = test_epoch(test_data, keep_prob=1.0, batch_size=args.batch_size)
    print('Testing result: accuracy {}, loss {}'.format(test_acc.avg, test_loss.avg))


def simple_cut(text):
    if text:
        text_len = len(text)
        text = list(text)
        X_batch = wordtag2Xy(text, word2id)
        X_batch = np.asarray(X_batch).reshape([-1, args.time_step])
        fetches = [y_pred]
        feed_dict = {X_input: X_batch, keepProb: 1.0, batch_size: 1}
        _y_pred = sess.run(fetches, feed_dict)[0][:text_len]  # padding填充的部分直接丢弃
        nodes = [dict(zip(['s', 'b', 'm', 'e'], each[1:])) for each in _y_pred]
        tags = viterbi(nodes, trans_prob)
        words = []
        for i in range(len(text)):
            if tags[i] in ['s', 'b']:
                words.append(text[i])
            else:
                words[-1] += text[i]
        return words
    else:
        return []


def seg_sentence(sentence):
    saver.restore(sess, 'data/best_model.ckpt-10')
    not_cuts = re.compile(u'([0-9\da-zA-Z ]+)|[。，、？！.\.\?,!]')
    result = []
    start = 0
    for seg_sign in not_cuts.finditer(sentence):
        result.extend(simple_cut(sentence[start:seg_sign.start()]))
        result.append(sentence[seg_sign.start():seg_sign.end()])
        start = seg_sign.end()
    result.extend(simple_cut(sentence[start:]))
    rst_str = ''
    for words in result:
        rst_str = rst_str + words + '/'
    return rst_str


if __name__ == '__main__':
    if args.maybe_train == 1:
        train()
    else:
        rst_str = seg_sentence(args.c_str)
        print('The result is:')
        print(rst_str)
