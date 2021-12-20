import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import math


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def find_files_by_extensions(root, exts=[]):
    def _has_ext(name):
        if not exts:
            return True
        name = name.lower()
        for ext in exts:
            if name.endswith(ext):
                return True
        return False
    for path, _, files in os.walk(root):
        for name in files:
            if _has_ext(name):
                yield os.path.join(path, name)


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def parse_batch(items):
    def _process(item):
        item = tf.io.parse_single_example(
            item, {'seq': tf.io.FixedLenFeature([], tf.string)})
        seq = tf.io.parse_tensor(item['seq'], tf.int64)
        return seq
    seq = map(_process, items)
    return np.array(list(seq))


def batch_parse2(items):
    items = tf.io.parse_example(
        items, {'seq': tf.io.FixedLenFeature([], tf.string)})
    items = items['seq']
    items = tf.map_fn(lambda x: tf.io.parse_tensor(
        items[x], tf.int64), fn_output_signature=tf.int64, elems=tf.convert_to_tensor(range(len(items))))
    return items


# for CPWordEncoding
# 
def token2vocab(token):
    token = np.array(token)
    pad = np.not_equal(token, 0) * 1
    token[...,0] -= 1 # family
    token[...,1] = [1 if i == 1 else i - 189 for i in token[...,1]] # bar
    token[...,2] -= 3# pitch
    token[...,3] -= 92 # Velocity
    token[...,4] -= 125 # Duration
    token[...,5] -= 223 # Chord
    token[...,6] -= 241 # Rest
    token[...,7] -= 251 # Tempo
    token *= pad
    return token 

def vocab2token(vocab):
    vocab = np.array(vocab)
    pad = np.not_equal(vocab, 0) * 1
    vocab[...,0] += 1 # family
    vocab[...,1] = [1 if i == 1 else i + 189 for i in vocab[...,1]] # bar
    vocab[...,2] += 3 # pitch
    vocab[...,3] += 92 # Velocity
    vocab[...,4] += 125 # Duration
    vocab[...,5] += 223 # Chord
    vocab[...,6] += 241 # Rest
    vocab[...,7] += 251 # Tempo
    vocab *= pad
    return vocab
    
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        
        
# def checkMidi(seq)
#     ft = seq[0] # family_type

def nucleus(probs, p):
    probs = np.asarray(probs).astype('float64')
    probs /= (sum(probs) + 1e-5)
#     probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word

def weight(probs):
     pdf = tfp.distributions.Categorical(probs=probs)
     sp = pdf.sample(1)[0]
     return tf.cast(sp, tf.int64)

def sampling(logit, p=False, t=1.0):
    probs = tf.nn.softmax(logit/t, -1)    
    if p != False:
        return [nucleus(pb, p) for pb in probs]
    else:
        return weight(probs)
    
class DynamicTemperature():
    def __init__(self, warmup_num):
        super(DynamicTemperature, self).__init__()
        self.warmup_num = warmup_num

    def __call__(self, base_t, number):
        arg1 = 1 / math.sqrt(number)
        arg2 = number * (self.warmup_num ** -1.5)
        p = (1 / math.sqrt(1 / base_t)) * min(arg1, arg2)
        return p * 12.8

    
def dt(t, length):
    return t + math.sin(length/2) * 0.08
