import math
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tensorflow_probability as tfp
import random
from pathlib import Path
from utils import *

# tf.config.run_functions_eagerly(True)


class model:

    def getTransformerXL(self, config, log_dir, checkpoint_dir):

        if log_dir != None:
            for key in config.keys():
                log_dir += f"-{key}{config[key]}"
            Path(log_dir).mkdir(parents=True, exist_ok=True)

        if checkpoint_dir != None:
            for key in config.keys():
                checkpoint_dir += f"-{key}{config[key]}"
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        mt = TransformerXL(
            num_layers=config['num_layers'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            dff=config['dff'],
            vocab_size=config['vocab_size'],
            length=config['length'],
            rate=config['dropout_rate'],
            rpr=config['rpr']
        )

        mt.log_dir = log_dir
        mt.checkpoint_dir = checkpoint_dir

        return mt

    def getTransformer(self, config, log_dir, checkpoint_dir):

        if log_dir != None:
            for key in config.keys():
                log_dir += f"-{key}{config[key]}"
            Path(log_dir).mkdir(parents=True, exist_ok=True)

        if checkpoint_dir != None:
            for key in config.keys():
                checkpoint_dir += f"-{key}{config[key]}"
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        mt = Transformer(
            num_layers=config['num_layers'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            dff=config['dff'],
            vocab_size=config['vocab_size'],
            enc_length=config['enc_length'],
            dec_length=config['dec_length'],
            rate=config['dropout_rate'],
            rpr=config['rpr']
        )

        mt.log_dir = log_dir
        mt.checkpoint_dir = checkpoint_dir

        return mt

    def getLinearTransformerXL(self, config, log_dir=None, checkpoint_dir=None):
        c = config
        # if log_dir !=None :
        #   for key in c.keys(): log_dir += f"-{key}{ '|'.join(c[key]) if isinstance(c[key], list) else c[key]}"
        #   Path(log_dir).mkdir(parents=True, exist_ok=True)

        def process_dir(dir):
            for key in c.keys():
                dir += f"-{key}{ '-'.join([str(i) for i in c[key]]) if isinstance(c[key], list) else c[key]}"
            return dir

        if log_dir != None:
            log_dir = process_dir(log_dir)
            Path(log_dir).mkdir(parents=True, exist_ok=True)

        if checkpoint_dir != None:
            checkpoint_dir = process_dir(checkpoint_dir)
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        mt = LinearTransformerXL(
            vocab_sizes=config['vocab_sizes'],
            emb_sizes=config['emb_sizes'],
            num_layers=config['num_layers'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            dff=config['dff'],
            length=config['length'],
            rate=config['dropout_rate'],
            rpr=config['rpr']
        )

        mt.log_dir = log_dir
        mt.checkpoint_dir = checkpoint_dir

        return mt


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, maximum_position_encoding, rpr=True):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.max_seq = maximum_position_encoding
        self.rpr = rpr
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def build(self, x):
        self.E = self.add_weight('emb', shape=[self.max_seq, int(self.depth)])
        # self.E_decoder = self.add_weight('emb', shape=[self.max_seq, int(self.depth)])

    def _get_left_embedding(self, len_q):
        # starting_point = max(0,self.max_seq-len_q)

        s = self.max_seq-len_q
        if s > 0:
            starting_point = s
        else:
            starting_point = 0

        e = self.E[starting_point:, :]
        return e

    @staticmethod
    def _qe_masking(qe):
        mask = tf.sequence_mask(
            tf.range(tf.shape(qe)[-1] - 1, tf.shape(qe)[-1] - tf.shape(qe)[-2] - 1, -1), tf.shape(qe)[-1])

        mask = tf.logical_not(mask)
        mask = tf.cast(mask, tf.float32)

        return mask * qe

    def _skewing(self, tensor: tf.Tensor):
        padded = tf.pad(tensor, [[0, 0], [0, 0], [0, 0], [1, 0]])
        reshaped = tf.reshape(
            padded, shape=[-1, tf.shape(padded)[1], tf.shape(padded)[-1], tf.shape(padded)[-2]])
        Srel = reshaped[:, :, 1:, :]
        if self.len_k > self.len_q:
            Srel = tf.pad(Srel, [[0, 0], [0, 0], [0, 0],
                          [0, self.len_k-self.len_q]])
        elif self.len_k < self.len_q:
            Srel = Srel[:, :, :, :self.len_k]
        return Srel

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
          q: query shape == (..., seq_len_q, depth)
          k: key shape == (..., seq_len_k, depth)
          v: value shape == (..., seq_len_v, depth_v)
          mask: Float tensor with shape broadcastable
                to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
          output, attention_weights
        """

        # (..., seq_len_q, seq_len_k)
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        if self.rpr:
            E = self._get_left_embedding(tf.shape(q)[2])
            QE = tf.einsum('bhld,md->bhlm', q, E)
            QE = self._qe_masking(QE)
            Srel = self._skewing(QE)
            matmul_qk = matmul_qk + Srel

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(
            scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)
        self.len_q = tf.shape(q)[2]
        self.len_k = tf.shape(k)[2]
        self.len_v = tf.shape(v)[2]
        # print('''''',self.len_q, self.len_k, self.len_v)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, maximum_position_encoding, rate=0.1, rpr=False):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(
            d_model, num_heads, maximum_position_encoding, rpr)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        # (batch_size, input_seq_len, d_model)
        attn_output, attn_weights = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2, attn_weights


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1, rpr=False):

        print('num_layers', num_layers)
        print('d_model', d_model)
        print('num_heads', num_heads)
        print('dff', dff)
        print('input_vocab_size', input_vocab_size)
        print('maximum_position_encoding', maximum_position_encoding)
        # maximum_position_encoding
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, maximum_position_encoding, rate, rpr)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, attn_weights = self.enc_layers[i](x, training, mask)
            attention_weights[f'encoder_layer{i+1}'] = attn_weights
        return x, attention_weights  # x = (batch_size, input_seq_len, d_model)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, maximum_position_encoding, rate=0.1, rpr=False):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(
            d_model, num_heads, maximum_position_encoding, rpr)
        self.mha2 = MultiHeadAttention(
            d_model, num_heads, maximum_position_encoding, rpr)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # (batch_size, target_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        # (batch_size, target_seq_len, d_model)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        # (batch_size, target_seq_len, d_model)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1, rpr=False):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(
            maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, maximum_position_encoding, rate, rpr)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size,
                 enc_length, dec_length, rate=0.1, rpr=False):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               vocab_size, enc_length, rate, rpr)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               vocab_size, dec_length, rate, rpr)

        self.final_layer = tf.keras.layers.Dense(vocab_size)

        self.enc_length = enc_length

    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(
            inp, tar)

        # (batch_size, inp_seq_len, d_model)
        enc_output, _ = self.encoder(inp, training, enc_padding_mask)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights

    def eval(self, eval_seq):

        enc_inp = eval_seq[:, :self.enc_length]
        dec_inp = eval_seq[:, :-1]
        tar_real = eval_seq[:, 1:]
        predictions, _ = self([enc_inp, dec_inp])
        loss = self.loss_function(tar_real, predictions)
        acc = self.accuracy_function(tar_real, predictions)
        return loss, acc

    def generate(self, inp, length=1024, random_seed=0.7):
        seed, gen = inp
        inp_len = len(gen[0])
        with tqdm(total=length - inp_len) as bar:
            bar.updata(inp_len)
            while len(gen[0]) < length:

                y, _ = self([seed, gen], False)
                u = random.uniform(0, 1)

                if u > random_seed:
                    tf.argmax(y[:, -1], -1)
                    y = tf.argmax(y[:, -1], -1)
                    y = tf.cast(y, tf.int64)
                    gen = tf.concat([gen, tf.expand_dims(y, -1)], -1)

                else:
                    probs = tf.nn.softmax(y[:, -1])
                    pdf = tfp.distributions.Categorical(probs=probs)
                    y = pdf.sample(1)
                    y = tf.transpose(y, (1, 0))
                    y = tf.cast(y, tf.int64)
                    gen = tf.concat([gen, y], -1)

                bar.update(1)

        return gen

    def train_setup(self, loss_function, accuracy_function, optimizer, mirrored_strategy=None):
        self.optimizer = optimizer
        self.accuracy_function = accuracy_function
        self.loss_function = loss_function
        self.mirrored_strategy = mirrored_strategy

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64)
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(self, seq):
        enc_inp = seq[:, :self.enc_length]
        dec_inp = seq[:, :-1]
        tar_real = seq[:, 1:]
        with tf.GradientTape() as tape:
            predictions, _ = self([enc_inp, dec_inp], training=True)
            loss = self.loss_function(tar_real, predictions)
            accuracy = self.accuracy_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        return loss, accuracy

    # @tf.function(input_signature=train_step_signature)
    @tf.function
    def distributed_train_step(self, seq):

        per_replica_losses, per_replica_accuracy = self.mirrored_strategy.run(
            self.train_step, args=(seq,))
        losses = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                               axis=None)
        accuracy = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_accuracy,
                                                 axis=None)
        return losses, accuracy

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask


class TransformerXL(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size,
                 length, rate=0.1, rpr=True):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               vocab_size, length, rate, rpr)

        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp = inputs

        _, look_ahead_mask, _ = self.create_masks(inp, inp)

        # (batch_size, inp_seq_len, d_model)
        out, attention_weights = self.encoder(inp, training, look_ahead_mask)

        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(out)

        return final_output, attention_weights

    def eval(self, eval_seq):

        eval_inp = eval_seq[:, :-1]
        eval_real = eval_seq[:, 1:]
        predictions, _ = self(eval_inp)
        eval_loss = self.loss_function(eval_real, predictions)
        eval_acc = self.accuracy_function(eval_real, predictions)
        return eval_loss, eval_acc

    def generate(self, inp, length=1024, random_seed=0.7):
        gen = inp
        inp_len = len(gen[0])
        with tqdm(total=length) as bar:
            bar.update(inp_len)
            while len(gen[0]) < length:

                y, _ = self(gen, False)

                u = random.uniform(0, 1)
                if u > random_seed:
                    y = tf.argmax(y[:, -1], -1)
                    y = tf.cast(y, tf.int64)
                    gen = tf.concat([gen, tf.expand_dims(y, -1)], -1)

                else:
                    probs = tf.nn.softmax(y[:, -1])
                    pdf = tfp.distributions.Categorical(probs=probs)
                    y = pdf.sample(1)
                    y = tf.transpose(y, (1, 0))
                    y = tf.cast(y, tf.int64)
                    gen = tf.concat([gen, y], -1)

                bar.update(1)

        return gen

    def train_setup(self, loss_function, accuracy_function, optimizer, mirrored_strategy=None):
        self.optimizer = optimizer
        self.accuracy_function = accuracy_function
        self.loss_function = loss_function
        self.mirrored_strategy = mirrored_strategy

    @tf.function
    def train_step(self, seq):
        dec_inp = seq[:, :-1]
        tar_real = seq[:, 1:]
        with tf.GradientTape() as tape:
            predictions, _ = self(dec_inp, training=True)
            loss = self.loss_function(tar_real, predictions)
            accuracy = self.accuracy_function(tar_real, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))
        return loss, accuracy

    # @tf.function(input_signature=train_step_signature)
    @tf.function
    def distributed_train_step(self, seq):

        with self.mirrored_strategy.scope():
            per_replica_losses, per_replica_accuracy = self.mirrored_strategy.run(
                self.train_step, args=(seq,))
            losses = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                                   axis=None)
            accuracy = self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_accuracy,
                                                     axis=None)
        return losses, accuracy
    # def train(self):
    # def generate(self, inp, length = 1024):
    #   # self.

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask


class LinearTransformerXL(tf.keras.Model):
    def __init__(self, vocab_sizes, emb_sizes, num_layers, d_model, num_heads, dff,
                 length, rate=0.1, rpr=True):
        super().__init__()

        self.vocab_sizes = vocab_sizes
        self.emb_sizes = emb_sizes
        self.d_model = d_model
        self.num_layers = num_layers

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, length, rate, rpr)
                           for _ in range(num_layers)]

        self.last_dim = len(vocab_sizes)

        # process inp
        self.inp_embeddings = [tf.keras.layers.Embedding(self.vocab_sizes[i], self.emb_sizes[i])
                               for i in range(self.last_dim)]

        self.inp_dense = tf.keras.layers.Dense(self.d_model)
        self.pos_encoding = positional_encoding(length, self.d_model)
        self.dropout = tf.keras.layers.Dropout(rate)

        # process out
        self.out_denses = [tf.keras.layers.Dense(self.vocab_sizes[i])
                           for i in range(self.last_dim)]
    def call(self, inputs, training, tar=None, for_family_type = False, family_type=None, h=None):
        
        if h !=None: # reasoning by family type
        
          x_ = tf.concat([h, self.inp_embeddings[0](family_type)], -1)
          out_group = [self.out_denses[i+1](x_) for i in range(self.last_dim - 1)]
          return out_group, x_
        
        seq_len = tf.shape(inputs)[1]
        attention_weights = {}

        inp_group = [self.inp_embeddings[i](inputs[..., i])
                     for i in range(self.last_dim)]

        all_embedding = tf.concat(
            inp_group, -1)
        x = self.inp_dense(all_embedding)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        mask = create_look_ahead_mask(seq_len)

        for i in range(self.num_layers):
            x, attn_weights = self.enc_layers[i](x, training, mask)
            attention_weights[f'encoder_layer{i+1}'] = attn_weights

        if for_family_type:
          return self.out_denses[0](x), x

        if training:
          x_ = tf.concat([x, self.inp_embeddings[0](tar[..., 0])], -1)
          out_group = [self.out_denses[i](x) if i == 0 else self.out_denses[i](x_)
                           for i in range(self.last_dim)]
          return out_group, x_

    def eval(self, eval_seq):
        eval_seq = np.array(eval_seq)
        inp = eval_seq[:, :-1]
        tar = eval_seq[:, 1:]
        predictions, _ = self(inp, True, tar)
        dim = inp.shape[-1]
        losses = [self.loss_function(tar[..., i], predictions[i])
                  for i in range(dim)]
        accuracy = [self.accuracy_function(
            tar[..., i], predictions[i]) for i in range(dim)]

        return losses, accuracy

    def test(self, ds, batch_size=20):
        
        m_loss = tf.keras.metrics.Mean(name=f'test_loss')
        m_acc = tf.keras.metrics.Mean(name=f'test_acc')
        m_loss.reset_states()
        m_acc.reset_states()
        total = math.ceil(ds.total_seq / batch_size)
        random.seed(1)
        idxs = random.sample(
            range(ds.total_seq), ds.total_seq)
        idxs = idxs[: 1000]
        with tqdm(total=total) as bar:
            for batch_idxs in batch(idxs, batch_size):
                seqs = ds.get_seqs(batch_idxs)
                e_losses, e_acc = self.eval(seqs)
                m_loss(np.sum([l.numpy() for l in e_losses]) / len(e_losses)) # 均值
                m_acc(np.sum(e_acc) / len(e_losses))
                bar.set_description('Test>>>>>>>>>>')
                bar.update(1)
        return m_loss.result().numpy(), m_acc.result().numpy()

    @tf.function
    def train_step(self, seq):

        dec_inp = seq[:, :-1]
        tar_real = seq[:, 1:]
        dim = dec_inp.shape[-1]

        # print(self.trainable_variables)
        with tf.GradientTape() as tape:
            predictions, _ = self(dec_inp, True, tar_real)
            losses = [self.loss_function(
                tar_real[..., i], predictions[i]) for i in range(dim)]
            accuracy = [self.accuracy_function(
                tar_real[..., i], predictions[i]) for i in range(dim)]

        gradients = tape.gradient(losses, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))
        return losses, accuracy

    def train_setup(self, loss_function, accuracy_function, optimizer, mirrored_strategy=None):
        self.optimizer = optimizer
        self.accuracy_function = accuracy_function
        self.loss_function = loss_function
        self.mirrored_strategy = mirrored_strategy
        
    def generate(self, inp, length, temperature, nucleus_p,  if_end = False):
        gen = inp
        inp_len = len(gen[0])
        tp = temperature
        dim = gen.shape[-1]
        eos_tokens = [i-1 for i in self.vocab_sizes]
        with tqdm(total=length) as bar:
            bar.update(inp_len)

            while len(gen[0]) < length:
                gen_ = gen[:,-1024:]
                y_ft, h = self(gen_, False, for_family_type = True) # get family type
                
                ft_logits = y_ft[:, -1] #(batch, vocab_size)
                
                sampling_ft = sampling(ft_logits, t=tp[0], p=nucleus_p[0])
                
                gen_ft = gen_[...,0][:, 1:] #(batch_size, seq_len-1)
                
                gen_ft = tf.concat([gen_ft, tf.expand_dims(sampling_ft, 1)],1) #(batch_size, seq_len)
                
                y,_ = self(gen_, False, family_type = gen_ft, h=h) # get [Bar/position Pitch Velocity Duration Tempo]
                
                r = np.array([sampling_ft]) # (1 batch_size )
                
                for i in range(dim-1):
                    
                    logits = np.array(y[i][:, -1])  
                    if if_end ==False: 
                           
                        logits[...,-1] -= 1e5 # mask eos token
                        
                    if i==6: # Tempo
                          
                          ones = np.ones(len(logits)) # (batch)  
                          mask = tf.cast(np.equal(r[0], ones*2), tf.float32) # if Family_Metric --> mask ignore
                          mask2 = tf.cast(np.not_equal(r[1], ones*1), tf.float32) # if Family_Metric --> mask ignore
                          logits[...,1] -= mask * mask2 * 1e5

                    if i==0: # position
                        
                          ones = np.ones(len(logits)) # (batch)                
                          last_positoin = np.squeeze(gen[:,-1,1]) # (batch)                            
                          mask = tf.cast(np.equal(last_positoin, ones*1), tf.float32) 
                          logits[...,1] -= mask * 1e5 # if last position is bar, mask this bar
                            
                    t = tp[i+1]
                    p = sampling(logits, t=t, p=nucleus_p[i+1])
                    r = np.append(r, [p], 0) # (last_dim, batch_size)
                if if_end:
                    squeeze_r = np.squeeze(r)
                    if np.any(np.equal(eos_tokens, squeeze_r) == True): break 
                gen = tf.concat([gen, tf.expand_dims(tf.transpose(r), 1)], 1)

                bar.update(1)

        return gen
                
        
    
class Train_process_json:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        Path(dir_path).touch(exist_ok=True)

    def get(self):
        f = open(self.dir_path, 'r')
        content = f.read()
        if content == '':
            return {"step": 0, "best_acc": 0.0, "best_loss": 10000.0}
        else:
            config = json.loads(content)
            for key in ['best_acc', 'best_loss']:
                config[key] = float(config[key])
            return config

    def set(self, step, best_acc, best_loss):
        config = self.get()
        config['step'] = step
        config['best_acc'] = str(best_acc)
        config['best_loss'] = str(best_loss)

        with open(self.dir_path, 'w') as f:
            json.dump(config, f)
        return
