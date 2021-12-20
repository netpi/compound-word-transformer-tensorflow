import numpy as np
import json
from utils import token2vocab
from miditok import CPWordEncoding

class CP_Word_Dataset():
    
    def __init__(self, dir, length, eos_tokens = None):
        self.dir = dir
        self.data = json.loads(open(self.dir, 'r').read())['data']
        self.length = length

        dataset_idx = [0]  # records the seq start index

        idx = 0
        bar_idx_final ={}
        for seq in self.data:
            seq = np.array(seq)
            seq_len = len(seq)
            bar_idx = np.where(seq[...,1] == 1)[0] # bar token
            
            bar_seq_len_final = len(np.where(bar_idx + length < seq_len)[0]) + 1
            bar_idx_final[idx] = bar_idx[:bar_seq_len_final]

            dataset_idx = np.append(
                dataset_idx, dataset_idx[-1] + bar_seq_len_final)
            idx+=1

        self.bar_idx_final = bar_idx_final
        self.dataset_idx = dataset_idx
        self.total_seq = dataset_idx[-1]
        self.eos_tokens = eos_tokens

    def get_seqs(self, idxs):

        return [token2vocab(self._getSeq(idx)) for idx in idxs]

    def _getSeq(self, idx):
        seq_n = np.where(self.dataset_idx > idx)[0][0] - 1

        bar_idx_n = self.bar_idx_final[seq_n]

        comsum_seq_count = self.dataset_idx[seq_n]

        start_idx = idx - comsum_seq_count

        final_start_idx = bar_idx_n[start_idx]

        r = self.data[seq_n][final_start_idx: final_start_idx + self.length] # (seq, last_dim)
        
        r = np.array(r)

        if r.shape[0] < self.length: # 

            r = np.concatenate([r, np.expand_dims(self.eos_tokens, 0)], 0)

            while r.shape[0] < self.length:
                pad_tokens = np.zeros((1, r.shape[-1]), dtype=np.int64)
                r = np.concatenate([r, pad_tokens], 0)

        return r
    
def get_tokenizer():
    pitch_range = range(21, 109)
    beat_res = {(0, 4): 8, (4, 12): 4}
    nb_velocities = 32

    additional_tokens = {
                    'Chord': True,
                    'Rest': True,
                    'Tempo': True,
                    'rest_range': (2, 8),  # (half, 8 beats)
                    'nb_tempos': 32,  # nb of tempo bins
                    'tempo_range': (40, 250),  # (min, max)
                    'Program':False,
                    }

    return CPWordEncoding(pitch_range, beat_res, nb_velocities, additional_tokens)