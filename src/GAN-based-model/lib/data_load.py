import itertools
import os
import pickle
import sys
import _pickle as pk
import numpy as np

import torch
import torch.nn.functional as F
import yaml

# from lib.alphabet import Alphabet
from distsup import utils
from distsup.alphabet import Alphabet


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(path):
    return AttrDict(yaml.safe_load(open(path, 'r')))


class data_loader(object):
    def __init__(self,
        config,
        feat_path,
        phn_path,
        orc_bnd_path,
        train_bnd_path=None,
        target_path=None,
        data_length=None,
        phn_map_path='./phones.60-48-39.map.txt',
        name='DATA LOADER'):

        cout_word = f'{name}: loading    '
        sys.stdout.write(cout_word)
        sys.stdout.flush()

        self.phn_max_length = config.phn_max_length
        self.feat_max_length = config.feat_max_length
        self.concat_window = config.concat_window
        self.sample_var = config.sample_var
        self.feat_path = feat_path
        self.phn_path = phn_path
        self.orc_bnd_path = orc_bnd_path
        self.train_bnd_path = train_bnd_path
        self.target_path = target_path
        #
        # self.read_phn_map(phn_map_path)
        #
        # feat = self.load_pickle(feat_path)
        # phn = self.load_pickle(phn_path)
        # orc_bnd = self.load_pickle(orc_bnd_path)
        # assert (len(feat) == len(phn) == len(orc_bnd))
        #
        # self.data_length = len(feat) if data_length is None else data_length
        # self.process_feat(feat[:self.data_length])
        # self.process_label(orc_bnd[:self.data_length], phn[:self.data_length])
        #
        # if train_bnd_path is not None:
        #     self.process_train_bnd(train_bnd_path)
        #
        # if target_path is not None:
        #     self.process_target(target_path)
        # print(os.listdir('../..'))
        # texts_path = '../../data/texts_train.pickle'
        # vocabulary = '../../data/tasman.alphabet.plus.space.mode5.json'
        # with open(texts_path, 'rb') as f:
        #     self.texts = pickle.load(f)
        # self.alphabet = Alphabet(vocabulary)
        # self.alignments = [
        #     torch.tensor(self.alphabet.symList2idxList(t))
        #     for t in self.texts
        # ]
        #
        # for i in range(len(self.alignments)):
        #     length = self.alignments[i].shape[0]
        #     if length < config.phn_max_length:
        #         self.alignments[i] = torch.cat([
        #             self.alignments[i],
        #             torch.zeros(config.phn_max_length - length,
        #                         dtype=torch.long)
        #         ])
        #     else:
        #         self.alignments[i] = self.alignments[i][:config.phn_max_length]
        # self.alignments = torch.stack(self.alignments, )
        # self.features = F.one_hot(
        #     self.alignments,
        #     num_classes=len(self.alphabet)
        # ).float().numpy()
        # self.alignments = self.alignments.numpy()

        self.distsup_dataloader = utils.construct_from_kwargs(
            yaml.safe_load(f'''
    class_name: distsup.data.FixedDatasetLoader
    batch_size: 32
    dataset:
        class_name: egs.scribblelens.simple_dataset.TextScribbleLensDataset
        shape_as_image: false
        max_lenght: 75
        tokens_protos:
            class_name: distsup.modules.gan.utils.EncoderTokensProtos
            path: "data/55_acc_letters_protoypes.npz"
            protos_per_token: 256
    shuffle: true
    num_workers: 4
    drop_last: true
            '''
            )
        )
        self.feat_dim = 68
        self.phn_size = 68
        self.data_length = len(self.distsup_dataloader) * 32
        self.target_data_length = self.data_length
        self.phn_mapping = {i: i for i in range(200)}

        self.batches = (
            {k: t.numpy() for k,t in batch.items()}
            for _ in itertools.count()
            for batch in self.distsup_dataloader
        )

        sys.stdout.write('\b' * len(cout_word))
        cout_word = f'{name}: finish     '
        sys.stdout.write(cout_word + '\n')
        sys.stdout.flush()
        print('=' * 80)

    def load_pickle(self, file_name):
        return pk.load(open(file_name, 'rb'))

    def read_phn_map(self, path):
        all_lines = open(path, 'r').read().splitlines()
        phn_mapping = {}
        for line in all_lines:
            if line.strip() == "":
                continue
            phn_mapping[line.split()[1]] = line.split()[2]

        all_phn = list(phn_mapping.keys())
        assert (len(all_phn) == 48)
        self.phn2idx = dict(zip(all_phn, range(len(all_phn))))
        self.idx2phn = dict(zip(range(len(all_phn)), all_phn))
        self.phn_size = len(all_phn)
        self.phn_mapping = {}
        self.sil_idx = self.phn2idx['sil']
        for phn in all_phn:
            self.phn_mapping[self.phn2idx[phn]] = phn_mapping[phn]

    def pad_value(self, seq, value, max_length):
        clip_seq = seq[:max_length]
        pad_lens = [(0, max_length - len(clip_seq))]
        pad_lens.extend([(0, 0) for _ in range(len(seq.shape) - 1)])
        return np.lib.pad(clip_seq, pad_lens,
                          'constant', constant_values=(0, value))

    def process_train_bnd(self, train_bnd_path):
        train_bound = self.load_pickle(train_bnd_path)[:self.data_length]
        assert (len(train_bound) == self.data_length)
        self.train_bnd = np.zeros(
            shape=[self.data_length, self.phn_max_length],
            dtype='int32'
        )
        self.train_bnd_range = np.zeros(
            shape=[self.data_length,
                self.phn_max_length],
            dtype='int32'
        )
        self.train_seq_length = np.zeros(
            shape=[self.data_length],
            dtype='int32'
        )

        for idx, bnd in enumerate(train_bound):
            self.train_bnd[idx] = self.pad_value(
                np.array(bnd[:-1]),
                0,
                self.phn_max_length
            )
            self.train_bnd_range[idx] = self.pad_value(
                np.array(bnd[1:]) - np.array(bnd[:-1]),
                0,
                self.phn_max_length
            )
            self.train_seq_length[idx] = len(bnd) - 1

    def process_feat(self, feature):
        self.feat_dim = feature[0].shape[-1]
        self.source_data = np.zeros(
            shape=[
                self.data_length,
                self.feat_max_length,
                self.feat_dim * self.concat_window
            ],
            dtype='float32'
        )
        self.source_data_length = np.zeros(
            shape=[self.data_length],
            dtype='int32'
        )

        for idx, feat in enumerate(feature):
            for l in range(len(feat)):
                half_window = int((self.concat_window - 1) / 2)
                if l < half_window:
                    pad_feat = np.tile(feat[0], (half_window - l, 1))
                    concat_feat = np.concatenate(
                        [pad_feat, feat[0:l + half_window + 1]], axis=0)
                elif l > len(feat) - half_window - 1:
                    pad_feat = np.tile(feat[-1],
                                       (half_window - (len(feat) - l - 1), 1))
                    concat_feat = np.concatenate(
                        [feat[l - half_window:len(feat)], pad_feat], axis=0)
                else:
                    concat_feat = feat[l - half_window:l + half_window + 1]

                self.source_data[idx][l] = np.reshape(concat_feat, [-1])
            self.source_data_length[idx] = len(feat)

    def process_label(self, oracle_bound, phoneme):
        self.frame_label = np.zeros(
            shape=[self.data_length, self.feat_max_length], dtype='int32')
        self.orc_bnd = np.array(oracle_bound)
        self.phn_label = phoneme

        for idx, bnd, phn in zip(range(self.data_length), oracle_bound,
                                 phoneme):
            assert (len(bnd) == len(phn) + 1)
            prev_b = 0
            for b, p in zip(bnd[1:], phn):
                self.frame_label[idx][prev_b:b] = np.array(
                    [self.phn2idx[p]] * (b - prev_b))
                prev_b = b
            self.frame_label[idx][b] = self.phn2idx[p]

    def process_target(self, target_path):
        target_data = [line.strip().split() for line in open(target_path, 'r')]
        self.target_data_length = len(target_data)
        self.target_data = np.zeros(
            shape=[self.target_data_length, self.phn_max_length], dtype='int32')
        self.target_length = np.zeros(shape=[self.target_data_length],
                                      dtype='int32')

        for idx, target in enumerate(target_data):
            self.target_data[idx][:len(target)] = np.array(
                [self.phn2idx[t] for t in target])
            self.target_length[idx] = len(target)

    def print_parameter(self, target=False):
        print('Data Loader Parameter:')
        print(f'   phoneme number:  {self.phn_size}')
        print(f'   phoneme length:  {self.phn_max_length}')
        print(f'   feature dim:     {self.feat_dim * self.concat_window}')
        print(f'   feature windows: {self.concat_window}')
        print(f'   feature length:  {self.feat_max_length}')
        print(f'   source size:     {self.data_length}')
        if target:
            print(f'   target size:     {self.target_data_length}')
        print(f'   feat_path:       {self.feat_path}')
        print(f'   phn_path:        {self.phn_path}')
        print(f'   orc_bnd_path:    {self.orc_bnd_path}')
        print(f'   train_bnd_path:  {self.train_bnd_path}')
        print(f'   target_path:     {self.target_path}')
        print('=' * 80)

    def get_sample_batch(self, batch_size, repeat=1):
        # batch_size = batch_size // 2
        # batch_idx = np.random.choice(
        #     self.data_length,
        #     batch_size,
        #     replace=False
        # )
        # batch_idx = np.tile(batch_idx, (repeat))
        # random_pick = np.clip(
        #     np.random.normal(
        #         0.5,
        #         0.2,
        #         [batch_size * 2 * repeat, self.phn_max_length]
        #     ),
        #     0.0,
        #     1.0
        # )
        # # For every S_i take starting bound and with
        # # random_pick sample which x_j should be taken
        #
        # # We have no 6 times same utterance with diffrent choosen y_j
        # sample_frame = np.around(
        #     np.tile(self.train_bnd[batch_idx], (2, 1))
        #     + random_pick * np.tile(self.train_bnd_range[batch_idx], (2, 1))
        # ).astype('int32')
        #
        # sample_source = np.tile(
        #     self.source_data[batch_idx],
        #     (2, 1, 1)
        # )[
        #     np.arange(batch_size * 2 * repeat).reshape([-1, 1]),
        #     sample_frame
        # ]
        # # repeat_num: How often "picked y_j" are the same for first
        # # and second part of the batch. We will use it in segment loss
        # repeat_num = np.sum(np.not_equal(
        #     sample_frame[:batch_size * repeat],
        #     sample_frame[batch_size * repeat:]
        # ).astype(np.int32), axis=1)
        # lenghts = np.tile(
        #         self.train_seq_length[batch_idx],
        #         (2)
        #     )
        #
        # print('fake_shape', sample_source.shape)
        batch = next(self.batches)
        sample_source = batch['features']
        lenghts = batch['features_len']
        repeat_num = np.full(batch_size, 0.0)
        return (
            sample_source,
            lenghts,
            repeat_num
        )

    def get_target_batch(self, batch_size):
        # batch_idx = np.random.choice(self.target_data_length, batch_size,
        #                              replace=False)
        # sample_source, lenghts = self.target_data[batch_idx],
        # self.target_length[batch_idx]
        batch = next(self.batches)
        sample_source = batch['alignment']
        lenghts = batch['alignment_len']
        return sample_source, lenghts

    def get_aug_target_batch(self, batch_size):
        batch_idx = np.random.choice(self.target_data_length, batch_size,
                                     replace=False)
        batch_target_data = np.zeros(shape=[batch_size, self.phn_max_length],
                                     dtype='int32')
        batch_target_length = np.zeros(shape=[batch_size], dtype='int32')
        for i, (seq, length) in enumerate(
            zip(self.target_data[batch_idx], self.target_length[batch_idx])):
            new_seq, new_legnth = self.data_augmentation(seq, length)
            if new_legnth > self.phn_max_length:
                new_legnth = self.phn_max_length
            batch_target_data[i][:new_legnth] = new_seq[:new_legnth]
            batch_target_length[i] = new_legnth
        return batch_target_data, batch_target_length

    def data_augmentation(self, seq, length):
        new_seq = []
        for s in seq[:length]:
            if s == self.sil_idx:
                # new_seq.extend([s]*np.random.choice([0, 1, 2], p=[0.04,
                # 0.8, 0.16]))
                new_seq.extend([s])
            else:
                new_seq.extend([s] * np.random.choice([0, 1, 2, 3],
                                                      p=[0.04, 0.78, 0.17,
                                                          0.01]))
        return np.array(new_seq), len(new_seq)

    def generate_batch_number(self, batch_size):
        self.batch_number = (self.data_length - 1) // batch_size #+1

    def reset_batch_pointer(self):
        self.pointer = 0

    def update_pointer(self, batch_size):
        self.pointer += batch_size

    def get_batch(self, batch_size):
        # self.generate_batch_number(batch_size)
        # self.reset_batch_pointer()

        for batch in self.distsup_dataloader:
            # batch_source = self.source_data[
            # self.pointer:self.pointer + batch_size]
            # batch_frame_label = self.frame_label[
            # self.pointer:self.pointer + batch_size]
            # batch_source_length = self.source_data_length[
            # self.pointer:self.pointer + batch_size]
            # self.update_pointer(batch_size)
            # yield batch_source, batch_frame_label, batch_source_length
            batch = {k: t.numpy() for k,t in batch.items()}
            batch_source = batch['features']
            batch_frame_label = batch['alignment']
            batch_source_length = batch['alignment_len']

            # self.update_pointer(batch_size)
            yield batch_source, batch_frame_label, batch_source_length
