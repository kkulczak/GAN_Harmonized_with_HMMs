import logging
import os
import numpy as np
import torch
from torch.nn.functional import one_hot
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from nltk.corpus import masc_tagged
import nltk

GAP_CHARACTER = '\0'


class AmericanNationalCorpusDataset(Dataset):
    """
        The Open American National Corpus Dataset
    """
    source_data = None

    def __init__(
        self,
        config,
        transform_raw_phrase: callable = None,
        transform_sample_dict: callable = None
    ):
        # Download corpus
        if os.path.join(os.getcwd(), 'raw_phrases') not in nltk.data.path:
            nltk.data.path.append(os.path.join(os.getcwd(), 'raw_phrases'))
        try:
            nltk.data.find('corpora/masc_tagged.zip')
        except LookupError as e:
            logging.info(
                'Corpus not found. Downloading american national corpus'
            )
            nltk.download(
                'masc_tagged',
                download_dir='raw_phrases',
                quiet=True,
            )

        self.phrase_length = config['phrase_length']
        self.concat_window = config['concat_window']
        self.ascii_size = config['ascii_size']
        self.transform_raw_phrase = transform_raw_phrase
        self.transform_sample_dict = transform_sample_dict
        self.raw_phrases = [
            processed
            for phrase in masc_tagged.sents()[:config['dataset_size']]
            if len(phrase) > 4 and len(phrase[0]) < 30
            for processed in [self.preprocess_phrase(phrase)]
            if processed is not None
        ]

        self.raw_phrases = np.array(
            self.raw_phrases
        )

        self.raw_phrases = one_hot(
            torch.from_numpy(self.raw_phrases),
            num_classes=self.ascii_size
        ).numpy()

    def build_windowed_phrase(self, feat):
        windowed_phrase = np.zeros(
            shape=[
                # self.data.shape[0],
                self.phrase_length,
                self.raw_phrases.shape[-1] * self.concat_window
            ],
            dtype=np.float32
        )

        for l in range(len(feat)):
            half_window = int((self.concat_window - 1) / 2)
            if l < half_window:
                pad_feat = np.tile(feat[0], (half_window - l, 1))
                concat_feat = np.concatenate(
                    [pad_feat, feat[0:l + half_window + 1]], axis=0
                )
            elif l > len(feat) - half_window - 1:
                pad_feat = np.tile(
                    feat[-1],
                    (half_window - (len(feat) - l - 1), 1)
                )
                concat_feat = np.concatenate(
                    [feat[l - half_window:len(feat)], pad_feat], axis=0
                )
            else:
                concat_feat = feat[l - half_window:l + half_window + 1]

            windowed_phrase[l] = np.reshape(concat_feat, [-1])
        return windowed_phrase

    def preprocess_phrase(self, phrase):
        lengths = np.array([len(x) for x in phrase])
        lengths[1:] += 1
        cum_sum = np.cumsum(lengths)
        last_word_id = np.searchsorted(
            cum_sum > self.phrase_length,
            1,
            side='left'
        )
        processed = ' '.join(phrase[:last_word_id]).lower()

        _ascii = np.array([int(ord(x)) for x in processed], dtype=int)

        if (_ascii > self.ascii_size).any():
            return None
        if (_ascii == ord(GAP_CHARACTER)).any():
            return None

        _ascii = np.concatenate(
            (_ascii,
            ord(GAP_CHARACTER) * np.ones(
                self.phrase_length - _ascii.size,
                dtype=int)
            )
        )

        if (_ascii == ord(GAP_CHARACTER)).all():
            breakpoint()

        return _ascii

    def __len__(self):
        return self.raw_phrases.shape[0]

    def __getitem__(self, item):
        sample = self.raw_phrases[item]
        if self.transform_raw_phrase is not None:
            sample = self.transform_raw_phrase(sample)

        sample_dict = {
            'raw_phrase':    sample,
            'concat_phrase': self.build_windowed_phrase(
                sample
            ),
        }
        if self.transform_sample_dict is not None:
            sample_dict = self.transform_sample_dict(sample_dict)

        return sample_dict

    def show(self, sample):
        if isinstance(sample, dict):
            xs = sample['raw_phrase'].numpy()
        elif isinstance(sample, np.ndarray):
            xs = sample
        elif isinstance(sample, torch.Tensor):
            xs = sample.numpy()
        else:
            raise ValueError('sample must be dict or array or torch.Tensor')
        xs = xs.reshape(-1, self.ascii_size)
        return ''.join(
            chr(x) if x != ord(GAP_CHARACTER) else "_"
                for x in np.argmax(xs, axis=1)
        )


class ObliterateLetters(object):
    def __init__(self, obliterate_ratio: float):
        self.obliterate_ratio = obliterate_ratio

    def __call__(self, sample):
        ids = np.random.uniform(size=sample.shape[:-1]) < self.obliterate_ratio
        one_hot_gap_character = np.zeros(sample.shape[-1], dtype=sample.dtype)
        one_hot_gap_character[ord(GAP_CHARACTER)] = 1
        sample[ids, :] = one_hot_gap_character
        return sample


class ToTensor(object):
    def __call__(self, sample_dict):
        return {k: torch.from_numpy(v) for k, v in sample_dict.items()}


class BatchSampler:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(data_loader)

    def sample_batch(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            return next(self.iter)


def create_data_samplers(config):
    noisy_phrases = AmericanNationalCorpusDataset(
        config,
        transform_raw_phrase=ObliterateLetters(
            obliterate_ratio=config['replace_with_noise_probability']
        ),
        # transform_sample_dict=ToTensor()
    )
    real_phrases = AmericanNationalCorpusDataset(
        config,
        transform_raw_phrase=None,
        # transform_sample_dict=ToTensor()
    )

    logging.info(f'Dataset_size: {len(noisy_phrases)}')

    noisy_data_loader = DataLoader(
        noisy_phrases,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True,
        drop_last=True,
    )
    real_data_loader = DataLoader(
        real_phrases,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True,
        drop_last=True,
    )

    noisy_sampler = BatchSampler(noisy_data_loader)
    real_sampler = BatchSampler(real_data_loader)

    return noisy_sampler, real_sampler

def show_examples(args, config, device='cpu', shuffle=False):
    noisy_phrases = AmericanNationalCorpusDataset(
        config,
        transform_raw_phrase=ObliterateLetters(
            obliterate_ratio=config['replace_with_noise_probability']
        ),
        transform_sample_dict=ToTensor()
    )

    noisy_data_loader = DataLoader(
        noisy_phrases,
        batch_size=1,
        num_workers=1,
        shuffle=shuffle
    )

    with torch.no_grad():
        for x in itertools.islice(noisy_data_loader, 5):
            _input = x['concat_phrase'].to(device)
            out = generator.forward(_input).cpu()
            print('#' * 40)
            print(noisy_phrases.show(x['raw_phrase']))
            print(noisy_phrases.show(out))
            print('#' * 40)


def measure_accuracy(generator, real_data_loader, fake_data_loader, device):
    correct = 0
    elements = 0
    with torch.no_grad():
        for fake_batch, real_batch in tqdm(
            zip(fake_data_loader, real_data_loader)
        ):
            _input = fake_batch['concat_phrase'].to(device)
            output = generator.forward(_input)

            correct += np.sum(
                np.argmax(output.detach().cpu().numpy(), axis=-1)
                == np.argmax(real_batch['raw_phrase'].numpy(), axis=-1)
            )
            elements += reduce(
                operator.mul,
                real_batch['raw_phrase'].shape[:-1],
                1
            )
    # logging.debug(f'{correct} {elements} {correct / elements}')
    return correct / elements