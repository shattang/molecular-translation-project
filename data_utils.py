import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL
from torchvision import transforms
import torch
import torch.utils.data as torchdata
import logging
import torchvision.transforms.functional as F

import tokenizer
import vocabulary


def tokenize_back_groups(s):
    ret = []
    if s is None:
        return ret
    curr_token = ''
    for i in range(len(s)):
        c = s[i]
        if c == '/':
            if len(curr_token) > 0:
                ret.append(curr_token)
                curr_token = ''
            curr_token += c
            continue

        if curr_token == '/':
            curr_token += c
            ret.append(curr_token)
            curr_token = ''
            continue

        if c.isdigit():
            curr_token += c
        else:
            if len(curr_token) > 0:
                ret.append(curr_token)
                curr_token = ''
                ret.append(c)
            else:
                ret.append(c)

    if len(curr_token) > 0:
        ret.append(curr_token)
    return ret


def preprocess_raw_data(data_dir):
    def flatten(l):
        return [y for x in l for y in x]

    mt = tokenizer.MoleculeTokenizer()

    def get_group1_tokens(s):
        from_ix = s.find('/') + 1
        to_ix = s.find('/', from_ix)
        l = mt.tokenize_formula(s[from_ix:to_ix])
        l = flatten(mt.to_count_form(l))
        return ' '.join(str(x) for x in l)

    def get_groupn_tokens(s):
        back_grps = s[s.find('/', s.find('/')+1):]
        return ' '.join(tokenize_back_groups(back_grps))

    out_path = os.path.join(data_dir, 'raw', 'train', 'processed_labels.h5')
    if os.path.exists(out_path):
        logging.info('Processed labels already exist: ' + out_path)
        return

    path = os.path.join(data_dir, 'raw', 'train', '_labels.csv')
    logging.info('Preprocessing raw labels file: ' + path)
    df = pd.read_csv(path)
    df['NumGroups'] = df.InChI.str.count('/') + 1
    df['Group1Tokens'] = df.InChI.map(get_group1_tokens)
    df['GroupNTokens'] = df.InChI.map(get_groupn_tokens)

    logging.info('Writing processed labels file: ' + out_path)
    df.to_hdf(out_path, 'labels', complevel=9)
    return df


class BmsDataContext(object):
    def __init__(self,
                 input_dir,
                 data_dir,
                 context_name,
                 context_random_seed,  # random seed to choose subset
                 target_sentence_col=None,
                 # func(dataframe) -> list(list(word))
                 target_sentence_fn=None,
                 num_train=1000,  # size of train set
                 num_val=100,  # size of validation set
                 num_test=100,  # size of test set
                 group_filters=None,  # filter for rows with specific num of inchi groups
                 image_size=(224, 224)) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.data_dir = data_dir
        self.context_name = context_name
        self.context_random_seed = context_random_seed
        self.target_sentence_col = target_sentence_col
        self.target_sentence_fn = target_sentence_fn
        self.random_state = np.random.RandomState(context_random_seed)
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.group_filters = set(group_filters or [])
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.vocab = None

        self.preprocess = transforms.Compose([
            transforms.Resize(
                size=image_size, interpolation=F.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])
        self.profile_dir = os.path.join(self.data_dir, self.context_name)

        if self.target_sentence_fn is None:
            def target_sentence_fn(df):
                return list(df[self.target_sentence_col].str.split())
            self.target_sentence_fn = target_sentence_fn

    def initialize(self):
        data_path = os.path.join(self.profile_dir, 'data.h5')

        if os.path.exists(data_path):
            logging.info(
                'Loading existing context data from: ' + self.profile_dir)
            self.train_df = pd.read_hdf(data_path, 'train_df')
            self.val_df = pd.read_hdf(data_path, 'val_df')
            self.test_df = pd.read_hdf(data_path, 'test_df')
            self.vocab = vocabulary.Vocabulary()
            self.vocab.from_dataframe(pd.read_hdf(data_path, 'vocab_df'))
            logging.info('Loaded train={}, val={}, test={}, vocab={}'.format(
                len(self.train_df), len(self.val_df), len(self.test_df), len(self.vocab.get_all_words())))
            return

        if os.path.exists(self.profile_dir):
            logging.info(
                'Preparing context data from specified image ids: ' + self.profile_dir)
            train_image_ids = set(pd.read_csv(os.path.join(
                self.profile_dir, 'train_image_ids.csv')).image_id)
            val_image_ids = set(pd.read_csv(os.path.join(
                self.profile_dir, 'val_image_ids.csv')).image_id)
            test_image_ids = set(pd.read_csv(os.path.join(
                self.profile_dir, 'test_image_ids.csv')).image_id)
            self.num_train = len(train_image_ids)
            self.num_val = len(val_image_ids)
            self.num_test = len(test_image_ids)
            # no filters since image ids were specified manually
            self.group_filters = set([])
            input_df = self._get_input_df()
            train_input_df = input_df[input_df.image_id.isin(train_image_ids)]
            val_input_df = input_df[input_df.image_id.isin(val_image_ids)]
            test_input_df = input_df[input_df.image_id.isin(test_image_ids)]
        else:
            logging.info(
                'Preparing context data using random sampling: ' + self.profile_dir)
            input_df = self._get_input_df()
            num_sample = self.num_train + self.num_val + self.num_test
            locs = self.random_state.choice(
                np.arange(len(input_df)), num_sample, replace=False)
            input_df = input_df.iloc[locs].reset_index(drop=True)
            train_input_df = input_df[0:self.num_train]
            val_input_df = input_df[self.num_train:self.num_train+self.num_val]
            test_input_df = input_df[self.num_train+self.num_val:num_sample]

        del(input_df)

        dataframes = {}
        for df_type, df in zip(['train', 'val', 'test'], [train_input_df, val_input_df, test_input_df]):
            image_ids = list(df.image_id.values)
            sentences = self.target_sentence_fn(df)
            dataframes[df_type] = pd.DataFrame({
                'ImageId': image_ids,
                'Sentence': sentences
            })

        train_df = dataframes['train']
        vocab = vocabulary.Vocabulary()

        def captionize(sentence):
            caption = [vocab.get_start_word()]
            caption.extend(vocab.add_word(word) for word in sentence)
            caption.append(vocab.get_end_word())
            return caption

        captions = []
        for sentence in train_df['Sentence']:
            captions.append(captionize(sentence))
        train_df['Caption'] = captions

        for df in dataframes.values():
            if not 'Caption' in df.columns:
                df['Caption'] = [vocab.encode_sentence(s)
                                 for s in df['Sentence']]

        self.vocab = vocab
        self.train_df = dataframes['train']
        self.val_df = dataframes['val']
        self.test_df = dataframes['test']

        self._save_hdf()
        logging.info('Context data created: ' + self.profile_dir)
        logging.info('Loaded train={}, val={}, test={}, vocab={}'.format(
            len(self.train_df), len(self.val_df), len(self.test_df), len(self.vocab.get_all_words())))

    def _save_hdf(self):
        if not os.path.exists(self.profile_dir):
            os.makedirs(self.profile_dir)
        data_path = os.path.join(self.profile_dir, 'data.h5')
        store = pd.HDFStore(data_path, mode='w')
        store['train_df'] = self.train_df
        store['val_df'] = self.val_df
        store['test_df'] = self.test_df
        store['vocab_df'] = self.vocab.to_dataframe()
        logging.info('Saved: {}'.format(store.info()))
        store.flush()
        store.close()

    def _get_input_df(self):
        input_path = os.path.join(
            self.input_dir, 'raw', 'train', 'processed_labels.h5')
        input_df = pd.read_hdf(input_path)
        if len(self.group_filters) > 0:
            logging.info('Filtering for groups: {}'.format(
                str(self.group_filters)))
            input_df = input_df[input_df.NumGroups.isin(self.group_filters)]
        return input_df

    def get_df(self, dataset_type):
        if dataset_type == 'test':
            return self.test_df
        elif dataset_type == 'val':
            return self.val_df
        elif dataset_type == 'train':
            return self.train_df
        else:
            raise Exception('invalid dataset type: {}'.format(dataset_type))

    def load_input_image(self, image_id):
        path = os.path.join(self.data_dir, 'raw', 'train',
                            *image_id[0:3], image_id + '.png')
        img = PIL.Image.open(path).convert('RGB')
        img = PIL.ImageOps.invert(img)
        return img


class BmsDataset(torchdata.Dataset):
    def __init__(self, context,
                 mode='train',  # train/val/test
                 target_column='Caption',
                 source_column='ImageId',
                 preload_images=False,
                 target_transform=None,  # func(dataset, target) -> tarnsformed
                 image_transform=None,  # func(dataset, image) -> transformed
                 sample_size=None
                 ) -> None:
        super().__init__()
        self.context = context  # type: BmsDataContext
        self.mode = mode
        self.source_column = source_column
        self.target_column = target_column
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.preload_images = preload_images
        self.preloaded_images = []
        self.sample_size=sample_size

    def initialize(self):
        df = self.context.get_df(self.mode)
        if not self.sample_size is None:
            logging.info('Sampling dataset for {} rows'.format(self.sample_size))
            df = df.sample(self.sample_size, random_state=self.context.random_state)

        if self.image_transform is None:
            self.image_transform = lambda ds, img: ds.context.preprocess(img)
        if self.target_transform is None:
            self.target_transform = lambda ds, tgt: tgt
        self.image_ids = df[self.source_column].values
        target_vals = df[self.target_column].values
        self.max_target_len = max(map(len, target_vals))
        self.target_vals = torch.stack(list(map(lambda x: self.target_transform(
            self, x), target_vals)))

        if self.preload_images:
            logging.info('Preloading {} images'.format(len(self.image_ids)))
            for i in range(len(self.image_ids)):
                self.preloaded_images.append(self.get_image(i))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        if self.preload_images:
            transformed_image = self.preloaded_images[index]
        else:
            transformed_image = self.get_image(index)
        target_val = self.target_vals[index]
        return transformed_image, target_val, index

    def get_image(self, index):
        image_id = self.image_ids[index]
        image = self.context.load_input_image(image_id)
        transformed_image = self.image_transform(self, image)
        return transformed_image


if __name__ == '__main__':
    # test code
    logging.getLogger().setLevel(logging.INFO)

    data_dir = '../data/bms'
    target_column = 'Group1Tokens'

    preprocess_raw_data(data_dir)

    context = BmsDataContext(context_name='test',
                             input_dir=data_dir,
                             data_dir=data_dir,
                             context_random_seed=100,
                             target_sentence_col=target_column,
                             group_filters=[4],
                             num_train=250,
                             num_val=50,
                             num_test=50)
    context.initialize()

    def transform_target(dataset, vals):
        max_len = dataset.max_target_len
        #import ipdb; ipdb.set_trace()
        padded = dataset.context.vocab.add_padding(vals, max_len)
        return torch.tensor(padded[0:max_len])

    for mode in ['train', 'val']:
        print('mode=' + mode)
        dataset = BmsDataset(context=context,
                             mode=mode,
                             target_transform=transform_target)
        dataset.initialize()
        print('target_vals', (dataset.target_vals).shape)
        print('image_ids', dataset.image_ids.shape)
        image, target, image_idx = dataset[0]
        print('image', image.shape)
        print('target', target)
        #import ipdb; ipdb.set_trace()
