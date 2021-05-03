# *Tasks for tomorrow April 24th
# Chose only inchis with 4 sections
# Take a training set of 10k images
# Validation set of 1k images
# Implement CNN with Attention and LSTM like image captioning
# Predict chemical formula (target=group1repeats)

import logging
import math
import numpy as np
import pandas as pd
import time
import os
import datetime
import torch
import torch.nn as nn
import torch.utils.data as torchdata
import torch.optim as optim
import torchvision.models as tvmodels
import tqdm
import editdistance as edd
import yaml
import glob

#from . import data_utils
import data_utils
from model1.encoder_decoder import EncoderDecoder


class ModelRunner:
    def __init__(self, input_dir, data_dir, context_name, settings) -> None:
        # Dont change any of defaults here, use settings to override
        # Since we only persist the overriden settings file
        self.settings = settings
        self.input_dir = input_dir
        self.data_dir = data_dir
        self.context_name = context_name
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = settings.get('device') or device
        # set this to false, unless using a small dataset
        self.preload_images = settings.get('preload_images') or False
        self.random_seed = settings.get('random_seed') or 489356152
        self.image_size = settings.get('image_size') or (224, 224)
        self.target_column = settings.get('target_column') or 'Group1Tokens'
        self.dataset_sample_size = settings.get('dataset_sample_size') or None

        self.num_epochs = settings.get('num_epochs') or 2
        self.save_per_batch_iter = settings.get('save_per_batch_iter') or 0
        self.train_batch_size = settings.get('train_batch_size') or 64
        self.eval_batch_size = settings.get('eval_batch_size') or 64
        self.num_data_loader_workers = settings.get(
            'num_data_loader_workers') or 0

        resnet_model = settings.get('resnet_model') or ''
        if resnet_model == 'resnet34':
            self.resnet_model = tvmodels.resnet34(pretrained=True)
        elif resnet_model == 'resnet50':
            self.resnet_model = tvmodels.resnet50(pretrained=True)
        else:
            self.resnet_model = None
        self.embed_size = settings.get('embed_size') or 256
        self.attention_dim = settings.get('attention_dim') or 256
        self.encoder_dim = settings.get('encoder_dim') or 512
        self.decoder_dim = settings.get('decoder_dim') or 512
        self.dropout_prob = settings.get('dropout_prob') or 0.3
        self.fine_tune = settings.get('fine_tune') or True
        self.learning_rate = settings.get('learning_rate') or 1e-4
        self.gradient_max_norm = settings.get('gradient_max_norm') or 1.0
        self.context = self.make_data_context()

    def make_data_context(self):
        context = data_utils.BmsDataContext(context_name=self.context_name,
                                            input_dir=self.input_dir,
                                            data_dir=self.data_dir,
                                            context_random_seed=self.random_seed,
                                            target_sentence_col=self.target_column,
                                            image_size=self.image_size)
        context.initialize()
        return context

    def make_dataset(self, context, mode):
        def target_transform(dataset, vals):
            max_len = dataset.max_target_len
            padded = dataset.context.vocab.add_padding(vals, max_len)
            return torch.tensor(padded[0:max_len])
        ds = data_utils.BmsDataset(context=context,
                                   mode=mode,
                                   target_transform=target_transform,
                                   preload_images=self.preload_images,
                                   sample_size=self.dataset_sample_size)
        ds.initialize()
        return ds

    def make_data_loader(self, dataset, batch_size):
        gen = torch.Generator(device='cpu')
        gen.manual_seed(self.random_seed)
        loader = torchdata.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.num_data_loader_workers,
            shuffle=True,
            generator=gen)
        return loader

    def make_model(self, dataset):
        vocab = dataset.context.vocab
        vocab_size = len(vocab.get_all_words())
        model = EncoderDecoder(
            embed_size=self.embed_size,
            vocab_size=vocab_size,
            attention_dim=self.attention_dim,
            encoder_dim=self.encoder_dim,
            decoder_dim=self.decoder_dim,
            drop_prob=self.dropout_prob,
            fine_tune=self.fine_tune,
            resnet_model=self.resnet_model
        ).to(self.device)
        criterion = nn.CrossEntropyLoss(ignore_index=vocab.get_pad_word())
        return model, criterion

    def evaluate(self, model_path, dataset_mode='val'):
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.info('Running evaluation: model={} dataset={} device={} ts={}'.format(
            model_path, dataset_mode, self.device, now))

        path_split = os.path.split(model_path)
        model_dir = path_split[0]
        model_filename = path_split[1]

        ncpu = os.cpu_count()
        if ncpu > 1:
            torch.set_num_threads(ncpu - 1)

        context = self.context
        dataset = self.make_dataset(context, dataset_mode)
        data_loader = self.make_data_loader(dataset, self.eval_batch_size)
        model, criterion = self.make_model(dataset)

        vocab = context.vocab
        vocab_size = len(vocab.get_all_words())
        start_caption = vocab.get_start_word()
        end_caption = vocab.get_end_word()
        pad_caption = vocab.get_pad_word()
        # since max_caption_len includes start caption
        max_len = dataset.max_target_len - 1

        total_loss = 0.0
        num_losses = 0.0
        all_dataset_idxs = []
        all_preds = []
        all_actuals = []
        all_accuracies = []

        logging.info(
            'Reading model for evaluation from: {}'.format(model_path))
        model.load_state_dict(torch.load(
            model_path, map_location=torch.device(self.device)))
        model.eval()  # set model in evaluation mode

        expected_iters = math.ceil(len(dataset)/data_loader.batch_size)
        for batch_idx, batch_input in tqdm.tqdm(enumerate(data_loader), total=expected_iters):
            images, captions, dataset_idxs = batch_input
            with torch.no_grad():
                images, captions = images.to(
                    self.device), captions.to(self.device)
                batch_output = model.predict(images, max_len, start_caption,
                                             end_caption, pad_caption)
                preds = batch_output[0]
                outputs = batch_output[1]
                #attentions = batch_output[2]
                actuals = captions[:, 1:]  # skip the start index
                loss = criterion(outputs.view(-1, vocab_size),
                                 actuals.reshape(-1))

            total_loss += loss.item()
            num_losses += 1

            # remove padding from both prediction and outputs
            preds = list(map(vocab.trim_padding, preds.tolist()))
            actuals = list(map(vocab.trim_padding, actuals.tolist()))
            all_dataset_idxs.extend(dataset_idxs.tolist())
            all_preds.extend(preds)
            all_actuals.extend(actuals)

        all_dataset_idxs = list(
            map(dataset.image_ids.__getitem__, all_dataset_idxs))
        all_accuracies = list(map(lambda x: edd.eval(
            *x), list(zip(all_actuals, all_preds))))

        # avg loss per example
        avg_loss = total_loss / num_losses if num_losses > 0 else 0
        results_df = pd.DataFrame({'ImageId': all_dataset_idxs,
                                   'Prediction': all_preds,
                                   'Actual': all_actuals,
                                   'Accuracy': all_accuracies})
        avg_acc = results_df.Accuracy.mean()
        logging.info("Evaluate model={}, dataset={}, avg_loss={}, avg_acc={}".format(
            model_path, dataset_mode, avg_loss, avg_acc))

        fname = 'eval_{}_{}_{}.store'.format(
            model_filename.replace('.', '_'), dataset_mode, now)
        fpath = os.path.join(model_dir, fname)
        store = pd.HDFStore(fpath, 'w')
        store['eval_info_df'] = pd.DataFrame({'Key': ['ModelPath', 'AvgLoss', 'AvgAcc', 'Dataset'],
                                              'Value': [model_path, avg_loss, avg_acc, dataset_mode]})
        store['results_df'] = results_df
        store.flush()
        store.close()
        logging.info('Saved evaluation {}'.format(fpath))
        return avg_loss, results_df

    def train(self, restore_model_path=None):
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = "{}_{}".format(self.settings['model_name'], now)
        model_dir = os.path.join(self.data_dir, self.context_name, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        logging.info('Running training: model={} device={} ts={}'.format(
            model_dir, self.device, now))

        model_settings_file = os.path.join(model_dir, 'settings.yaml')
        with open(model_settings_file, 'w') as fp:
            yaml.dump(self.settings, fp)

        ncpu = os.cpu_count()
        if ncpu > 1:
            torch.set_num_threads(ncpu - 1)

        mode = 'train'
        context = self.context
        dataset = self.make_dataset(context, mode)
        vocab = dataset.context.vocab
        vocab_size = len(vocab.get_all_words())
        data_loader = self.make_data_loader(dataset, self.train_batch_size)
        model, criterion = self.make_model(dataset)
        optimizer = optim.Adam(model.parameters(), self.learning_rate)

        if not restore_model_path is None:
            logging.info('Restoring model from: {}'.format(restore_model_path))
            model.load_state_dict(torch.load(restore_model_path))

        def save_epoch(epoch, batch_counter):
            fname = 'model_epoch_{}_{}_{}.pth'.format(
                epoch, batch_counter, now)
            fpath = os.path.join(model_dir, fname)
            logging.info('Saving epoch {}'.format(fpath))
            torch.save(model.state_dict(), fpath)

        #images, captions, img_idxs = next(iter(data_loader))
        batch_counter = 0
        for epoch in range(1, self.num_epochs+1):
            print("Start Epoch: {}".format(epoch))
            model.train()  # set model in train mode
            ts = time.time()
            total_loss = 0.0
            num_losses = 0.0
            expected_iters = math.ceil(len(dataset)/data_loader.batch_size)
            epoch_saved = 0
            for batch_idx, tup in tqdm.tqdm(enumerate(data_loader), total=expected_iters):
                # for i in range(1):
                images, captions, img_idxs = tup
                optimizer.zero_grad()
                images, captions = images.to(
                    self.device), captions.to(self.device)
                outputs = model(images, captions)
                loss = criterion(outputs.view(-1, vocab_size),
                                 captions[:, 1:].reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), max_norm=self.gradient_max_norm)
                optimizer.step()

                total_loss += loss.item()
                num_losses += 1
                batch_counter += 1

                save_batch = self.save_per_batch_iter > 0 and batch_counter % self.save_per_batch_iter == 0
                if save_batch:
                    save_epoch(epoch, batch_counter)
                    epoch_saved = 1

            if epoch_saved == 0:
                save_epoch(epoch, batch_counter)
                epoch_saved = 2

            avg_loss = total_loss/num_losses if num_losses > 0 else 0
            logging.info("Epoch: {} Loss: Total:{:.5f} Avg::{:.5f} Time={:.1f}".format(
                epoch, total_loss, avg_loss, (time.time() - ts)))

        if epoch_saved < 2:
            fname = 'model_final_{}_{}_{}.pth'.format(epoch, batch_counter, now)
            fpath = os.path.join(model_dir, fname)
            logging.info('Saving final model {}'.format(fpath))
            torch.save(model.state_dict(), fpath)
        return model


if __name__ == '__main__':
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=fmt)
    logging.getLogger().setLevel(logging.INFO)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train model', action='store_true')
    parser.add_argument('--evaluate', help='evaluate model',
                        action='store_true')
    parser.add_argument('--settings_path', help='settings yaml file path',
                        dest='settings_path', required=True)
    args = parser.parse_args()

    logging.info('Loading settings {}'.format(args.settings_path))
    with open(args.settings_path, 'r') as fp:
        settings = yaml.safe_load(fp)
    data_dir = settings['data_dir']
    input_dir = settings.get('input_dir') or data_dir
    context_name = settings['context_name']

    runner = ModelRunner(input_dir, data_dir, context_name, settings)
    # reproducability
    torch.manual_seed(runner.random_seed)

    if args.train:
        model = runner.train(settings.get('restore_model_path'))
    elif args.evaluate:
        eval_model_names = settings['eval_model_names']
        eval_dataset_types = settings['eval_dataset_types']
        for model_name in eval_model_names:
            glob_fil = os.path.join(
                data_dir, context_name, model_name, "model_*.pth")
            for model_path in glob.glob(glob_fil):
                for dataset_type in eval_dataset_types:
                    runner.evaluate(model_path, dataset_type)
    else:
        print('Need to specify either train or evaluate')
