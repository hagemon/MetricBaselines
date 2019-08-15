import os
import time
import json
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from dataset import get_datasets, get_data_loaders
from sampler import ClassMiningSampler
from test import test
from model import get_model
from transforms import get_transform

from losses.get_loss import class_centers, get_loss


class Trainer:
    def __init__(self, args):

        # Training configurations
        self.method = args.method
        self.dataset = args.dataset
        self.dim = args.dim
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.val_batch_size = self.batch_size // 2
        self.iteration = args.iteration
        self.evaluation = args.evaluation
        self.show_iter = 1
        self.update_epoch = 10
        self.balanced = args.balanced
        self.instances = args.instances
        self.cm = args.cm
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.file_name = '{}_{}_{}'.format(
            self.method,
            self.dataset,
            self.lr,
        )
        print('========================================')
        print(json.dumps(vars(args), indent=2))
        print(self.file_name)

        # Paths

        self.root_dir = os.path.join('/', 'home', 'lyz')
        self.data_dir = os.path.join(self.root_dir, 'datasets', self.dataset)
        self.model_dir = self._get_path('./trained_model')
        self.code_dir = self._get_path(os.path.join('codes', self.dataset))
        self.fig_dir = self._get_path(os.path.join('fig', self.dataset, self.file_name))

        # Preparing data
        self.transforms = get_transform()
        self.datasets = get_datasets(dataset=self.dataset, data_dir=self.data_dir, transforms=self.transforms)
        self.cm_sampler = ClassMiningSampler(
            self.datasets['train'],
            batch_size=self.batch_size,
            n_instance=self.instances,
            balanced=self.balanced)
        self.data_loaders = get_data_loaders(
            datasets=self.datasets,
            batch_size=self.batch_size,
            val_batch_size=self.val_batch_size,
            n_instance=self.instances,
            balanced=self.balanced,
            cm=self.cm_sampler if self.cm else None
        )
        self.dataset_sizes = {x: len(self.datasets[x]) for x in ['train', 'test']}

        # Set up model
        self.model = get_model(self.device, self.dim)

        self.optimizer = optim.SGD(
            [
                {'params': self.model.google_net.parameters()},
                {'params': self.model.linear.parameters(), 'lr': self.lr * 10, 'momentum': 0.9}
            ],
            lr=self.lr, momentum=0.9
        )
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.5)

    @staticmethod
    def _get_path(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path

    def train(self):
        since = time.time()
        start = time.time()

        self.scheduler.step()
        self.model.train()

        running_iter = 0
        running_loss = 0.0
        running_count = 0
        running_epoch = 0
        print('Start training')
        while running_iter < self.iteration:

            if self.cm and running_epoch % self.update_epoch == 0:
                embeddings, labels, spend = self.feed_embeddings('mean')
                centers = class_centers(embeddings, labels)
                self.cm_sampler.update_centers(centers, running_epoch)
                start += spend

            # Train an epoch
            for sample in self.data_loaders['train']:
                inputs = sample['image'].to(self.device)
                labels = sample['label'].to(self.device)
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs)
                    loss_fn = get_loss(self.method)
                    loss, count = loss_fn(outputs, labels)

                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() / count
                running_count += count
                if (running_iter + 1) % self.show_iter == 0:
                    print('Iteration {}/{} Loss {:.4f} Triplets: {:.0f} Spending {:.0f}s'.format(
                        running_iter + 1,
                        self.iteration,
                        running_loss / self.show_iter,
                        running_count / self.show_iter,
                        time.time() - start
                    ))
                    running_loss = 0.0
                    running_count = 0
                    start = time.time()

                running_iter += 1
            running_epoch += 1

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        return self.model

    def feed_embeddings(self, dataset, numpy=False):
        code = torch.FloatTensor().to(self.device)
        label = torch.LongTensor()
        start = time.time()
        for sample in self.data_loaders[dataset]:
            inputs = sample['image'].to(self.device)
            labels = sample['label']
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
            code = torch.cat((code, outputs), 0)
            label = torch.cat((label, labels.long()), 0)
        end = time.time()
        print('Finish generating {} {} codes in {:.0f}s'.format(
            len(code),
            dataset,
            end - start))
        if numpy:
            code = code.cpu().numpy()
            label = label.cpu().numpy()
        return code, label, end-start

    def generate_codes(self):
        self.model.eval()
        test_feature, test_label, _ = self.feed_embeddings('test', numpy=True)
        np.save(os.path.join(self.code_dir, 'test_ft.npy'), test_feature)
        np.save(os.path.join(self.code_dir, 'test_labels.npy'), test_label)
        return test_feature, test_label

    def save_model(self):
        model_path = os.path.join(self.model_dir, self.file_name+'.pkl')
        torch.save(self.model.state_dict(), model_path)

    def load_model(self):
        model_path = os.path.join(self.model_dir, self.file_name+'.pkl')
        if not os.path.exists(model_path):
            raise Exception('Can not find trained model {}'.format(model_path))
        self.model.load_state_dict(torch.load(model_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Metric Baselines')
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--iteration', default=20000, type=int)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--batch-size', default=120, type=int)
    parser.add_argument('--dataset', default='CUB', type=str)
    parser.add_argument('-e', '--evaluation', dest='evaluation', action='store_true')
    parser.add_argument('--method', default='Triplet', type=str)
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--instances', default=4, type=int)
    parser.add_argument('--cm', action='store_true')

    trainer = Trainer(parser.parse_args())

    if not trainer.evaluation:
        trainer.train()
        trainer.save_model()
    else:
        trainer.load_model()
    test_ft, test_labels = trainer.generate_codes()
    test(trainer.file_name, test_ft, test_labels)
