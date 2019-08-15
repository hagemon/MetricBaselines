from torch.utils.data.sampler import Sampler
import numpy as np


class BalancedSampler(Sampler):

    def __init__(self, dataset, batch_size, n_instance):
        n_classes = batch_size // n_instance
        self.labels = dataset.labels
        self.labels_set = list(set(self.labels))
        self.label_indices = {
            label: np.where(self.labels == label)[0]
            for label in self.labels_set
        }
        for label in self.label_indices:
            np.random.shuffle(self.label_indices[label])
        self.used_ind_pos = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_instance
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes
        super(BalancedSampler, self).__init__(dataset)

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for c in classes:
                c_indices = self.label_indices[c]
                pos = self.used_ind_pos[c]
                indices.extend(c_indices[pos:pos+self.n_samples])
                self.used_ind_pos[c] += self.n_samples
                if self.used_ind_pos[c] > len(self.label_indices[c]):
                    np.random.shuffle(self.label_indices[c])
                    self.used_ind_pos[c] = 0
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return len(self.dataset) // self.batch_size


class ClassMiningSampler(Sampler):

    def __init__(self, dataset, batch_size, n_instance, balanced=False):
        n_classes = batch_size // n_instance
        self.labels = dataset.labels
        self.labels_set = list(set(self.labels))
        self.label_indices = {
            label: np.where(self.labels == label)[0]
            for label in self.labels_set
        }
        for label in self.label_indices:
            np.random.shuffle(self.label_indices[label])
        self.used_ind_pos = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_instance
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes
        self.centers = []
        self.dist = [0]
        self.dist_rank = [0]
        self.balanced = balanced
        super(ClassMiningSampler, self).__init__(dataset)

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            if len(self.centers) == 0:
                classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            else:
                # label set start from 1 while dist_rank start from 0
                c = np.random.choice(self.labels_set, 1)[0]
                classes = self.dist_rank[c-1][:self.n_classes]+1
            indices = []
            if self.balanced:
                for c in classes:
                    c_indices = self.label_indices[c]
                    pos = self.used_ind_pos[c]
                    indices.extend(c_indices[pos:pos + self.n_samples])
                    self.used_ind_pos[c] += self.n_samples
                    if self.used_ind_pos[c] > len(self.label_indices[c]):
                        np.random.shuffle(self.label_indices[c])
                        self.used_ind_pos[c] = 0
                yield indices
                self.count += self.batch_size
            else:
                indices = np.random.choice(np.where(np.isin(self.labels, classes))[0], self.batch_size, replace=False)
                yield indices
                self.count += self.batch_size

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def update_centers(self, centers, epoch):
        print('update centers in epoch %d' % epoch)
        self.centers = centers.cpu().numpy()
        self.dist = np.sum((np.expand_dims(self.centers, 1)-np.expand_dims(self.centers, 0))**2, axis=2)
        self.dist_rank = np.argsort(self.dist)
