import abc
import random
from itertools import permutations
from typing import Set

class AbstractDataset(abc.ABC):
    def __init__(self, group_elements1: Set, group_elements2: Set, frac_train: float, frac_mislabeled: float):
        self.frac_train = frac_train
        self.group_elements1 = group_elements1
        self.group_elements2 = group_elements2
        self.ordered_group_elements1 = list(self.group_elements1)
        self.ordered_group_elements2 = list(self.group_elements2)
        self.idx2vocab = ['o', '='] + list(group_elements1.union(group_elements2))
        self.vocab2idx = {vocab: idx for idx, vocab in enumerate(self.idx2vocab)}
        self.n_vocab = len(self.idx2vocab)
        self.n_out = len(group_elements1.union(group_elements2))
        idxs = list(range(len(self.group_elements1)*len(self.group_elements2)))
        random.shuffle(idxs)
        valcutoff = int(len(idxs)*frac_train)
        self.train_pairs, self.val_pairs = idxs[:valcutoff], idxs[valcutoff:]
        self.n_mislabeled = int(len(self.train_pairs)*frac_mislabeled)
        
        elements = list(group_elements1.union(group_elements2))
        self.mislabeling = [random.choice(elements) for _ in range(len(idxs))]
    
    @abc.abstractmethod
    def fetch_output(self, a, b):
        pass

    def encode(self, sequence):
        return [self.vocab2idx[item] for item in sequence]
    
    def decode(self, sequence):
        return [self.idx2vocab[item] for item in sequence]
    
    def form_equation(self, a, b, c):
        return [a, 'o', b, '=', c]
    
    def fetch_random(self, idx):
        return self.mislabeling[idx]
    
    def fetch_example(self, idx, mislabel=False):
        a = self.ordered_group_elements1[idx // len(self.group_elements2)]
        b = self.ordered_group_elements2[idx % len(self.group_elements2)]
        if mislabel:
            c = self.mislabeling[idx]
        else:
            c = self.fetch_output(a, b)
        equation = self.form_equation(a, b, c)
        return self.encode(equation[:-1]), (self.vocab2idx[c]-2), equation
    
    def fetch_train_example(self):
        i = random.choice(range(len(self.train_pairs)))
        idx = self.train_pairs[i]
        return self.fetch_example(idx, mislabel=i < self.n_mislabeled)
    
    def fetch_ground_truth_example(self):
        i = random.choice(range(self.n_mislabeled))
        idx = self.train_pairs[i]
        return self.fetch_example(idx, mislabel= False)

    def fetch_val_example(self):
        idx = random.choice(self.val_pairs)
        return self.fetch_example(idx)

class ModSumDataset(AbstractDataset):
    def __init__(self, p, frac_train, frac_mislabeled):
        super(ModSumDataset, self).__init__(set(range(p)), set(range(p)), frac_train, frac_mislabeled)
        self.p = p
    
    def fetch_output(self, a, b):
        return (a + b) % self.p

class ModSubtractDataset(AbstractDataset):
    def __init__(self, p, frac_train, frac_mislabeled):
        super(ModSubtractDataset, self).__init__(set(range(p)), set(range(p)), frac_train, frac_mislabeled)
        self.p = p
    
    def fetch_output(self, a, b):
        return (a - b) % self.p

class ModDivisonDataset(AbstractDataset):
    def __init__(self, p, frac_train, frac_mislabeled):
        super(ModDivisonDataset, self).__init__(set(range(p)), set(range(1, p)), frac_train, frac_mislabeled)
        self.p = p
    
    def fetch_output(self, a, b):
        return (a * pow(b, self.p-2, self.p)) % self.p

class PermutationGroup(AbstractDataset):
    def __init__(self, k, frac_train, frac_mislabeled):
        perms = set(map(tuple, permutations(list(range(k)))))
        super(PermutationGroup, self).__init__(perms, perms, frac_train, frac_mislabeled)
        self.k = k

    def fetch_output(self, a, b):
        return tuple([a[b[i]] for i in range(len(b))])