import torch
import datasets
import models
from torch.utils.data import IterableDataset
from datasets import AbstractDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm.auto import tqdm

from utils import combine_logs

class GroupDataset(IterableDataset):
    def __init__(self, dataset: AbstractDataset, split: str):
        super(GroupDataset, self).__init__()
        assert split in {'train', 'val'}
        self.dataset = dataset
        self.split = split
        self.fetch_f = None
        if self.split == 'train':
            self.fetch_f = self.dataset.fetch_train_example
        elif self.split == 'val':
            self.fetch_f = self.dataset.fetch_val_example
        else:
            raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        x, y, _ = self.fetch_f()
        return torch.tensor(x), torch.tensor(y)

def train(config):
    print('using config:', config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = getattr(datasets,config['dataset'])(config["p"], config["frac_train"], config["frac_mislabeled"])
    train_data = GroupDataset(dataset, 'train')
    val_data = GroupDataset(dataset, 'val')
    model = getattr(models, config['model'])(config["transformer_config"], dataset.n_vocab, dataset.n_out, device)
    model.train()
    train_dataloader = DataLoader(train_data, num_workers=config['num_workers'], batch_size=config['bsize'])
    val_dataloader = DataLoader(val_data, num_workers=config['num_workers'], batch_size=config['bsize'])
    optim = torch.optim.AdamW(model.parameters(), lr=config['lr'], 
                              weight_decay=config['weight_decay'], 
                              betas=config['betas'])
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda s: min(s / config['warmup_steps'], 1))
    step = 0
    for x, y in tqdm(train_dataloader):
        loss, logs = model.get_loss(x.to(device), y.to(device))
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schedule.step()
        if (step+1) % config['eval_every'] == 0:
            model.eval()
            with torch.no_grad():
                all_val_logs = []
                for i, (val_x, val_y) in tqdm(enumerate(val_dataloader)):
                    if i >= config['eval_batches']:
                        break
                    _, val_logs = model.get_loss(val_x.to(device), val_y.to(device))
                    all_val_logs.append(val_logs)
            out_log = {'val': combine_logs(all_val_logs), 'train': combine_logs([logs]), 'step': (step+1), 
                       'lr': float(lr_schedule.get_last_lr()[0])}
            print(out_log)
            model.train()
        step += 1
        if config['max_steps'] is not None and step >= config['max_steps']:
            break

config = {
  'frac_train': 0.4,
  'frac_mislabeled' : 0.0,
  'p': 96,
  'transformer_config': {
    'pre_norm': True,
    'max_length': 5,
    'heads': 4,
    'hidden_dim': 128,
    'attn_dim': 32,
    'intermediate_dim': 512,
    'num_blocks': 2,
    'block_repeats': 1,
    'dropout': 0.1,
  },
  'num_workers': 0,
  'bsize': 512,
  'lr': 1e-3,
  'weight_decay': 0.0,
  'betas': [0.9, 0.98],
  'warmup_steps': 10,
  'eval_every': 10,
  'eval_batches': 8,
  'max_steps': 1e6,
  'dataset' : "ModSumDataset",
  "model" : "GrokkModel"
}

if __name__ == "__main__":
    train(config)