"""Train-only, missing-AP-safe RSS augmentation."""

import torch


class DeviceAugmentedRSSDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, gain_std=0.10, noise_std=0.05, dropout_prob=0.06):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.gain_std = gain_std
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        rss = self.x[index].clone()
        observed = rss > -0.999
        gain = torch.randn(()) * self.gain_std
        noise = torch.randn_like(rss) * self.noise_std
        rss = torch.where(observed, (rss + gain + noise).clamp(-1, 1), rss)
        dropped = (torch.rand_like(rss) < self.dropout_prob) & observed
        rss[dropped] = -1.0
        return rss, self.y[index]
