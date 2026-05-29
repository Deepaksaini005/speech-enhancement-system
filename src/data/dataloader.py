from torch.utils.data import DataLoader


def create_dataloader(dataset, batch_size=16, shuffle=True, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
