import json
import torch.utils.data as data_utl
from torchvision import transforms
from transforms import *
from make_dataset import *


def choose_dataset():
    train_transforms = transforms.Compose([RandomCrop(IMG_SIZE), RandomBC(), RandomHorizontalFlip(), Normalize()])
    test_transforms = transforms.Compose([CenterCrop(IMG_SIZE), Normalize()])

    with open(SPLIT_FILE, 'r') as f:
        data = json.load(f)

    all_labels = {}
    test = {}
    validation = {}

    # make a dictionary with label as a key and video numbers as values
    for key in data.keys():
        label = data[key]['action'][0]
        if label in all_labels.keys():
            all_labels[label].append(key)
        else:
            all_labels[label] = [key]

    # randomly assign 15% of videos to a test set and 15% of videos to a validation set
    for label in all_labels.keys():
        videos_id = all_labels[label]
        N = int(len(videos_id) * 0.3)
        middle = int(N/2)
        random_samples = random.sample(videos_id, N)
        test_set = random_samples[:middle]
        for dictionary in [test, validation]:
            for id_key in test_set:
                dictionary[id_key] = data[id_key]
                del data[id_key]
            test_set = random_samples[middle:]

    # make datasets
    train_dataset = Dataset(data, train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10,
                                                   pin_memory=False)

    val_dataset = Dataset(validation, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10,
                                                 pin_memory=False)

    test_dataset = Dataset(test, test_transforms)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10,
                                                  pin_memory=False)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}

    return dataloaders, datasets
