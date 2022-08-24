import csv
from torch import optim
from model import *
from choose_dataset import *
from train import *
from test import *

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    torch.manual_seed(0)
    np.random.seed(0)

    dataloaders, datasets = choose_dataset()

    mobilenet = MobileNetV2(num_classes=100)
    mobilenet.cuda()
    mobilenet = nn.DataParallel(mobilenet)
    optimizer = optim.AdamW(mobilenet.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True, threshold=0,
                                                           eps=0)
    criterion = nn.CrossEntropyLoss()

    save_data = open(CSV_PATH, 'w')
    writer = csv.writer(save_data)
    writer.writerow(
        ['epoch', 'train_acc', 'train_loss', 'val_acc', 'Top5_acc', 'val_loss', 'f1', 'f1_w', 'prec', 'prec_w', 'rec',
         'rec_w'])
    save_data.flush()

    mobilenet, optimizer, scheduler, criterion = train(mobilenet, dataloaders, save_data, writer, optimizer, scheduler, criterion)
    test(mobilenet, dataloaders, save_data, writer, criterion)


