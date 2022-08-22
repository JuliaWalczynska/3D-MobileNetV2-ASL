import time
import torch
from torch.autograd import Variable
from config import *
from sklearn.metrics import f1_score, recall_score, precision_score


def train(model, dataloaders, save_data, writer, optimizer, scheduler, criterion):
    best_val_loss = 100

    for epoch in range(0, EPOCHS):
        print("\n--- EPOCH", epoch + 1, "/", EPOCHS, "---")
        train_acc, correct_train, train_loss, target_count = 0, 0, 0, 0
        start_time = time.time()
        for j, data in enumerate(dataloaders['train']):
            model.train()
            inputs, labels, vid_id = data
            labels = torch.mean(labels, -1)
            _, label_idx = labels.max(dim=1)

            optimizer.zero_grad()

            inputs = inputs.cuda()
            label_idx = label_idx.cuda()

            input_var = Variable(inputs)
            target_var = Variable(label_idx)

            outputs = model(input_var)

            loss = criterion(outputs, target_var)
            loss.backward()
            optimizer.step()

            _, predicted_idx = outputs.max(dim=1)

            target_count += target_var.size(0)

            correct_train += (label_idx == predicted_idx).sum().item()
            train_acc = 100 * correct_train / target_count

            train_loss += loss.item()
            if j % 20 == 1:
                print("Step:", j, " acc:", train_acc, "loss:", train_loss / target_count)

        total_train_acc = 100 * correct_train / target_count
        total_train_loss = train_loss / target_count
        end_time = time.time()
        print("Train acc:", total_train_acc, "loss:", total_train_loss, "time:", end_time - start_time)

        val_acc, correct_val, val_loss, target_count, top5 = 0, 0, 0, 0, 0
        start_time = time.time()
        y_pred = []
        y_true = []

        for k, data in enumerate(dataloaders['val']):
            model.eval()
            inputs, labels, vid_id = data
            labels = torch.mean(labels, -1)
            _, label_idx = labels.max(dim=1)

            inputs = inputs.cuda()
            label_idx = label_idx.cuda()

            input_var = Variable(inputs)
            target_var = Variable(label_idx)

            outputs = model(input_var)

            loss = criterion(outputs, target_var)

            _, predicted_idx = outputs.max(dim=1)
            _, top5_idx = torch.topk(outputs, 5, dim=1)

            target_count += target_var.size(0)

            correct_val += (label_idx == predicted_idx).sum().item()

            output = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)
            labels = label_idx.data.cpu().numpy()
            y_true.extend(labels)

            for idx in range(0, BATCH_SIZE):
                if label_idx[0] in top5_idx[0]:
                    top5 += 1

            val_loss += loss.item()

        val_total_acc = 100 * correct_val / target_count
        val_total_loss = val_loss / target_count
        top5_acc = 100 * top5 / target_count
        end_time = time.time()
        f1 = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')

        recall = recall_score(y_true, y_pred, average='macro')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')

        precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=1)

        print("Validation acc:", val_total_acc, "Top 5 accuracy:", top5_acc, " loss: ", val_total_loss, "time:",
              end_time - start_time)
        print("f1:", f1, " f1_weighted:", f1_weighted, " recall:", recall, " recall_weighted", recall_weighted,
              " precision:", precision, " precision_weighted:", precision_weighted)
        if (val_total_loss <= best_val_loss) and (val_total_acc <= total_train_acc) and (
                val_total_loss >= total_train_loss):
            print("Model saved!")
            best_val_loss = val_total_loss
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }

            torch.save(state, SAVE_PATH)
        scheduler.step(val_total_loss)
        writer.writerow([epoch, total_train_acc, total_train_loss, val_total_acc, top5_acc, val_total_loss,
                         f1, f1_weighted, precision, precision_weighted, recall, recall_weighted])
        save_data.flush()
        return model, optimizer, scheduler, criterion
