import time
import torch
from torch.autograd import Variable
from config import *
from sklearn.metrics import f1_score, recall_score, precision_score


def test(model, dataloaders, save_data, writer, criterion):
    test_acc, correct_test, test_loss, target_count, top5 = 0, 0, 0, 0, 0
    start_time = time.time()
    y_pred = []
    y_true = []
    model.etest()
    for k, data in enumerate(dataloaders['test']):
        inputs, labels, vid_id = data
        labels = torch.mean(labels,-1)
        _, label_idx = labels.max(dim=1)
    
        inputs=inputs.cuda()
        label_idx=label_idx.cuda()
    
        input_var = Variable(inputs)
        target_var = Variable(label_idx)
    
        outputs = model(input_var)
    
        loss = criterion(outputs, target_var)
    
        _, predicted_idx = outputs.max(dim=1)
        _, top5_idx = torch.topk(outputs, 5, dim=1)
    
        target_count += target_var.size(0)
    
        correct_test += (label_idx == predicted_idx).sum().item()
    
        output = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)
        labels = label_idx.data.cpu().numpy()
        y_true.extend(labels)
    
        for idx in range(0, BATCH_SIZE):
            if (label_idx[0] in top5_idx[0]):
                top5 += 1
    
        test_loss += loss.item()
    
    test_total_acc = 100 * correct_test / target_count
    test_total_loss = test_loss/target_count
    top5_acc = 100 * top5 / target_count
    end_time = time.time()
    f1 = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    recall = recall_score(y_true, y_pred, average='macro')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    precision_weighted = precision_score(y_true, y_pred, average='weighted',zero_division=1)
    
    print("Test acc:", test_total_acc, "Top 5 accuracy:", top5_acc, " loss: ", test_total_loss, "time:", end_time-start_time)
    print("f1:", f1, " f1_weighted:", f1_weighted, " recall:", recall, " recall_weighted", recall_weighted,
          " precision:", precision, " precision_weighted:", precision_weighted)

    writer.writerow(["TEST", "None", "None", test_total_acc, top5_acc, test_total_loss,
                     f1, f1_weighted, precision, precision_weighted, recall, recall_weighted])
    save_data.flush()