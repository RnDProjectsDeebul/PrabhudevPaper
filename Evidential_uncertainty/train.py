import torch
import torch.nn as nn
import copy
import time
from helpers import get_device, one_hot_embedding
from losses import relu_evidence
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
import torch.nn.functional as nnf
from scipy.special import entr


def print_cf_mat(model,dataloaders,name):
    y_pred = []
    y_true = []
    y_pred_distribution = []

    # iterate over test data
    for inputs, labels in dataloaders:
            output = model(inputs) # Feed Network

            evidence = relu_evidence(output)
            alpha = evidence + 1
            uncertainty = 10 / torch.sum(alpha, dim=1, keepdim=True)
            y_pred_distribution.extend(uncertainty.detach().numpy().flatten())

            _, predicted = torch.max(model(inputs), 1)

            y_pred.extend(predicted.data.cpu().numpy()) # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth


    # constant for classes
    classes = (
                "aamir_khan" ,
                "aaron_staton",
                "aaron_tveit",
                "aaron_yoo",
                "abbie_cornish",
                "abel_ferrara",
                "abigail_breslin",
                "abigail_spencer",
                "a_j__buckley", 
                "a_r__rahman")

    # classes = (
    #             "0" ,
    #             "1",
    #             "2",
    #             "3",
    #             "4",
    #             "5",
    #             "6",
    #             "7",
    #             "8", 
    #             "9",
    #             "10",
    #             "11",
    #             "13",
    #             "14",
    #             "15",
    #             "16", )

    # classes = (
    #             "ball" ,
    #             "building",
    #             "chainsaw",
    #             "dog",
    #             "fish",
    #             "mellophone",
    #             "parachute",
    #             "pump_station",
    #             "radio", 
    #             "truck",)

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.title("confusion matrix")
    plt.tight_layout()
    plt.savefig(name+'_confusion_matrix.png')
    plt.clf()

    fig = plt.figure(figsize =(10, 7))
    plt.boxplot(y_pred_distribution)
    plt.title("confidence plot")
    plt.ylabel('confidence level')
    # plt.xlabel('confidence level')
    plt.savefig(name+'y_pred_distribution_singlePlot.png')
    plt.clf()

    data1 = pd.DataFrame({0: y_pred_distribution})
    data = np.hstack((data1,data1))
    df = pd.DataFrame(data,columns=['validation','validation'])
    a=plt.ylabel('confidence level')
    boxplot = df.boxplot(grid=True)
    plt.title("confidence plot")
    plt.savefig(name+'_confidence_plot.png')
    plt.clf()

    data2 = pd.DataFrame({0: y_pred})
    data = np.hstack((data2,data2))
    df = pd.DataFrame(data,columns=['validation','validation'])
    a=plt.ylabel('confidence level')
    boxplot = df.boxplot(grid=True)
    plt.title("confidence plot")
    plt.savefig(name+'_confidence_plot_y_pred.png')
    plt.clf()

def train_model(
    model,
    dataloaders,
    num_classes,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=50,
    device=None,
    uncertainty=False,
):

    since = time.time()

    if not device:
        device = get_device()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    losses = {"loss": [], "phase": [], "epoch": []}
    accuracy = {"accuracy": [], "phase": [], "epoch": []}
    evidences = {"evidence": [], "type": [], "epoch": []}

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                print("Training...")
                model.train()  # Set model to training mode
            else:
                print("Validating...")
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            correct = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):

                    if uncertainty:
                        y = one_hot_embedding(labels, num_classes)
                        y = y.to(device)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(
                            outputs, y.float(), epoch, num_classes, 10, device
                        )

                        match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
                        acc = torch.mean(match)
                        evidence = relu_evidence(outputs)
                        alpha = evidence + 1
                        u = num_classes / torch.sum(alpha, dim=1, keepdim=True)

                        total_evidence = torch.sum(evidence, 1, keepdim=True)
                        mean_evidence = torch.mean(total_evidence)
                        mean_evidence_succ = torch.sum(
                            torch.sum(evidence, 1, keepdim=True) * match
                        ) / torch.sum(match + 1e-20)
                        mean_evidence_fail = torch.sum(
                            torch.sum(evidence, 1, keepdim=True) * (1 - match)
                        ) / (torch.sum(torch.abs(1 - match)) + 1e-20)

                    else:
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if scheduler is not None:
                if phase == "train":
                    scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            losses["loss"].append(epoch_loss)
            losses["phase"].append(phase)
            losses["epoch"].append(epoch)
            accuracy["accuracy"].append(epoch_acc.item())
            accuracy["epoch"].append(epoch)
            accuracy["phase"].append(phase)

            print(
                "{} loss: {:.4f} acc: {:.4f}".format(
                    phase.capitalize(), epoch_loss, epoch_acc
                )
            )

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # for i, (inputs, labels) in enumerate(dataloaders[phase]):
    print_cf_mat(model,dataloaders["train"],"imagenette_train")
    print_cf_mat(model,dataloaders["val"],"imagenette_val")

    # load best model weights
    model.load_state_dict(best_model_wts)
    metrics = (losses, accuracy)

    return model, metrics
