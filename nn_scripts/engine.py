# Testing the engine.py file functions here before adding them to the script

"""
This script contains all the training functions needed for
training torch models.

The functions include:
    train       : trains the model for specified number of epochs
    evaluate    : evaluates the model on the provided dataloader
    predict     : makes predictions using the model on provided data
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

__all__ = ["train", "evaluate", "predict"]

def __train_step(model: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 loss_fn: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device) -> Tuple[float, float]:
    """
    Performs forward and back propagation on the model on the data from the dataloader.
    Used for multi-class classification.

    Args:
        model          - the torch model to train
        dataloader     - the training data with the features and labels
        loss_fn        - the loss function for the model
        optimizer      - optimizer used to update model params
        device         - device on which training is done
    
    Returns:
        A tuple with the loss and accuracy of the model
        (loss, accuracy)
    """
    from tqdm.auto import tqdm
    model.to(device)
    model.train()
    train_loss, train_accuracy = 0, 0
    for batch, (X, y) in tqdm(enumerate(dataloader), total = len(dataloader),
                              desc = "train-step",
                              nrows = 5,
                              leave = False):
        X, y = X.to(device), y.to(device)
        logits = model(X)
        preds = F.softmax(logits, dim=1).argmax(dim=1)
        loss = loss_fn(logits, y)
        train_loss += loss.item()
        train_accuracy += (preds == y).sum().item()/len(preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(dataloader)
    train_accuracy /= len(dataloader)
    return train_loss, train_accuracy

def __test_step(model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                device: torch.device) -> Tuple[float, float]:
    """
    Performs a simple forward pass on the model and evaluates the model's
    loss and accuracy.

    Args:
        model          - the torch model to train
        dataloader     - the training data with the features and labels
        loss_fn        - the loss function for the model
        device         - device on which training is done
    
    Returns:
        A tuple with the loss and accuracy of the model on the testing data
        (loss, accuracy)
    """
    from tqdm.auto import tqdm
    model.to(device)
    model.eval()
    test_loss, test_accuracy = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in tqdm(enumerate(dataloader), total = len(dataloader),
                                  desc = "test-step",
                                  nrows = 5,
                                  leave = False):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = F.softmax(logits, dim=1).argmax(dim=1)
            loss = loss_fn(logits, y)
            test_loss += loss.item()
            test_accuracy += (preds == y).sum().item()/len(preds)
        test_loss /= len(dataloader)
        test_accuracy /= len(dataloader)
        return test_loss, test_accuracy


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          test_freq: int = 1):
    """
    Trains the model for "epochs" and returns a dict with the model's
    performance per epoch on the training data.

    Args:
        model              - torch model to train
        train_dataloader   - torch dataloader with the training features and labels
        train_dataloader   - torch dataloader with the testing features and labels
        loss_fn            - loss function used to evaluate the model
        optimizer          - optimizer used to update the model's parameters
        epochs             - number of training steps performed on the model
        device             - where to run the training at ("cpu" "mps" "cuda")
        test_freq          - the interval, in epochs, at which the model's scores are printed
    
    Returns:
        {train_loss: [],
         train_accuracy: [],
         test_loss: [],
         test_accuracy: []}
    """
    from tqdm.auto import tqdm
    results = {"train_loss": [],
               "train_accuracy": [],
               "test_loss": [],
               "test_accuracy": [],}
    
    for epoch in tqdm(range(epochs),
                      desc = "Training the model"):
        train_loss, train_accuracy = __train_step(model=model,
                                                  dataloader=train_dataloader,
                                                  loss_fn=loss_fn,
                                                  optimizer=optimizer,
                                                  device=device)
        test_loss, test_accuracy = __test_step(model=model,
                                               dataloader=test_dataloader,
                                               loss_fn=loss_fn,
                                               device=device)
        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_accuracy)
        results["test_loss"].append(test_loss)
        results["test_accuracy"].append(test_accuracy)
        if test_freq == 0:
            continue
        elif epoch % test_freq == 0:
            print(f"train_loss : {train_loss}, train_acc : {train_accuracy} | test_loss : {train_loss}, test_acc : {train_accuracy}")
    return results

def evaluate(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """
    Evaluates the model on the data provided in the dataloader
    and returns the model's loss and accuracy.

    Args:
        model       - model to evaluate
        dataloader  - data to evaluate the model on
        loss_fn     - loss function used to evaluate the model's performance
        device      - device that all the calculations run on
    
    Returns:
        (loss, accuracy)
    """
    from tqdm.auto import tqdm
    model.to(device)
    model.eval()
    loss, accuracy = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in tqdm(enumerate(dataloader), total = len(dataloader),
                                desc = "Evaluating"):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = F.softmax(logits, dim=1).argmax(dim=1)
            loss += loss_fn(logits, y)
            accuracy += (preds == y).sum().item()/len(preds)
        loss /= len(dataloader)
        accuracy /= len(dataloader)
    return loss, accuracy

def predict(model: torch.nn.Module,
            features,
            device: torch.device):
    """
    Makes predictions on features using the model
    and returns the logits along with the predictions

    Args:
        model       - model to make predictions
        features    - inputs to the model
        device      - where the computations are performed
    
    Returns:
        (logits, predictions)
    """
    model.to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(features)
        preds = F.softmax(logits, dim=1).argmax(dim=1)
    return logits, preds