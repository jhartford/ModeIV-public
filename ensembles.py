import numpy
import torch
import numpy as np

def collect_predictions(models, treatment, features=None, instruments=None):
    '''
    Get the predictions from a list of models and return them in a single tensor
    '''
    ensemble_size = len(models)
    batch_size = treatment.shape[0]
    predictions = torch.zeros((batch_size, ensemble_size))
    for i, m in enumerate(models):
        if i == 0:
            predictions = predictions.to(m.device)
        pred = m.predict(treatment, features, instruments, numpy=False)
        predictions[:, i] = pred.detach().flatten()
    return predictions

class ModeIV():
    '''
    ModeIV implementation of the Venter mode.
    '''
    def __init__(self, models, percent_valid=0.5, k=None):
        self.models = models
        self.perc = percent_valid
        self.k = k

    def estimate_mode(self, predictions, k):
        '''
        Estimate the Venter mode - the mean of the k closest predictions.

        Args:
            predictions: A tensor of size (batch_size, ensemble_size) containing the predictions
                         from each ensemble member.
            k: The number of elements that will form part of the modal interval. 2 <= k <= ensemble_size.

        Returns:
            The Dalenius / Venter mode - the mean of the k closest predictions.
        '''
        sorted_pred, _ = torch.sort(predictions, axis=1)
        min_idx = torch.argmin(sorted_pred[:,k-1:] - sorted_pred[:,:(predictions.shape[1] - k + 1)], axis=1)
        modal_indices = torch.cat([min_idx[:,None] + i for i in range(k)], dim=1)
        return torch.gather(sorted_pred, 1, modal_indices).mean(axis=1)
    
    def predict(self, treatment, features=None, instruments=None, numpy=True):
        predictions = collect_predictions(self.models.values(), treatment, features, instruments)
        # get modal interval size either as a proportion of candidates or a fixed number
        k = self.k if self.k is not None else int(self.perc * instruments.shape[1])
        mode = self.estimate_mode(predictions, k)
        if numpy:
            return mode.detach().cpu().numpy()
        else:
            return mode

class MeanEnsemble():
    '''
    Mean ensemble baseline
    ''' 
    def __init__(self, models):
        self.models = models

    def predict(self, treatment, features=None, instruments=None, numpy=True):
        predictions = collect_predictions(self.models.values(), treatment, features, instruments)
        if numpy:
            return predictions.mean(dim=1).detach().cpu().numpy()
        else:
            return predictions.mean(dim=1)