import sys
sys.path.append('../')
sys.path.append('./')


import torch
import torch.nn as nn
import numpy as np

import seisbench.models as sbm

from utils import utils


class PhaseNetHandler():
    def __init__(self):
        self.original = sbm.PhaseNet.from_pretrained('original')
        self.stead = sbm.PhaseNet.from_pretrained('stead')

        self.original_collapsed = sbm.PhaseNet.from_pretrained('original')
        self.stead_collapsed = sbm.PhaseNet.from_pretrained('stead')
        layer_original = nn.Conv1d(1, 8, 7, padding='same', stride=(1,))
        layer_stead = nn.Conv1d(1, 8, 7, padding='same', stride=(1,))
        layer_original.weight.data = self.original_collapsed.inc.weight.data[:,0,:].unsqueeze(1)
        layer_stead.weight.data = self.stead_collapsed.inc.weight.data[:,0,:].unsqueeze(1)
        self.original_collapsed.inc = layer_original
        self.stead_collapsed.inc = layer_stead 
        self.original_collapsed.in_channels = 1
        self.stead_collapsed.in_channels = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_preds(self, x, model_type='original', denoise=False):
        assert len(x.shape) == 3, "Input should be of shape (batch_size, channels, 6000)"
            
        # If given 3 channel input but model is collapsed, take the first channel of the input
        if x.shape[1] == 3 and model_type == 'original_collapsed' or model_type == 'stead_collapsed':
            x = x[:,0,:].unsqueeze(1)
            
        
        if denoise:
            x = torch.Tensor(utils.denoise(x.cpu().detach().numpy()))
            
        model = getattr(self, model_type)
        model.to(self.device)
        x.to(self.device)
        preds = model(x).cpu().detach().numpy()
        return preds
    
    def plot_preds(self, x, y, model_type='original', denoise=False):
        if model_type == 'stead' or model_type == 'stead_collapsed':
            utils.plot_preds(x, self.get_preds(x, model_type, denoise=denoise), y, 3, axis=[0, 1])
        else:
            utils.plot_preds(x, self.get_preds(x, model_type, denoise=denoise), y, 3)
        

if __name__ == '__main__':
    handler = PhaseNetHandler()
    train, test, y_train, y_test = utils.load_dataset_from_disk("data/STEAD/chunk2/merged.csv", "data/STEAD/chunk2/merged.hdf5", frac=0.0001)
    handler.plot_preds(train, y_train, 'original', denoise=False)



