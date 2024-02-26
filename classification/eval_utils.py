import numpy as np
import torch 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, random_split
import pandas as pd
import h5py
import torch.nn.functional as f
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from phasenet import PhaseNet

class ClassificationEvaluator():
    
    def __init__(self, model, val_loader, wave_length=6000, batch_size=64, num_batches=8, pretrained=False):
        self.model = model
        self.val_loader = val_loader
        self.wave_length = wave_length
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.pretrained = pretrained
        
        num_waves = batch_size*num_batches
        self.raw_wave = torch.empty(num_waves, wave_length)
        self.raw_labels = torch.empty(num_waves, 3, wave_length)
        self.preds = torch.empty(num_waves, 3, wave_length)
        self.get_predictions()
        
        # Transformations on predictions later used for some metrics
        preds_maxs = (self.preds.max(axis = 2))[0]
        largmax = self.raw_labels.max(axis=2)[1]
        noise_mask = largmax[:, 0] == 0

        self.preds_maxs_eq = preds_maxs[~noise_mask][:, 0:2]
        self.preds_maxs_noise = preds_maxs[noise_mask][:, 0:2]

        self.theta_p = self.best_threshold(self.preds_maxs_eq[:, 0], self.preds_maxs_noise[:, 0])
        self.theta_s = self.best_threshold(self.preds_maxs_eq[:, 1], self.preds_maxs_noise[:, 1])
        
        # More transformations needed for "within range ROC curve" plot (not sure i really need this)
        """picks = largmax[:, 0:2]
        preds_pick = (self.preds.max(axis = 2))[1]

        p_preds_eq = self.preds[:, 0, :][~noise_mask]
        p_preds_noise = self.preds[:, 0, :][noise_mask]

        s_preds_eq = self.preds[:, 1, :][~noise_mask]
        s_preds_noise = self.preds[:, 1, :][noise_mask]"""
        
    def best_threshold(self, eq_vec, noise_vec, num_thetas=100):
        best_f1 = 0
        best_theta = 0
        for theta in np.linspace(0,1,num_thetas):
            f1 = general_f1score(eq_vec, noise_vec, theta)
            if f1 > best_f1: 
                best_theta = theta
                best_f1 = f1
    
        return best_theta
    
    def get_predictions(self):
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.val_loader):
                if i == self.num_batches: break
                self.raw_wave[self.batch_size*i:self.batch_size*(i+1)] = inputs[:, 0, :]
                self.raw_labels[self.batch_size*i:self.batch_size*(i+1)] = labels
                if self.pretrained: inputs = inputs[:, 0, :]
                self.preds[self.batch_size*i:self.batch_size*(i+1)] = self.model(inputs, logits=False)

    def precision_recall(self, eq_vec, noise_vec, theta):
        true_positives = (eq_vec > theta).sum()
        false_positives = (noise_vec > theta).sum()
        false_negatives = (eq_vec < theta).sum()
        
        #  precision, recall
        return true_positives / (true_positives+false_positives), true_positives / (true_positives + false_negatives)

    def f1score(self, theta_p = None, theta_s = None):
        if theta_p == None: theta_p = self.theta_p
        if theta_s == None: theta_s = self.theta_s
        
        precision, recall = self.precision_recall(self.preds_maxs_eq[:, 0], self.preds_maxs_noise[:, 0], theta_p)
        f1_p = 2*precision*recall / (precision + recall)
        
        precision, recall = self.precision_recall(self.preds_maxs_eq[:, 1], self.preds_maxs_noise[:, 1], theta_s)
        f1_s = 2*precision*recall / (precision + recall)
        
        return f1_p, f1_s
    
    def roc_rates(self, eq_vec, noise_vec, num_thetas = 100):
        true_positives = np.empty(num_thetas)
        false_positives = np.empty(num_thetas)
        false_negatives = np.empty(num_thetas)
        true_negatives = np.empty(num_thetas)
        
        for i, theta in enumerate(np.linspace(0,1,num_thetas)):
            true_positives[i] = (eq_vec > theta).sum()
            
        for i, theta in enumerate(np.linspace(0,1,num_thetas)):
            false_positives[i] = (noise_vec > theta).sum()
            
        for i, theta in enumerate(np.linspace(0,1,num_thetas)):
            false_negatives[i] = (eq_vec < theta).sum()
            
        for i, theta in enumerate(np.linspace(0,1,num_thetas)):
            true_negatives[i] = (noise_vec < theta).sum()
            
        # TPR, FPR
        return true_positives / (true_positives+false_negatives), false_positives / (false_positives + true_negatives)

    def roc_curve(self, num_thetas=100, ax=None):
        tpr_p, fpr_p = self.roc_rates(self.preds_maxs_eq[:, 0], self.preds_maxs_noise[:, 0], num_thetas=num_thetas)
        tpr_s, fpr_s = self.roc_rates(self.preds_maxs_eq[:, 1], self.preds_maxs_noise[:, 1], num_thetas=num_thetas)
        
        if ax == None: fig, ax = plt.subplots(1,1, figsize=(5, 5))
        
        ax.plot(fpr_p, tpr_p)
        ax.plot(fpr_s, tpr_s)
        ax.plot(np.linspace(0,1,100), np.linspace(0,1,100), color = 'gray', linestyle='--', alpha=0.3) # Diagonal
        ax.title.set_text('ROC Curve')
        ax.set(xlabel='FPR', ylabel='TPR')
        ax.legend(['P', 'S'])
        
    def plot_confusion_matrix(self, theta_p=None, theta_s=None):
        if (theta_p == None): theta_p = self.theta_p
        if (theta_s == None): theta_s = self.theta_s
        
        y_pred = torch.concat(((self.preds_maxs_eq[:, 0] > theta_p), (self.preds_maxs_noise[:, 0] > theta_p)))
        y_true = torch.concat((torch.ones(len(self.preds_maxs_eq)), torch.zeros(len(self.preds_maxs_noise))))
        
        cm = confusion_matrix(y_true, y_pred)
        disp1 = ConfusionMatrixDisplay(cm)
        disp1.plot()
        
        y_pred = torch.concat(((self.preds_maxs_eq[:, 1] > theta_s), (self.preds_maxs_noise[:, 1] > theta_s)))
        y_true = torch.concat((torch.ones(len(self.preds_maxs_eq)), torch.zeros(len(self.preds_maxs_noise))))
        
        cm = confusion_matrix(y_true, y_pred)
        disp2 = ConfusionMatrixDisplay(cm)
        disp2.plot()
    
    def plot_preds(self, num_plots, offset=0):
        fig, axs = plt.subplots(num_plots, num_plots, figsize=(12, 10))  # Create 2 subplots
        
        for i in range(num_plots):
            for j in range(num_plots):
                axs[i][j].plot(self.raw_labels[i+num_plots*j + offset, 2, :])
                axs[i][j].plot(self.raw_wave[i+num_plots*j + offset, :], alpha=0.4)
                axs[i][j].plot(self.preds[i+num_plots*j + offset, 0, :])
                axs[i][j].plot(self.preds[i+num_plots*j + offset, 1, :])
                
        fig.show()
        
        
def get_preds(model, wave_tensor, num_waves, eq = False):
    preds = torch.empty(num_waves, 3, 6000)
    pick_preds = torch.empty(num_waves, 3)
    
    for i in range(num_waves):
        if (eq): preds[i] = torch.concat(model(wave_tensor[i].unsqueeze(0)))[[1, 2, 0], :]
        else: preds[i] = model(wave_tensor[i].unsqueeze(0))
        pick_preds[i, 0] = preds[i][0].argmax()
        pick_preds[i, 1] = preds[i][1].argmax()
        
        
    return preds, pick_preds


def plot_preds(raw, preds, labels, num_plots, offset = 0, figsize=(20,10), noise=False):
    fig, ax = plt.subplots(num_plots, 1, figsize=figsize, sharex=True)
    plt.tick_params(axis='both', which='major', labelsize=12)
    legend = ['Wave (Z)', 'P-Prediction', 'S-Prediction']
    fig.text(0.07, 0.5, 'Probability', va='center', rotation='vertical', fontsize=20)
    ymin, ymax = 0, 1
    j = 0
    cutoff = 3000
    x = np.arange(cutoff)/100
    for i in range(num_plots):
        #for j in range(num_plots):
        idx = offset + num_plots*i + j
        cur_ax = ax[i]
        cur_ax.tick_params(labelsize=15)
        
        p_pick = labels[idx][0].detach().numpy().argmax()
        s_pick = labels[idx][1].detach().numpy().argmax()
        
        if (p_pick != 0 and s_pick != 0):
            cur_ax.plot(x, (raw[idx]/(raw[idx].abs().max())-raw[idx].min())[:cutoff], color = 'gray', alpha=0.4)
            cur_ax.vlines(p_pick/100, ymin, ymax, color='tab:red', linestyle='--', linewidth=2, alpha=0.4, label='P-arrival')
            cur_ax.vlines(s_pick/100, ymin, ymax, color='tab:blue', linestyle='--', linewidth=2, alpha=0.4, label='S-arrival')
        else:
            cur_ax.plot(x, (raw[idx]/(raw[idx].abs().max())-raw[idx].min())[:cutoff], color = 'gray', alpha=0.4)
        legend = ['Wave (Z)', 'P-Pick', 'S-Pick', 'P-Prediction', 'S-Prediction']
            
        cur_ax.plot(x, preds[idx][0][:cutoff].detach().numpy(), linewidth=1, color='tab:red')
        cur_ax.plot(x, preds[idx][1][:cutoff].detach().numpy(), linewidth=1, color='tab:blue')
        #ax[i][j].plot(preds[num_plots*i+j][2].detach().numpy(), linewidth=1)
        cur_ax.legend(legend, loc = 'upper right')
    plt.ylim(ymin-0.1,ymax)
        
    ax[2].set_xlabel('Time (s)', fontsize = 20)

    
def harder_window_roc_rates(eq_preds, noise_preds, picks, num_thetas = 100, window=10):
    """ Take the max over all preds as the prediction, check if that max is a) within window of the label pick, b) has confidence > theta"""
    true_positives = np.zeros(num_thetas)
    false_positives = np.zeros(num_thetas)
    false_negatives = np.zeros(num_thetas)
    true_negatives = np.zeros(num_thetas)
    
    for i, theta in enumerate(np.linspace(0,1,num_thetas)):
        for j in range(len(eq_preds[:,0])):
            pick = picks[j]
            conf, loc = eq_preds[j].max(), eq_preds[j].argmax()
            true_positives[i] += (conf > theta and abs(pick - loc) <= window).sum()
            
    false_negatives = len(eq_preds[:,0]) - true_positives
    
    for i, theta in enumerate(np.linspace(0,1,num_thetas)):
        false_positives[i] = (noise_preds.max(axis=1)[0] > theta).sum()

    true_negatives = len(noise_preds[:,0]) - false_positives
        
    # TPR, FPR
    return true_positives / (true_positives+false_negatives), false_positives / (false_positives + true_negatives)


def window_roc_rates(eq_preds, noise_preds, picks, num_thetas = 100, window=10):
    true_positives = np.zeros(num_thetas)
    false_positives = np.zeros(num_thetas)
    false_negatives = np.zeros(num_thetas)
    true_negatives = np.zeros(num_thetas)
    
    for i, theta in enumerate(np.linspace(0,1,num_thetas)):
        for j in range(len(eq_preds[:,0])):
            l = max(picks[j]-window, 0); r = max(picks[j]+window,1)
            max_in_window = eq_preds[j, l:r].max()
            true_positives[i] += (max_in_window > theta).sum()
            false_negatives[i] += (max_in_window < theta).sum()
        
    for i, theta in enumerate(np.linspace(0,1,num_thetas)):
        
        false_positives[i] = (noise_preds.max(axis=1)[0] > theta).sum()
        true_negatives[i] = (noise_preds.max(axis=1)[0] < theta).sum()
        
        
    # TPR, FPR
    return true_positives / (true_positives+false_negatives), false_positives / (false_positives + true_negatives)


def window_roc_curve(preds_maxs_eq, preds_maxs_noise, picks, num_thetas=100, ax=None):
    tpr_p, fpr_p = harder_window_roc_rates(preds_maxs_eq, preds_maxs_noise, picks, num_thetas=num_thetas, window=10)
    #tpr_s, fpr_s = window_roc_rates(preds_maxs_eq[1], preds_maxs_noise[1], picks, num_thetas=num_thetas)
    
    if ax == None: fig, ax = plt.subplots(1,1, figsize=(5, 5))
    
    #ax.plot(fpr_p, tpr_p)
    ax.scatter(fpr_p, tpr_p, c=(np.arange(100)/100))
    #ax.plot(fpr_s, tpr_s)
    ax.plot(np.linspace(0,1,100), np.linspace(0,1,100), color = 'gray', linestyle='--', alpha=0.3) # Diagonal
    ax.title.set_text('ROC Curve')
    ax.set(xlabel='FPR', ylabel='TPR')
    #ax.legend(['P', 'S'])

    
def roc_rates(eq_vec, noise_vec, num_thetas = 100):
    true_positives = np.empty(num_thetas)
    false_positives = np.empty(num_thetas)
    false_negatives = np.empty(num_thetas)
    true_negatives = np.empty(num_thetas)
    
    for i, theta in enumerate(np.linspace(0,1,num_thetas)):
        true_positives[i] = (eq_vec > theta).sum()
        
    for i, theta in enumerate(np.linspace(0,1,num_thetas)):
        false_positives[i] = (noise_vec > theta).sum()
        
    for i, theta in enumerate(np.linspace(0,1,num_thetas)):
        false_negatives[i] = (eq_vec < theta).sum()
        
    for i, theta in enumerate(np.linspace(0,1,num_thetas)):
        true_negatives[i] = (noise_vec < theta).sum()
        
        
    # TPR, FPR
    return true_positives / (true_positives+false_negatives), false_positives / (false_positives + true_negatives)


def precision_recall(eq_vec, noise_vec, theta):
    
    true_positives = (eq_vec > theta).sum()
        
    false_positives = (noise_vec > theta).sum()
    
    false_negatives = (eq_vec < theta).sum()
    
    #  precision, recall
    return true_positives / (true_positives+false_positives), true_positives / (true_positives + false_negatives)


def general_f1score(eq_vec, noise_vec, theta):
    precision, recall = precision_recall(eq_vec, noise_vec, theta)
    return 2*precision*recall / (precision + recall)


def best_threshold(eq_vec, noise_vec, num_thetas=100):
    best_f1 = 0
    best_theta = 0
    for i, theta in enumerate(np.linspace(0,1,num_thetas)):
        f1 = f1score(eq_vec, noise_vec, theta)
        if f1 > best_f1: 
            best_theta = theta
            best_f1 = f1
    
    return best_theta
        

def roc_curve(preds_maxs_eq, preds_maxs_noise, num_thetas=100, ax=None):
    tpr_p, fpr_p = roc_rates(preds_maxs_eq[:, 0], preds_maxs_noise[:, 0], num_thetas=num_thetas)
    tpr_s, fpr_s = roc_rates(preds_maxs_eq[:, 1], preds_maxs_noise[:, 1], num_thetas=num_thetas)
    
    if ax == None: fig, ax = plt.subplots(1,1, figsize=(5, 5))
    
    ax.plot(fpr_p, tpr_p)
    ax.plot(fpr_s, tpr_s)
    ax.plot(np.linspace(0,1,100), np.linspace(0,1,100), color = 'gray', linestyle='--', alpha=0.3) # Diagonal
    ax.title.set_text('ROC Curve')
    ax.set(xlabel='FPR', ylabel='TPR')
    ax.legend(['P', 'S'])
    
    
def plot_confusion_matrix(preds_maxs_eq, preds_maxs_noise, theta):
    y_pred = torch.concat(((preds_maxs_eq[:, 0] > theta), (preds_maxs_noise[:, 0] > theta)))
    y_true = torch.concat((torch.ones(len(preds_maxs_eq)), torch.zeros(len(preds_maxs_noise))))
    
    cm = confusion_matrix(y_true, y_pred)
    disp1 = ConfusionMatrixDisplay(cm)
    disp1.plot()
    
    y_pred = torch.concat(((preds_maxs_eq[:, 1] > theta), (preds_maxs_noise[:, 1] > theta)))
    y_true = torch.concat((torch.ones(len(preds_maxs_eq)), torch.zeros(len(preds_maxs_noise))))
    
    cm = confusion_matrix(y_true, y_pred)
    disp2 = ConfusionMatrixDisplay(cm)
    disp2.plot()