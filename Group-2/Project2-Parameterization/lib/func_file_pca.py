import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import netCDF4 as ncd
import torch
import torch.nn as nn
from torch import optim
import matplotlib.cm as cm
from sklearn.decomposition import PCA, FastICA, SparsePCA, KernelPCA
from tqdm import tqdm  # Import tqdm for the progress bar

np.random.seed(10)  # Set random seed for reproducibility

# Define feedforward neural networks with different numbers of hidden layers.
# Each class corresponds to a model with an increasing number of hidden layers:
# - learnKappa_layers1: 1 hidden layer
# - learnKappa_layers2: 2 hidden layers
# - learnKappa_layers3: 3 hidden layers
# - learnKappa_layers4: 4 hidden layers
# Parameters:
# - In_nodes: Number of input nodes
# - Hid: Number of hidden layer nodes
# - Out_nodes: Number of output nodes

class learnKappa_layers1(nn.Module):
    def __init__(self, In_nodes, Hid, Out_nodes):
        super(learnKappa_layers1, self).__init__()
        self.linear1 = nn.Linear(In_nodes, Hid)  # Input to hidden layer
        self.linear2 = nn.Linear(Hid, Out_nodes)  # Hidden to output layer
        self.dropout = nn.Dropout(0.25)  # Dropout to reduce overfitting

    def forward(self, x):
        x2 = self.linear1(x)
        h1 = torch.relu(x2)  # ReLU activation
        h1 = self.dropout(h1)
        y_pred = self.linear2(h1)  # Output predictions
        return y_pred


class learnKappa_layers2(nn.Module):
    def __init__(self, In_nodes, Hid, Out_nodes):
        super(learnKappa_layers2, self).__init__()
        self.linear1 = nn.Linear(In_nodes, Hid)
        self.linear2 = nn.Linear(Hid, Hid)
        self.linear3 = nn.Linear(Hid, Out_nodes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x2 = self.linear1(x)
        h1 = torch.relu(x2)
        h1 = self.dropout(h1)
        h2 = self.linear2(h1)
        h3 = torch.relu(h2)
        h3 = self.dropout(h3)
        y_pred = self.linear3(h3)
        return y_pred


class learnKappa_layers3(nn.Module):
    def __init__(self, In_nodes, Hid, Out_nodes):
        super(learnKappa_layers3, self).__init__()
        self.linear1 = nn.Linear(In_nodes, Hid)
        self.linear2 = nn.Linear(Hid, Hid)
        self.linear3 = nn.Linear(Hid, Hid)
        self.linear4 = nn.Linear(Hid, Out_nodes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x2 = self.linear1(x)
        h1 = torch.relu(x2)
        h1 = self.dropout(h1)
        h2 = self.linear2(h1)
        h3 = torch.relu(h2)
        h3 = self.dropout(h3)
        h4 = self.linear3(h3)
        h5 = torch.relu(h4)
        h5 = self.dropout(h5)
        y_pred = self.linear4(h5)
        return y_pred


class learnKappa_layers4(nn.Module):
    def __init__(self, In_nodes, Hid, Out_nodes):
        super(learnKappa_layers4, self).__init__()
        self.linear1 = nn.Linear(In_nodes, Hid)
        self.linear2 = nn.Linear(Hid, Hid)
        self.linear3 = nn.Linear(Hid, Hid)
        self.linear4 = nn.Linear(Hid, Hid)
        self.linear5 = nn.Linear(Hid, Out_nodes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x2 = self.linear1(x)
        h1 = torch.relu(x2)
        h1 = self.dropout(h1)
        h2 = self.linear2(h1)
        h3 = torch.relu(h2)
        h3 = self.dropout(h3)
        h4 = self.linear3(h3)
        h5 = torch.relu(h4)
        h5 = self.dropout(h5)
        h6 = self.linear4(h5)
        h7 = torch.relu(h6)
        h7 = self.dropout(h7)
        y_pred = self.linear5(h7)
        return y_pred

# Data preprocessing function
# Purpose: Standardizes input data, shuffles samples, and prepares features (x) and labels (y) for training.
def preprocess_train_data(data_load):
    # Create and shuffle indices
    ind = np.arange(0, len(data_load), 1)
    ind_shuffle = copy.deepcopy(ind)
    np.random.shuffle(ind_shuffle)

    # Standardize the first 4 columns (features)
    l_mean, l_std = np.mean(data_load[:, 0]), np.std(data_load[:, 0])
    data_load[:, 0] = (data_load[:, 0] - l_mean) / l_std  # Standardize column 0 (l)

    h_mean, h_std = np.mean(data_load[:, 1]), np.std(data_load[:, 1])
    data_load[:, 1] = (data_load[:, 1] - h_mean) / h_std  # Standardize column 1 (b0)

    t_mean, t_std = np.mean(data_load[:, 2]), np.std(data_load[:, 2])
    data_load[:, 2] = (data_load[:, 2] - t_mean) / t_std  # Standardize column 2 (u*)

    hb_mean, hb_std = np.mean(data_load[:, 3]), np.std(data_load[:, 3])
    data_load[:, 3] = (data_load[:, 3] - hb_mean) / hb_std  # Standardize column 3 (w*)

    # Log-transform and standardize the remaining columns (outputs)
    for j in range(len(data_load[:, 0])):
        data_load[j, 4:] = np.log(data_load[j, 4:] / np.max(data_load[j, 4:]))

    k_mean = np.mean(data_load[:, 4:], axis=0)
    k_std = np.std(data_load[:, 4:], axis=0)
    for k in range(data_load.shape[1] - 4):
        data_load[:, k + 4] = (data_load[:, k + 4] - k_mean[k]) / k_std[k]

    # Split into inputs (x) and outputs (y)
    x = data_load[:, :4]  # First 4 columns as input features
    y = data_load[:, 4:]  # Remaining columns as output labels

    # Return preprocessed data, statistics, and shuffle order
    stats = np.array([l_mean, l_std, h_mean, h_std, t_mean, t_std, hb_mean, hb_std])
    return data_load, x, y, stats, k_mean, k_std

# Data preprocessing function
# Purpose: Standardizes input data, shuffles samples, and prepares features (x) and labels (y) for training.
def preprocess_train_data_2(
        data_load:np.array, 
        dimension_reduction_model:str, 
        dimension_reduction_n_components:int
) -> tuple:
    
    # define data object
    data_obj = get_data_object(data=data_load, 
                               dimension_reduction_model=dimension_reduction_model,
                               dimension_reduction_n_components=dimension_reduction_n_components)
    
    # shuffle data
    data_obj.shuffle_data()
    
    return data_obj

def get_data_object(
        data:np.array,
        dimension_reduction_model:str,
        dimension_reduction_n_components:int,
        mean_override:np.array=None,
        std_override:np.array=None
):

    # define data object
    data_obj = DataStatistics(data=data, 
                              exogenous_column_indexes=list(range(4)),
                              mean_override=mean_override,
                              std_override=std_override)

    # transform endogenous and exogenous variables
    data_obj.transform_data()
    
    # preprocess data
    data_obj.preprocess_data(
        dimension_reduction_model=dimension_reduction_model,
        dimension_reduction_n_components=dimension_reduction_n_components
    )

    return data_obj

class DimensionReduction(object):
    def __init__(self, column_indexes, transformed_data, inverse, transform):
        self.column_indexes = column_indexes
        self.transformed_data = transformed_data
        self.inverse = inverse
        self.transform = transform

class TrainingParameters(object):
    def __init__(self, epochs, patience, model):
        self.loss_array = torch.zeros([epochs, 3])  # Array to store epoch, train, and validation losses
        self.patience = patience
        self.best_loss = float('inf')  # Initialize the best validation loss as infinity
        self.no_improvement = 0  # Counter for epochs without improvement
        self.best_model_state = None  # Placeholder for the best model state
        self.model = model

    def update_results(self, k:int, train_loss:float, valid_loss:float, model_state:dict) -> None:

        # Record the losses for this epoch
        self.loss_array[k, 0] = k  
        self.loss_array[k, 1] = train_loss 
        self.loss_array[k, 2] = valid_loss

        # Early stopping: Check if validation loss improves
        if valid_loss < self.best_loss:
            self.best_loss = valid_loss  # Update best loss
            self.no_improvement = 0
            self.best_model_state = model_state 
        else:
            self.no_improvement = self.no_improvement + 1  # Increment no improvement counter

    def exceeded_patience(self) -> bool:
        return self.no_improvement >= self.patience

    def restore_best_model(self, k:int):
        """
        Restore the best model state after training
        """
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        self.loss_array = self.loss_array[:k,:]
        
        

class DataStatistics(object):
    def __init__(self, data:np.array=np.array([]), exogenous_column_indexes:list=[], mean_override:np.array=None, std_override:np.array=None, dimension_reduction_override=None):
        self.raw_data = data
        self.axis = 0
        self.exogenous_columns = exogenous_column_indexes
        self.endogenous_columns = list(set(range(len(data[0])))-set(exogenous_column_indexes)) if len(data) != 0 else []

        self.transformed_data = data
        self.transformed_mean = mean_override
        self.transformed_std = std_override
        self.dimension_reduction = dimension_reduction_override
        
        self.preprocessed_data = None
        self.preprocessed_exogenous_columns = exogenous_column_indexes
        self.preprocessed_endogenous_columns = self.endogenous_columns
        self.preprocessed_shuffled_data = None

    def exogenous_data(self) -> np.array:
        return self.preprocessed_data[:, self.preprocessed_exogenous_columns]

    def endogenous_data(self) -> np.array:
        return self.preprocessed_data[:, self.preprocessed_endogenous_columns]
    
    def shuffle_data(self) -> None:
        # Create and shuffle indices
        ind = np.arange(0, len(self.preprocessed_data), 1)
        ind_shuffle = copy.deepcopy(ind)
        np.random.shuffle(ind_shuffle)
    
        # Split into inputs (x) and outputs (y)
        self.preprocessed_shuffled_data = self.preprocessed_data[ind_shuffle]
    
    def _get_mean(self, data:np.array) -> np.array:
        return np.mean(data, axis=self.axis)

    def _get_std(self, data:np.array) -> np.array:
        return np.std(data, axis=self.axis)

    def _standardize(self, data:np.array, column_indexes:list) -> np.array:
        mean = self._get_mean(data=data) if self.transformed_mean is None else self.transformed_mean[column_indexes]
        std = self._get_std(data=data) if self.transformed_std is None else self.transformed_std[column_indexes]
        return (data - mean)/std

    def _destandardize(self, data:np.array, column_indexes:list) -> np.array:
        mean = self._get_mean(data=data) if self.transformed_mean is None else self.transformed_mean[column_indexes]
        std = self._get_std(data=data) if self.transformed_std is None else self.transformed_std[column_indexes]
        return data * std + mean

    def preprocess_data(self, dimension_reduction_model:str, dimension_reduction_n_components:int) -> None:
        self._preprocess_exogenous_data()
        self._preprocess_endogenous_data(
            dimension_reduction_model=dimension_reduction_model, 
            dimension_reduction_n_components=dimension_reduction_n_components)
    
    def _preprocess_exogenous_data(self) -> None:
        """
        preprocess exogenous variables
        """
        
        data = self.transformed_data[:, self.exogenous_columns]

        # standardize data
        z = self._standardize(data=data, column_indexes=self.exogenous_columns)

        self.preprocessed_exogenous_columns = self._add_preprocessed_data(z)

    def inverse_process_data(self) -> None:
        self._inverse_process_exogenous_data()
        self._inverse_process_endogenous_data()

    def _inverse_process_exogenous_data(self) -> None:
        """
        apply inverse transformations to exogenous variables. This function must be consistent with preprocess_exogenous_data()
        """

        data = self.preprocessed_data[:, self.preprocessed_exogenous_columns]

        # destandardize data
        z_inv = self._destandardize(data=data, column_indexes=self.preprocessed_exogenous_columns)

        # assign endogenous columns after increasing dimensionality
        self.exogenous_colums = list(range(z_inv.shape[1]))

        if self.exogenous_colums != []:
            if len(self.transformed_data.shape) < 2:
                self.transformed_data = z_inv
            else:
                self.transformed_data[:, self.exogenous_colums] = z_inv

    def _preprocess_endogenous_data(self, dimension_reduction_model:str, dimension_reduction_n_components:int) -> None:
        """
        preprocess endogenous variables
        """
        
        data = self.transformed_data[:, self.endogenous_columns]

        # standardize data
        z = self._standardize(data=data, column_indexes=self.endogenous_columns)

        # reduce dimensionality
        z_red = self.reduce_dimensionality(data=z,
                                           column_indexes=self.endogenous_columns,
                                           model=dimension_reduction_model,
                                           components=dimension_reduction_n_components)
        
        self.preprocessed_endogenous_columns = self._add_preprocessed_data(z_red)

    def _inverse_process_endogenous_data(self) -> None:
        """
        apply inverse transformations to endogenous variables. This function must be consistent with preprocess_endogenous_data()
        """

        data = self.preprocessed_data[:, self.preprocessed_endogenous_columns]

        # apply inverse of dimension reduction
        data_inv = self.inverse_reduce_dimensionality(data=data)

        # assign endogenous columns after increasing dimensionality
        start = len(self.exogenous_columns)
        self.endogenous_columns = list(range(start, data_inv.shape[1]))
        
        # destandardize data
        z_inv = self._destandardize(data=data_inv, column_indexes=self.endogenous_columns)

        if self.endogenous_columns != []:
            if len(self.transformed_data.shape) < 2:
                self.transformed_data = z_inv
            else:
                self.transformed_data[:, self.endogenous_columns] = z_inv

    def _add_preprocessed_data(self, preprocessed_data:np.array) -> list:
        if self.preprocessed_data is None:
            self.preprocessed_data = preprocessed_data
            return list(range(preprocessed_data.shape[1]))

        start = self.preprocessed_data.shape[1]
        end = self.preprocessed_data.shape[1] + preprocessed_data.shape[1]
        self.preprocessed_data = np.concat((self.preprocessed_data, preprocessed_data), axis=1)
        return list(range(start, end))
    
    def transform_data(self) -> None:
        transformed_data = self.raw_data.copy()
        transformed_data[:, self.endogenous_columns] = self._transform_endogenous_data()
        transformed_data[:, self.exogenous_columns] = self._transform_exogenous_data()
        
        # save statistics
        self.transformed_mean = self._get_mean(data=transformed_data) if self.transformed_mean is None else self.transformed_mean
        self.transformed_std = self._get_std(data=transformed_data)  if self.transformed_std is None else self.transformed_std
        self.transformed_data = transformed_data

    def inverse_transform_data(self) -> None:
        exogenous_data = self._inverse_transform_exogenous_data()
        endogenous_data = self._inverse_transform_endogenous_data()
        self.raw_data = np.concatenate((exogenous_data, endogenous_data), axis=1)
    
    def _transform_exogenous_data(self) -> np.array:
        return self.raw_data[:, self.exogenous_columns]

    def _inverse_transform_exogenous_data(self) -> np.array:
        return self.transformed_data[:, self.exogenous_columns]
    
    def _transform_endogenous_data(self) -> np.array:
        endogenous = self.raw_data[:, self.endogenous_columns]

        # divide by max across columns
        max_per_row = np.max(endogenous, axis=1)
        endogenous = endogenous / max_per_row[:,None]

        # apply log transform
        return np.log(endogenous)

    def _inverse_transform_endogenous_data(self) -> np.array:
        endogenous = self.transformed_data[:, self.endogenous_columns]
        
        # apply exponential
        return np.exp(endogenous)

    def reduce_dimensionality(self, data:np.array, column_indexes:list, model:str, components:int) -> np.array:
        """
        reduce dimensionality of given indexes via PCA or ICA

        Parameters:
        -----------
            column_indexes (list): column indexes to be used in the dimension reduction. This should be a subset of the columns in self.raw_data

        Returns:
        --------
            (np.array): data in the PCA/ICA space with reduced dimensionality
        """

        if self.dimension_reduction is None:
        
            if model == 'ICA':
                # Compute ICA
                model_obj = FastICA(n_components=components, whiten="arbitrary-variance")
        
            elif model == 'PCA':
                # For comparison, compute PCA
                model_obj = PCA(n_components=components)
                #print(np.cumsum(model_obj.explained_variance_ratio_))

            elif model == 'Sparse':
                # compute Sparse PCA
                model_obj = SparsePCA(n_components=components, random_state=0)

            elif model == 'KernelPCA':
                # compute Kernel PCA
                model_obj = KernelPCA(
                    n_components=components, 
                    kernel="poly", 
                    gamma=10, 
                    fit_inverse_transform=True, 
                    alpha=0.1
                )
    
            # reduce dimensionality
            transformed_data = model_obj.fit_transform(data)
            inverse = model_obj.inverse_transform
            transform = model_obj.transform
    
            # save dimension reduction info
            self.dimension_reduction = DimensionReduction(column_indexes=column_indexes,
                                                          transformed_data=transformed_data, 
                                                          inverse=inverse,
                                                          transform=transform)

        else:
            # transform data with stored dimension reduction model
            dimension_reduction = self.dimension_reduction
            transformed_data = dimension_reduction.transform(data)

            # replace transformed data in dimension reduction object
            dimension_reduction.transformed_data = transformed_data
            self.dimension_reduction = dimension_reduction

        return transformed_data

    def inverse_reduce_dimensionality(self, data:np.array) -> np.array:
        """
        apply inverse transform of dimension reduction
        """
        inverse_func = self.dimension_reduction.inverse
        return inverse_func(data)

def get_inverse_statistic(y, train_data_obj):
    
    # create data statistics
    y_ds = DataStatistics()
    y_ds.preprocessed_data = y.cpu().detach().numpy()
    y_ds.preprocessed_exogenous_columns = []
    y_ds.preprocessed_endogenous_columns = list(range(len(y[0])))
    y_ds.transformed_mean = train_data_obj.transformed_mean[train_data_obj.endogenous_columns]
    y_ds.transformed_std = train_data_obj.transformed_std[train_data_obj.endogenous_columns]
    y_ds.dimension_reduction = train_data_obj.dimension_reduction
    y_ds.inverse_process_data()
    y_ds.inverse_transform_data()

    return y_ds
    
def modeltrain_loss(lr, epochs, x, y, valid_x, valid_y, model, y_weights, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr)  # Adam optimizer
    loss_fn = torch.nn.L1Loss(reduction='mean')  # L1 loss for gradient computation
    training_params = TrainingParameters(epochs=epochs, patience=patience, model=model)

    # Add a progress bar
    with tqdm(total=epochs, desc="Training Progress", unit="epoch") as pbar:
        for k in range(epochs):
            
            optimizer.zero_grad()  # Clear gradients from the previous step
            
            y_pred = model(x)  # Forward pass for training data
            valid_pred = model(valid_x)  # Forward pass for validation data
            
            # Loss used for gradient calculation
            loss = loss_fn(y_pred * y_weights, y * y_weights)
            
            train_loss = calculate_loss(actual=y, prediction=y_pred)
            valid_loss = calculate_loss(actual=valid_y, prediction=valid_pred)
            
            loss.backward()  # Backpropagate the gradient
            optimizer.step()  # Update model parameter

            # update progress bar
            pbar = update_progress_bar(
                pbar=pbar,
                train_loss=train_loss,
                valid_loss=valid_loss,
                no_improvement=training_params.no_improvement
            )

            # update results
            training_params.update_results(
                k=k,
                train_loss=train_loss,
                valid_loss=valid_loss,
                model_state=model.state_dict()
            )

            # If no improvement for 'patience' epochs, stop training
            if training_params.exceeded_patience():
                print(f"\nEarly stopping at epoch {k+1}. Validation loss has not improved for {training_params.patience} epochs.")
                break

            # Free memory by deleting intermediate variables
            del loss, y_pred
            
    # Restore the best model state after training
    training_params.restore_best_model(k)
    
    return training_params

def calculate_loss(actual:np.array, prediction:np.array) -> float:
    """
    calculate loss
    """
    return torch.mean(torch.abs(actual - prediction)).item()
    
def update_progress_bar(pbar, train_loss, valid_loss, no_improvement):
    # Update the progress bar with the current epoch and losses
    pbar.set_postfix(
        train_loss=train_loss, 
        valid_loss=valid_loss, 
        patience_count=no_improvement
    )
    pbar.update(1)  # Increment the progress bar
    
    return pbar

def performance_sigma_point(model, x, valid_x, y, valid_y, train_data_obj):
    """Plot the performance of a neural network model.

    Parameters:
        model: Trained neural network model.
        x: Training input data.
        valid_x: Validation input data.
        y: Training output data.
        valid_y: Validation output data.
        k_mean: Mean normalization values.
        k_std: Standard deviation normalization values.
    """
    # plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['font.size'] = 15
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['mathtext.fontset'] = 'stix'  # ensures it can math compatibility with symbols in your code without erroring fix no cursive_fontsystem


    y_pred_train = model(x)
    y_pred_test = model(valid_x)

    # create data statistics
    y_ds = get_inverse_statistic(
        y=y, 
        train_data_obj=train_data_obj)

    y_valid_ds = get_inverse_statistic(
        y=valid_y, 
        train_data_obj=train_data_obj)

    y_pred_test_ds = get_inverse_statistic(
        y=y_pred_test, 
        train_data_obj=train_data_obj)

    kappa_mean = y_ds._get_mean(data=y_ds.raw_data)
    n = len(kappa_mean)

    yptestcpu = y_pred_test_ds.raw_data
    ytestcpu = y_valid_ds.raw_data
    ytestcpu = y_valid_ds.raw_data

    plt.figure(figsize=(15, 10))

    ind = np.arange(0, n)
    ind_tick = np.arange(1, 17)[::-1]

    # Subplot 1: Boxplot of network output differences
    plt.subplot(1, 4, 1)
    for i in range(16):
        plt.boxplot(ytestcpu[:, i] - yptestcpu[:, i], vert=False, positions=[i], showfliers=False, whis=(5, 95), widths=0.5)
    plt.xlim([-2.0, 2.0])
    plt.yticks(ind, ind_tick)
    plt.title(r'(a) Output of network $\mathcal{N}_1$ ')
    plt.ylabel('Node')

    # Subplot 2: Boxplot of shape function differences
    plt.subplot(1, 4, 2)
    for i in range(16):
        plt.boxplot(kappa_mean[i] + ytestcpu[:,i] - yptestcpu[:,i],
                    vert=False, positions=[i], showfliers=False, whis=(5, 95), widths=0.5)
    plt.yticks([])
    plt.title(r'(b) Shape function $g(\sigma)$')
    plt.xlabel(r'$g(\sigma)$')

    # Subplots 3 & 4: Histograms
    k12 = 15
    for k in range(16):
        
        plt.subplot(16, 4, 4 * k + 3)
        
        vals, binss = get_hist(ytestcpu[:, k12])
        plt.plot(binss, vals, color='blue')

        vals, binss = get_hist(yptestcpu[:, k12])
        plt.plot(binss, vals, color='red')
        adjust_plot(k=k, title='(c) Probability density histogram')

        
        plt.subplot(16, 4, 4 * k + 4)
        
        vals, binss = get_hist2(ytestcpu[:, k12] - yptestcpu[:, k12])
        plt.plot(binss, vals, color='green')
        adjust_plot(k=k, title='(d) Error histogram ')

        k12 -= 1

    plt.tight_layout()

def adjust_plot(k, title):
    if k < 15:
        plt.xticks([])
    plt.yticks([])
    if k == 0:
        plt.title(title)
    return plt
    

def get_hist(y):
    """Get histogram values for normalized data."""
    vals, binss = np.histogram(y, range=(0, 1.2), bins=100)
    return vals, 0.5 * (binss[0:-1] + binss[1:])

def get_hist2(y):
    """Get histogram values for error data."""
    vals, binss = np.histogram(y, range=(-0.2, 0.2), bins=100)
    return vals, 0.5 * (binss[0:-1] + binss[1:])
