import numpy as np
import scipy
import asyncio, time
import pandas as pd
from queue import LifoQueue
from typing import List

class Node(object):
    def __init__(self, parent: Node, num_samples, is_left=False):
        self.parent: Node = parent
        self.left_child: Node
        self.right_child: Node
        self.is_left = None
        self.weight = None
        self.split_feature = None
        self.split_val = None
        self.sparse_dir_left = None
        self.num_samples = num_samples

    @property
    def is_leaf(self):
        return not bool(self.left_child) and not bool(self.right_child)
    
    def calculate_weight(self, gradient, hessian, lmbd):
        if isinstance(hessian, np.ndarray):
            hessian[hessian == 0.] += 1e-8
        return np.sum(gradient)/(np.sum(hessian) + lmbd)
    
    def calculate_quality(self, G, H, lmbd):
        if isinstance(H, np.ndarray):
            H[H == 0.] += 1e-8
        return np.square(G)/(H + lmbd)
    
    def calculate_gain(self, G, H, G_l, H_l, G_r, H_r, lmbd):
        return self.calculate_quality(G_l, H_l, lmbd) +\
                self.calculate_quality(G_r, H_r, lmbd) - \
                self.calculate_quality(G, H, lmbd)
    
    def find_split(self, data, data_idx, lmbd, num_featuress=None, min_samples_leaf=1, eps=0.3, approximate=False):
        feature_idx = data.sample_features(num_featuress)
        G_node, H_node = data.get_gradient_hessian(data_idx)

        # get a mtrix with n_features x n_points with gains for sorted feature values
        # gradient_matrix = data.get_gradient_matrix(feature_idx, data_idx, aproximate)
        # hessian_matrix = data.get_hessian_matrix(feature_idx, data_idx, aproximate)
        # split_value_matrix = data.get_split_value_matrix(feature_idx, data_idx, aproximate)
        # print (data_idx)
        if min_samples_leaf > len(data_idx)/2:
            return None, None, None
        if not approximate:
            gradient_matrix, hessian_matrix, split_value_matrix = data.get_GHS_matrices(feature_idx, data_idx, approximate)
        if approximate:
            gradient_matrix, hessian_matrix, split_value_matrix, num_samples_matrix = \
                data.get_GHS_matrices_approx(feature_idx, data_idx, eps)
        G_l_matrix = np.nancumsum(gradient_matrix, axis=1)
        H_l_matrix = np.nancumsum(hessian_matrix, axis=1)
        G_l_matrix[np.isnan(gradient_matrix)] = np.nan
        

        gain_matrix = self.calculate_gain(G_node, H_node, 
                                          G_l_matrix, H_l_matrix, 
                                          G_node - G_l_matrix, H_node - H_l_matrix,
                                          lmbd)
        
        # removing splits that may result in less than min_samples_leaf in either child
        if not approximate:
            gain_matrix = np.hstack([gain_matrix[:, min_samples_leaf - 1: -min_samples_leaf], gain_matrix[:, -1:]])
            if min_samples_leaf > 1:
                split_value_matrix = np.hstack([split_value_matrix[:, min_samples_leaf - 1: 1 - min_samples_leaf]])
        else:
            num_samples_matrix = np.nancumsum(num_samples_matrix, axis=1)
            last_col = gain_matrix[:, -1:].copy()
            gain_matrix[num_samples_matrix < min_samples_leaf] = np.nan
            # last_col = gain_matrix[:, -1:]
            gain_matrix[-1*(num_samples_matrix - num_samples_matrix[:, -1:]) < min_samples_leaf] = np.nan
            gain_matrix[:, -1:] = last_col
        
        # print (gain_matrix)
        best_split = np.unravel_index(np.nanargmax(gain_matrix), shape=gain_matrix.shape)
        # print (feature_idx[best_split[0]], best_split[1])
        if not approximate and best_split[1] >= len(data_idx) - (2 * min_samples_leaf) + 2 - 1:
            return None, None, None
        elif approximate and best_split[1] >= gain_matrix.shape[1] - 1:
            return None, None, None
        # find the value of feature split from split_index
        split_dir = True
        if best_split[0] >= len(feature_idx):
            split_dir = False ## nans go right
        
        return feature_idx[best_split[0] % len(feature_idx)], split_value_matrix[best_split], split_dir
    
    def predict(self, X, idx, verbose=False):
        if self.is_leaf:
            if verbose: print ('leaf:', self.weight, 'num_samples:', self.num_samples)
            return self.weight
        else:
            # print (idx)
            if verbose: print('idx:',idx)
            out = np.zeros(len(idx), dtype='float')
            # split_left_right = X[idx, self.split_feature] < self.split_val
            
            split_left_right = np.logical_or(X[idx, self.split_feature] < self.split_val, 
                                       np.isnan(X[idx, self.split_feature]) & self.sparse_dir_left)
            left_idx = idx[split_left_right]
            if verbose: print ('left:', left_idx)
            right_idx = idx[~(split_left_right)]
            if verbose: print ('right:',right_idx)
            out_left_idx = np.where(split_left_right)[0]
            out_right_idx = np.where(~(split_left_right))[0]

            if verbose: print("left ->")
            out[out_left_idx] = self.left_child.predict(X, left_idx, verbose)
            if verbose: print ("right ->")
            out[out_right_idx] = self.right_child.predict(X, right_idx, verbose)
            if verbose: print (out, 'outidx', out_left_idx, out_right_idx, 'num_samples:', self.num_samples)
            return out

class Tree(object):
    def __init__(self, obj='sq_error', max_depth=4, num_features=None, min_samples_leaf=1, eps=0.3):
        self.residuals = None
        self.hessians = None
        self.root = None
        self.max_depth = max_depth
        self.num_features = num_features
        self.min_samples_leaf = min_samples_leaf
        self.predictions = None
        self.obj = obj
        self.eps = eps

    def get_train_predictions(self):
        # if self.obj == "sq_error":
            return self.predictions
        # elif self.obj == 'bin_logistic':
        #     return 1.0/(1.0 + np.exp(-1.0 * self.predictions))
    
    def update_data_vars(self, data_obj):
        data_obj.update_residuals_hessians(self.residuals, self.hessians)

    def grow_tree(self, data, gradients, hessians, lmbd, approximate=False):
        self.residuals = gradients
        self.hessians = hessians
        self.predictions = np.zeros_like(self.residuals)
        self.update_data_vars(data_obj=data)
        self.root = Node(None, data.num_samples)
        depth = 0
        node_stack = LifoQueue()
        self.root.weight = self.root.calculate_weight(gradients, hessians, lmbd)
        node_stack.put((self.root, depth, np.arange(data.num_samples)))
        while (not node_stack.empty()):
            node, depth, data_idx = node_stack.get()
            if depth < self.max_depth:
                left_data_idx, right_data_idx = \
                    self.split_node(node, data, data_idx, lmbd, approximate)
                # print (left_data_idx, right_data_idx)
                if node.is_leaf and left_data_idx is None: # leaf node
                    self.predictions[data_idx] = node.weight
                else:
                    node.left_child.weight = node.left_child.calculate_weight(data.residuals[left_data_idx], 
                                                                            data.hessians[left_data_idx], lmbd)
                    node.right_child.weight = node.right_child.calculate_weight(data.residuals[right_data_idx], 
                                                                            data.hessians[right_data_idx], lmbd)
                    node_stack.put((node.left_child, depth + 1, left_data_idx))
                    node_stack.put((node.right_child, depth + 1, right_data_idx))
            else: #leaf node
                self.predictions[data_idx] = node.weight

    def split_node(self, node, data, data_idx, lmbd, approximate=False):
        split_feature, split_value, split_dir = node.find_split(data, data_idx, 
                                                     lmbd, self.num_features, 
                                                     min_samples_leaf=self.min_samples_leaf, 
                                                     eps=self.eps,
                                                     approximate=approximate)
        if split_feature is None:
            return None, None
        node.split_feature =  split_feature
        node.split_val = split_value
        node.sparse_dir_left = split_dir
        # split_data_idx = data.X[data_idx, split_feature] < split_value
        split_data_idx = np.logical_or(data.X[data_idx, split_feature] < split_value, 
                                       np.isnan(data.X[data_idx, split_feature]) & split_dir)
        node.left_child = Node(node, split_data_idx.sum(), is_left=True)
        node.right_child = Node(node, data_idx.shape[0] - split_data_idx.sum())
        return data_idx[split_data_idx], data_idx[~split_data_idx]
    
    def predict(self, X, verbose=False):
        idx = np.arange(X.shape[0])
        return self.root.predict(X, idx, verbose)


class XGBoost(object):
    def __init__(self, obj='sq_error', # or 'logistic'
                 n_estimators=100, lmbd=0, eta=0.3, 
                 max_depth=4, min_samples_leaf=1,
                 num_features=None, eps=0.3, approximate=False,
                 case_weights=None):
        self.n_estimators = n_estimators
        self.eta = eta
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.num_features = num_features
        self.approximate = approximate
        self.lmbd = lmbd
        self.learners: List[Tree]
        self.obj = obj
        self.eps = eps
        self.case_weights = case_weights if case_weights is not None else 1.0

    def __call__(self):
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__ + " n_estimators: " + str(self.n_estimators)

    def train(self, data, verbose=True):
        losses = []
        self.learners: List[Tree] = []
        predictions = np.zeros((data.num_samples,))
        for i in tqdm(range(self.n_estimators)):
            gradients, hessians = self.calculate_gradients(predictions, data)
            learner = Tree(obj=self.obj, 
                           max_depth=self.max_depth, 
                           num_features=self.num_features,
                           min_samples_leaf=self.min_samples_leaf,
                           eps=self.eps)
            learner.grow_tree(data, gradients, hessians, self.lmbd, approximate=self.approximate)
            predictions += self.eta * learner.get_train_predictions()
            self.learners.append(learner)

            ## validate 
            train_loss = self.loss(data.y, predictions)
            losses.append(train_loss)
            if verbose:
                print ("Training loss:", train_loss)
        return losses

    def predict(self, X):
        return self.eta * sum([l.predict(X) for l in self.learners])
    
    def sigmoid(self, X):
        return 1.0/(1.0 + np.exp(-1.0 * X))
    
    def loss(self, y, preds):
        if self.obj == 'sq_error':
            return np.mean(np.square(y - preds))
        elif self.obj == 'logistic':
            preds = self.sigmoid(preds)
            return -1 * np.mean((y * np.log(preds)) + ((1 - y) * np.log(1 - preds)))
    
    def validate(self, X, y):
        preds = self.predict(X)
        preds = np.where(preds > 1.0e-10, preds, 1.0e-10)
        return self.loss(y, preds)
    
    def calculate_gradients(self, predictions, data):
        if self.obj == 'sq_error':
            residuals = (data.y - predictions) * self.case_weights
            hessian = np.ones(data.y.shape[0]) * self.case_weights
        elif self.obj == 'logistic':
            probs = self.sigmoid(predictions)
            residuals = (data.y - probs) * self.case_weights
            hessian = probs * (1.0 - probs) * self.case_weights
        return residuals, hessian


class Data(object):
    # holds columns in sorted orders to reduce 
    # cost of sorting each time 
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.data = []
        self.sort_features(self.X)
        self.residuals = None
        self.hessians = None

    @property
    def num_samples(self):
        return self.X.shape[0]
    
    def sample_features(self, num_features):
        if num_features:
            return np.random.choice(np.arange(self.X.shape[1]), num_features, replace=False)
        else:
            return np.random.choice(np.arange(self.X.shape[1]), self.X.shape[1], replace=False)
    
    def sort_features(self, X):
        self.sorted_idx = np.argsort(X, axis=0)
        self.nan_idx = {f_id: set(np.where(np.isnan(X[:, f_id]))[0]) for f_id in range(X.shape[1])}

    def update_residuals_hessians(self, residuals, hessians):
        self.residuals = residuals
        self.hessians = hessians
    
    def get_GHS_matrices(self, feature_idx, data_idx, eps, aproximate=False):
        # may add multiprocessing here
        data_idx = set(data_idx)
        grad_matrix = np.zeros((len(feature_idx) * 2, len(data_idx)))
        grad_matrix[:, :] = np.nan
        
        hess_matrix = np.zeros((len(feature_idx) * 2, len(data_idx)))
        split_val_matrix = np.zeros((len(feature_idx) * 2, len(data_idx) - 1))
        for i, f_id in enumerate(feature_idx):
            grad_row = []
            hess_row = []
            split_val_row = []
            idx_nan = []
            mask = np.in1d(self.sorted_idx[:, f_id], list(data_idx))
            sorted_idx = self.sorted_idx[mask, f_id]
            # for d_id in self.sorted_idx[:, f_id]:
            for d_id in sorted_idx:
                if d_id in data_idx:
                    if d_id in self.nan_idx.get(f_id, {}):
                        idx_nan.append(d_id)
                    else:
                        grad_row.append(self.residuals[d_id])
                        hess_row.append(self.hessians[d_id])
                        split_val_row.append(self.X[d_id, f_id])
            
            if len(idx_nan) == 0:
                grad_matrix[i] = np.array(grad_row, dtype='float')
                hess_matrix[i] = np.array(hess_row, dtype='float')
                split_val_row_a = np.array(split_val_row, dtype='float')
                split_val_matrix[i] = (split_val_row_a[:-1] + split_val_row_a[1:])/2
            else:
                grad_matrix[len(feature_idx) + i] = np.array(grad_row + [None] * len(idx_nan), dtype='float')
                hess_matrix[len(feature_idx) + i] = np.array(hess_row + [None] * len(idx_nan), dtype='float')
                split_val_row_a = np.array(split_val_row + [None] * len(idx_nan), dtype='float')
                split_val_matrix[len(feature_idx) + i] = (split_val_row_a[:-1] + split_val_row_a[1:])/2

                if len(grad_row) > 0:
                    grad_row[0] += self.residuals[idx_nan].sum()
                    hess_row[0] += self.hessians[idx_nan].sum()
                grad_matrix[i] = np.array([None] * len(idx_nan) + grad_row, dtype='float')
                hess_matrix[i] = np.array([None] * len(idx_nan) + hess_row, dtype='float')
                split_val_row_a = np.array([None] * len(idx_nan) + split_val_row, dtype='float')
                split_val_matrix[i] = (split_val_row_a[:-1] + split_val_row_a[1:])/2

        if np.sum(~np.isnan(grad_matrix[len(feature_idx):])) == 0:
            return grad_matrix[:len(feature_idx)], hess_matrix[:len(feature_idx)], split_val_matrix[:len(feature_idx)]
        
        return grad_matrix, hess_matrix, split_val_matrix
    
    def get_GHS_matrices_approx(self, feature_idx, data_idx, eps):
        data_idx = set(data_idx)
        # grad_matrix = np.zeros((len(feature_idx) * 2, len(data_idx)))
        grad_matrix = []
        # grad_matrix[:, :] = np.nan
        
        # hess_matrix = np.zeros((len(feature_idx) * 2, len(data_idx)))
        hess_matrix = []
        # split_val_matrix = np.zeros((len(feature_idx) * 2, len(data_idx) - 1))
        split_val_matrix = []
        num_samples_matrix = []
        idx_nan_list = []
        max_width = -1
        total_weight = self.hessians[list(data_idx)].sum()
        for i, f_id in enumerate(feature_idx):
            # total_weight = self.hessians[list(data_idx - self.nan_idx.get(f_id, {}))].sum()
            total_weight_f = total_weight - self.hessians[list(data_idx.intersection(self.nan_idx.get(f_id, {})))].sum()
            # print ('*'*100, total_weight_f)
            grad_row, hess_row, split_val_row, idx_nan, num_samples_row = \
                self.get_GHS_row_approx(data_idx, f_id, eps * total_weight_f)
            grad_matrix.append(grad_row)
            hess_matrix.append(hess_row)
            split_val_matrix.append(split_val_row)
            num_samples_matrix.append(num_samples_row)
            idx_nan_list.append(idx_nan)
            max_width = max(max_width, len(hess_row))
        grad_matrix = np.array([[None] * (max_width - len(grad_row)) + grad_row \
                                for grad_row in grad_matrix], 
                               dtype='float')
        hess_matrix = np.array([[None] * (max_width - len(hess_row)) + hess_row \
                                for hess_row in hess_matrix], 
                               dtype='float')
        split_val_matrix = np.array([[None] * (max_width - len(split_val_row)) + split_val_row \
                                     for split_val_row in split_val_matrix], 
                                    dtype='float')
        num_samples_matrix = np.array([[None] * (max_width - len(num_samples_row)) + num_samples_row \
                                       for num_samples_row in num_samples_matrix], 
                                      dtype='float')
        r_grad_matrix = np.zeros_like(grad_matrix)
        r_grad_matrix[:, :] = np.nan

        r_hess_matrix = np.zeros_like(hess_matrix)
        r_num_samples_matrix = np.zeros_like(num_samples_matrix)

        for i, idx_nan in enumerate(idx_nan_list):
            if len(idx_nan) != 0 and int(np.isnan(grad_matrix[i]).sum()) < grad_matrix.shape[1]:
                r_grad_matrix[i] = grad_matrix[i].copy()
                r_hess_matrix[i] = hess_matrix[i].copy()
                r_num_samples_matrix[i] = num_samples_matrix[i].copy()

                grad_matrix[i, int(np.isnan(grad_matrix[i]).sum())] += self.residuals[idx_nan].sum()
                hess_matrix[i, int(np.isnan(hess_matrix[i]).sum())] += self.hessians[idx_nan].sum()
                num_samples_matrix[i, int(np.isnan(num_samples_matrix[i]).sum())] += len(idx_nan)

        if np.sum(~np.isnan(grad_matrix[len(feature_idx):])) != 0:
            return (np.vstack([grad_matrix, r_grad_matrix]), 
                    np.vstack([hess_matrix, r_hess_matrix]), 
                    np.vstack([split_val_matrix, split_val_matrix]),
                    np.vstack([num_samples_matrix, r_num_samples_matrix]))
        
        return grad_matrix, hess_matrix, split_val_matrix, num_samples_matrix
    
    def get_GHS_row_approx(self, data_idx, f_id, eps):
        grad_row = []
        hess_row = []
        split_val_row = []
        num_samples = []
        idx_nan = []
        h_current = 0.0
        g_current = 0.0
        ns_current = 0
        mask = np.in1d(self.sorted_idx[:, f_id], list(data_idx))
        sorted_idx = self.sorted_idx[mask, f_id]
        # for d_id in self.sorted_idx[:, f_id]:
        for d_id in sorted_idx:
            if d_id in data_idx:
                if d_id in self.nan_idx.get(f_id, {}):
                    idx_nan.append(d_id)
                else:
                    if h_current + self.hessians[d_id] < eps or ns_current <= 0:
                        h_current += self.hessians[d_id]
                        g_current += self.residuals[d_id]
                        ns_current += 1
                    else:
                        hess_row.append(h_current)
                        grad_row.append(g_current)
                        split_val_row.append(self.X[d_id, f_id])
                        num_samples.append(ns_current)
                        h_current = self.hessians[d_id]
                        g_current = self.residuals[d_id]
                        ns_current = 1
        if h_current > 0.0:
            grad_row.append(g_current)
            hess_row.append(h_current)
            num_samples.append(ns_current)
        return grad_row, hess_row, split_val_row, idx_nan, num_samples
    
    def get_gradient_hessian(self, data_idx):
        return self.residuals[data_idx].sum(), self.hessians[data_idx].sum()
    
    def get_gradient_matrix(self, feature_idx, data_idx, aproximate=False):
        # may add multiprocessing here
        grad_matrix = np.zeros(len(feature_idx), len(data_idx))
        for i, f_id in enumerate(feature_idx):
            grad_matrix[i] = np.array([self.residuals[d_id] for d_id in self.sorted_idx[:, f_id] if d_id in data_idx])
        return grad_matrix
    
    def get_hessian_matrix(self, feature_idx, data_idx, aproximate=False):
        # may add multiprocessing here
        hess_matrix = np.zeros(len(feature_idx), len(data_idx))
        for i, f_id in enumerate(feature_idx):
            hess_matrix[i] = np.array([self.hessians[d_id] for d_id in self.sorted_idx[:, f_id] if d_id in data_idx])
        return hess_matrix
    
    def get_split_value_matrix(self, feature_idx, data_idx, aproximate=False):
        # may add multiprocessing here
        split_val_matrix = np.zeros(len(feature_idx), len(data_idx) - 1)
        for i, f_id in enumerate(feature_idx):
            split_val_row = np.array([self.X[d_id, f_id] for d_id in self.sorted_idx[:, f_id] if d_id in data_idx])
            split_val_matrix[i] = (split_val_row[1:] + split_val_row[:-1])/2
        return split_val_matrix
    
np.random.seed(11)
sample_data_X = np.vstack([np.random.normal(loc=[1, 0, 1], scale=1, size=[5,3]),
                         np.random.normal(loc=[0, 0, 1], scale=1, size=[5,3]),
                         np.random.normal(loc=[1, 1, 0], scale=1, size=[5,3])])

# print (sample_data_X)

sample_data_X.shape

sample_data_X[[14,13],0] = None
sample_data_y = np.array([0]*5 + [1]*5 + [2]*5)

data = Data(sample_data_X, sample_data_y)
data.X.shape

xgb = XGBoost(lmbd=1, max_depth=5, min_samples_leaf=4, approximate=True, eps=0.01)

xgb.train(data)

preds = xgb.predict(data.X)
print (preds)
print ('train_rmse:', np.sqrt(((preds - data.y)**2).mean()))

sample_data_X = np.vstack([np.random.normal(loc=[1, 0, 1], scale=1, size=[5,3]),
                         np.random.normal(loc=[0, 0, 1], scale=1, size=[5,3]),
                         np.random.normal(loc=[1, 1, 0], scale=1, size=[5,3])])
sample_data_y = np.array([0]*5 + [1]*5 + [2]*5)

data_test = Data(sample_data_X, sample_data_y)

preds = xgb.predict(data_test.X)
print (preds)
print ('test_rmse:', np.sqrt(((preds - data_test.y)**2).mean()))



