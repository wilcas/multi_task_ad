import pandas as pd
import numpy as np
from functools import reduce
from sklearn.decomposition import PCA

def norm01(vec):
    return (vec - min(vec)) / (max(vec) - min(vec))

def load_and_process_expression(fname):
    expression = pd.read_csv(fname,sep = "\t")
    expression = expression.iloc[:,1:expression.shape[1]].apply(norm01,axis = 1)
    f = PCA(n_components=500)
    f.fit(expression.iloc[:,1:expression.shape[1]])
    return expression.columns.to_numpy(), f.components_.T

def load_and_process_phenotype(fname):
    phenotype = pd.read_csv(fname,index_col=0)[["ceradsc", "amyloid","tangles", "nft", "braaksc", "np"]]
    phenotype = phenotype.apply(norm01,axis = 1)
    return phenotype.index.to_numpy().astype(str), phenotype.columns.to_numpy(), phenotype.to_numpy()


def match_samples(*samples):
    """Sort data by sample ids"""
    shared = reduce(np.intersect1d, samples)
    shared_idx = []
    for sample_vec in samples:
        indices = sample_vec.argsort()
        to_keep = np.array([elem in shared for elem in  sample_vec[indices]])
        shared_idx.append(np.extract(to_keep, indices))
    return shared_idx

def integrated_gradients(model,base_data,input_data):
    base_outs = model(base_data)
    baselines = [torch.mean(out) for out in base_outs]
    activations = model(input_data)
    return [activation - baseline for (activation, baseline) in zip(activations,baselines)]

def get_pre_split(model,data):
    activations = []
    def hook(module, input, output):
        activations.append(output.detach())
    model.base_net[1].register_forward_hook(hook)
    model(data)
    return activations