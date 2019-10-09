import pandas as pd
import numpy as np
import torch
import utils
from nn_architectures import train_MDAD



def main():
    # Load data
    phen_samples, trait_labels, phen_data = utils.load_and_process_phenotype("/media/wcasazza/DATA2/wcasazza/ROSMAP/ROSMAP_PHEN.csv")
    exp_samples, exp_data = utils.load_and_process_expression("/media/wcasazza/DATA2/wcasazza/ROSMAP/normalized_dlpfc_v2.tsv")
    exp_ord, phen_ord = utils.match_samples(exp_samples, phen_samples)
    exp_samples, exp_data =  exp_samples[exp_ord], exp_data[exp_ord,:]
    phen_samples, phen_data = phen_samples[phen_ord], phen_data[phen_ord,:]
    model, loss = train_MDAD(
        torch.tensor(exp_data.astype(np.float64)),
        torch.tensor(phen_data.astype(np.float64)),
        plot_loss=False,
        #save_loss="/home/wcasazza/saved_mdad_norm_phen.png",
        #save_model = "/home/wcasazza/saved_mdad_norm_phen.pt",
        use_validation=False,
        verbose = True)



if __name__ == "__main__":
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    main()