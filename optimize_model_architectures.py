import csv
import torch
import nn_architectures as nets
import utils
from sklearn.model_selection import KFold
from itertools import product
def main():
    phen_samples, trait_labels, phen_data = utils.load_and_process_phenotype("/media/wcasazza/DATA2/wcasazza/ROSMAP/ROSMAP_PHEN.csv")
    exp_samples, exp_data = utils.load_and_process_expression("/media/wcasazza/DATA2/wcasazza/ROSMAP/normalized_dlpfc_v2.tsv")
    exp_ord, phen_ord = utils.match_samples(exp_samples, phen_samples)
    exp_samples, exp_data =  exp_samples[exp_ord], exp_data[exp_ord,:]
    phen_samples, phen_data = phen_samples[phen_ord], phen_data[phen_ord,:]
    exp_data = torch.tensor(exp_data).double()
    phen_data = torch.tensor(phen_data).double()
    # Cross Validation
    kf = KFold(n_splits=5, shuffle=True)
    sizes = range(10,500,50)
    layers = range(1,5)
    param_list = [{"input_size": 500, "hidden_nodes": size,"hidden_layers": layer} for size,layer in product(sizes,layers)]
    result = []
    loss = torch.nn.MSELoss()
    for params in param_list:
        tmp_result = {}
        tmp_result.update(params)
        losses = []
        for train,test in kf.split(exp_data):
            model = nets.train_MDAD(exp_data[train,:], phen_data[train,:],params)
            predictions = model(exp_data[test,:])
            cur_loss = [loss(predictions[i].flatten(),phen_data[test,i].flatten()) for i in range(len(predictions))]
            losses.append(sum(cur_loss))

        tmp_result['cv_loss_mdad'] = sum(losses) / float(len(losses))
        linear_losses = []
        mlp_losses = []
        for train,test in kf.split(exp_data):
            linear_models, mlp_models = nets.train_single_models(exp_data[train,:], phen_data[train,:],params)
            linear_predictions = [model(exp_data[test,:]) for model in linear_models]
            mlp_predictions = [model(exp_data[test,:]) for model in mlp_models]
            linear_loss = [loss(linear_predictions[i].flatten(),phen_data[test,i].flatten()) for i in range(len(linear_predictions))]
            mlp_loss = [loss(mlp_predictions[i].flatten(),phen_data[test,i].flatten()) for i in range(len(mlp_predictions))]
            linear_losses.append(linear_loss)
            mlp_losses.append(mlp_loss)
        for i in range(len(linear_loss)):
            tmp_result[f'cv_loss_linear_{i}'] = sum([losses[i] for losses in linear_losses ]) / float(len(linear_losses))
            tmp_result[f'cv_loss_mlp_{i}'] = sum([losses[i] for losses in mlp_losses ]) / float(len(mlp_losses))
        result.append(tmp_result)
    with open('cv_data.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames = list(tmp_result[0].keys()))
        writer.writeheader()
        writer.writerows(params)
    return 0
    # write out data


if __name__ == "__main__":
    main()