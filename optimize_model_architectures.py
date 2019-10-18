import csv
import torch
import joblib
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
    exp_data = torch.tensor(exp_data).double()[:300,:]
    phen_data = torch.tensor(phen_data).double()[:300,:]
    # Cross Validation
    kf = KFold(n_splits=5, shuffle=True)
    sizes = range(10,500,50)
    layers = range(1,5)
    flag = True
    param_list = [{"hidden_nodes": size,"hidden_layers": layer} for size,layer in product(sizes,layers)]
    loss = torch.nn.MSELoss()
    for params in param_list:
        tmp_result = {}
        tmp_result.update(params)
        losses = []
        linear_losses = []
        mlp_losses = []
        for train,test in kf.split(exp_data):
            test_masks = [~torch.isnan(phen_data[test,i]) for i in range(phen_data.shape[1])]
            linear_models, mlp_models = nets.train_single_models(exp_data[train,:], phen_data[train,:],params)
            linear_predictions = [model(exp_data[test,:]) for model in linear_models]
            mlp_predictions = [model(exp_data[test,:]) for model in mlp_models]
            linear_loss = [loss(linear_predictions[i].flatten()[test_masks[i]],phen_data[test,i].flatten()[test_masks[i]]) for i in range(len(linear_predictions))]
            mlp_loss = [loss(mlp_predictions[i].flatten()[test_masks[i]],phen_data[test,i].flatten()[test_masks[i]]) for i in range(len(mlp_predictions))]
            linear_losses.append(linear_loss)
            mlp_losses.append(mlp_loss)
        for i in range(len(linear_loss)):
            tmp_result[f'cv_loss_linear_{i}'] = sum([losses[i].detach().numpy() for losses in linear_losses ]) / float(len(linear_losses))
            tmp_result[f'cv_loss_mlp_{i}'] = sum([losses[i].detach().numpy() for losses in mlp_losses ]) / float(len(mlp_losses))
        with joblib.parallel_backend('loky',n_jobs=5):
            models = joblib.Parallel()(
                joblib.delayed(nets.train_MDAD)(exp_data[train,:], phen_data[train,:],params)
                for train,test in kf.split(exp_data)
            )
        for (i,(train,test)) in enumerate(kf.split(exp_data)):
            test_masks = [~torch.isnan(phen_data[test,i]) for i in range(phen_data.shape[1])]
            model = models[i]
            predictions = model(exp_data[test,:])
            cur_loss = [loss(predictions[j].flatten()[test_masks[j]],phen_data[test,j].flatten()[test_masks[j]]) for j in range(len(predictions))]
            losses.append(sum(cur_loss).detach().numpy())

        tmp_result['cv_loss_mdad'] = sum(losses) / float(len(losses))
        with open('cv_data_redo.csv', 'a') as f:
            writer = csv.DictWriter(f, fieldnames = list(tmp_result.keys()))
            if flag:
                writer.writeheader()
                flag = False
            writer.writerow(tmp_result)
    return 0
    # write out data


if __name__ == "__main__":
    main()
