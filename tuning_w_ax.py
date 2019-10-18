import nn_architectures as nets
import torch
import utils
from ax.service.ax_client import AxClient
from ax.utils.measurement.synthetic_functions import branin


def mdad_loss(exp_data, phen_data,hidden_layers,hidden_nodes,lr, epochs, batch_size):
    params = {
        'hidden_layers': hidden_layers,
        'hidden_nodes': hidden_nodes
    }
    model = nets.train_MDAD(exp_data,phen_data, params, epochs=epochs,lr=lr,batch_size=batch_size)
    loss = torch.nn.MSELoss()
    outputs = model(exp_data)
    nans = [torch.isnan(phen_data[:,i]) for i in range(len(outputs))]
    value = sum([loss(output.flatten()[~nans[i]],phen_data[:,i].flatten()[~nans[i]]) for i,output in enumerate(outputs)])
    return {"MDAD": (value.detach().item(),0.0)}


def single_model_loss(exp_data,phen_data,model_type, which_trait, hidden_layers,hidden_nodes,lr, epochs, batch_size):
    params = {
        'hidden_layers': hidden_layers,
        'hidden_nodes': hidden_nodes
    }
    model = nets.train_one_model(exp_data, phen_data, params,model_type,which_trait, epochs=epochs,lr=lr,batch_size=batch_size)
    loss = torch.nn.MSELoss()
    output = model(exp_data)
    nans = torch.isnan(phen_data[:, which_trait])
    value = loss(output.flatten()[~nans],phen_data[:,i].flatten()[~nans])
    return {f"{model_type}_trait{which_trait}": (value.detach().item(),0.0)}


def optimize_mdad(exp_data,phen_data):
    ax = AxClient()
    ax.create_experiment(
        name="mdad_test_experiment",
        parameters=[
            {
                "name": "hidden_layers",
                "type": "range",
                "bounds": [0, 10],
                "value_type": "int"
            },
            {
                "name": "hidden_nodes",
                "type": "range",
                "bounds": [0, 500],
                "value_type": "int"
            },
            {
                "name": "lr",
                "type": "range",
                "bounds": [1e-10,1e-1],
                "value_type": "float"
            },
            {
                "name": "epochs",
                "type": "range",
                "bounds": [50,500],
                "value_type": "int"
            },
            {
                "name": "batch_size",
                "type": "range",
                "bounds": [10, 300],
                "value_type": "int"
            },
        ],
        objective_name="MDAD",
        minimize=True,
    )

    for _ in range(25):
        parameters, trial_index = ax.get_next_trial()
        ax.complete_trial(
            trial_index=trial_index,
            raw_data=mdad_loss(
                exp_data,
                phen_data,
                parameters['hidden_layers'],
                parameters['hidden_nodes'],
                parameters['lr'],
                parameters['epochs'],
                parameters['batch_size']
            )
        )
    best_parameters, metrics = ax.get_best_parameters()
    ax.get_trials_data_frame().sort_values('trial_index').to_csv("ax_parameter_tuning_MDAD.csv", index=False)
    print(best_parameters)
    print(metrics)
    return 0


def optimize_single_model(exp_data, phen_data, model_type, which_trait):
    ax = AxClient()
    ax.create_experiment(
        name="mdad_test_experiment",
        parameters=[
            {
                "name": "hidden_layers",
                "type": "range",
                "bounds": [0, 10],
                "value_type": "int"
            },
            {
                "name": "hidden_nodes",
                "type": "range",
                "bounds": [0, 500],
                "value_type": "int"
            },
            {
                "name": "lr",
                "type": "range",
                "bounds": [1e-10,1e-1],
                "value_type": "float"
            },
            {
                "name": "epochs",
                "type": "range",
                "bounds": [50, 500],
                "value_type": "int"
            },
            {
                "name": "batch_size",
                "type": "range",
                "bounds": [10, 300],
                "value_type": "int"
            },
        ],
        objective_name=f"{model_type}_trait{which_trait}",
        minimize=True,
    )

    for _ in range(25):
        parameters, trial_index = ax.get_next_trial()
        ax.complete_trial(
            trial_index=trial_index,
            raw_data=single_model_loss(
                exp_data,
                phen_data,
                model_type,
                which_trait,
                parameters['hidden_layers'],
                parameters['hidden_nodes'],
                parameters['lr'],
                parameters['epochs'],
                parameters['batch_size']
            )
        )


    best_parameters, metrics = ax.get_best_parameters()
    ax.get_trials_data_frame().sort_values('trial_index').to_csv(f"ax_parameter_tuning_{model_type}_trait{which_trait}.csv", index=False)
    print(best_parameters)
    print(metrics)


def main():
    phen_samples, trait_labels, phen_data = utils.load_and_process_phenotype("/media/wcasazza/DATA2/wcasazza/ROSMAP/ROSMAP_PHEN.csv")
    exp_samples, exp_data = utils.load_and_process_expression("/media/wcasazza/DATA2/wcasazza/ROSMAP/normalized_dlpfc_v2.tsv")
    exp_ord, phen_ord = utils.match_samples(exp_samples, phen_samples)
    exp_samples, exp_data =  exp_samples[exp_ord], exp_data[exp_ord,:]
    phen_samples, phen_data = phen_samples[phen_ord], phen_data[phen_ord,:]
    exp_data = torch.tensor(exp_data).double()[:300,:]
    phen_data = torch.tensor(phen_data).double()[:300,:]
    optimize_mdad(exp_data, phen_data)
    for i in range(phen_data.shape[1]):
        optimize_single_model(exp_data,phen_data,"mlp",i)
        optimize_single_model(exp_data,phen_data,"nested_linear",i)
    return 0


if __name__ == "__main__":
    main()
