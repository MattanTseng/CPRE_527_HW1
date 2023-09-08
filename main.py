from src import load_cifar10, Net, training_step, evaluate, config_loader
import numpy as np
import time
import torch
import os

from src.analytics import My_Analytics

if __name__ == '__main__':

    # check to see what hardware we can run this on.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Using device: ", device)

    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    config_location = os.path.join(current_dir, "hyperparameters.YAML")



    hyper_params = config_loader(config_location)
    n_epochs = hyper_params["epochs"]
    learning_rate = hyper_params["learning_rate"]
    
    




    print("Hyperparameters: \n", hyper_params)

    mean_train_losses = np.empty(0)
    validation_accuracies = np.empty(0)

    model = Net()

    dir_name = "E_" + str(n_epochs) + "_lr_" + str(learning_rate) + "_BS_" + str(hyper_params["batch_size"]) + "_" + model.__str__() 
    output_location = os.path.join(current_dir, dir_name)

    analytics = My_Analytics(output_location)

    model.to(device)

    print("loading data: ")
    train_loader, test_loader,val_loader, classes = load_cifar10(hyper_params)


    print("Starting training: ")
    start_time = time.time()
    for epoch in range(n_epochs):
        mean_train_losses = np.concatenate((mean_train_losses, training_step(model, train_loader, epoch, device, learning_rate)))
        validation_accuracies = np.concatenate((validation_accuracies, np.array([evaluate(model, val_loader, device)])))
        print("-"*10,"Training finshed","-"*10)

    end_time = time.time()
    print("Done training")
    run_time = end_time - start_time

    test_accuracy = evaluate(model, test_loader, device)

    analytics.create_1D_graphs(mean_train_losses, "image*10^4", "loss", "Training Losses")
    analytics.create_1D_graphs(validation_accuracies, "image*10^4", "accuracy", "Validation Accuracies")
    analytics.export_graphs()
    analytics.create_report(run_time, hyper_params, test_accuracy, device)




    print("Here are the average losses for every epoch: ", mean_train_losses)
    print("Here are the validation_accuracies of every epoch: ", validation_accuracies)
    print("Here are is the accuracy on the test set: ", test_accuracy)
    print("The run took: ", run_time, " seconds to run")
