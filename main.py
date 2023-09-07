from src import load_cifar10, Net, training_step, evaluate, config_loader
import numpy as np
import time
import torch

if __name__ == '__main__':

    # check to see what hardware we can run this on.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"


    config_location = "/Users/mattantseng/Documents/Python/CPRE_527_HW1/hyperparameters.YAML"

    hyper_params = config_loader(config_location)
    print("Hyperparameters: \n", hyper_params)

    test_array = np.empty(0)
    print("test_array: ", test_array, " Type: ", type(test_array))

    blah = [1, 2, 3, 4]
    print("Can I concate?", np.concatenate((test_array, np.array(blah))))



    mean_train_losses = np.empty(0)
    validation_accuracies = np.empty(0)
    n_epochs = hyper_params["epochs"]
    model = Net()

    model.to(device)

    print("loading data: ")
    train_loader, test_loader,val_loader, classes = load_cifar10(hyper_params)


    print("Starting training: ")
    start_time = time.time()
    for epoch in range(n_epochs):
        training_step(model, train_loader, epoch, device)
        np.concatenate((validation_accuracies, np.array([evaluate(model, val_loader, device)])))
        print("-"*10,"Training finshed","-"*10)

    end_time = time.time()
    print("Done training")

    test_accuracy = evaluate(model, test_loader, device)



    run_time = end_time - start_time

    print("Here are the average losses for every epoch: ", mean_train_losses)
    print("Here are the validation_accuracies of every epoch: ", validation_accuracies)
    print("Here are is the accuracy on the test set: ", test_accuracy)
    print("The run took: ", run_time, " seconds to run")
