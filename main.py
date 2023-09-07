from src import load_cifar10, Net, training_step, evaluate, config_loader
import numpy as np
import time

if __name__ == '__main__':
    config_location = "/Users/mattantseng/Documents/Python/CPRE_527_HW1/hyperparameters.YAML"

    hyper_params = config_loader(config_location)




    train_accuracies = np.ndarray()
    validation_accuracies = np.ndarray()
    n_epochs = hyper_params["epochs"]
    model = Net()
    trainloader,testloader,val_loader, classes = load_cifar10()
    print("Starting training: ")
    start_time = time.time()
    for epoch in range(n_epochs):
        training_step(model, trainloader, epoch)
        np.concatenate(train_accuracies, evaluate(model, val_loader))
        print("-"*10,"Training finshed","-"*10)

    end_time = time.time()
    print("Done training")
    run_time = end_time - start_time

    print("Here are the train_accuracies of every epoch: ", train_accuracies)
    print("The run took: ", run_time, " seconds to run")
