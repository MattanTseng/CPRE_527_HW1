from src import load_cifar10, Net, training_step, evaluate
import numpy as np
import time

if __name__ == '__main__':
    accuracies = np.ndarray()
    n_epochs = 10
    model = Net()
    trainloader,testloader,classes = load_cifar10()
    print("Starting training: ")
    start_time = time.time()
    for epoch in range(n_epochs):
        training_step(model, trainloader, epoch)
        np.concatenate(accuracies, evaluate(model, testloader))
        print("-"*10,"Training finshed","-"*10)

    end_time = time.time()
    print("Done training")
    run_time = end_time - start_time

    print("Here are the accuracies of every epoch: ", accuracies)
    print("The run took: ", run_time, " seconds to run")
