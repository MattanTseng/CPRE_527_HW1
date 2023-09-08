from matplotlib import pyplot as plt
import os
import numpy as np
import yaml

class My_Analytics:
    def __init__(self, output_location: str):
        self.output_location = output_location
        os.mkdir(output_location)
        self.my_graphs = []

    def create_1D_graphs(self, losses: np.ndarray, x_name: str, y_name: str, title: str, show: bool = False):
        plt.plot(losses)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title(title)

        if show:
            plt.show()

        self.my_graphs.append(plt)


    def export_graphs(self):
        # Save all the graphs in the list
        print("saving ", len(self.my_graphs), "graphs to: ", self.output_location)
        for i, graph in enumerate(self.my_graphs):
            filename = f'graph_{i}.png'  # Generate a unique filename for each graph
            graph.savefig(os.path.join(self.output_location, filename))

    def create_report(self, time: float, config: dict, test_accuracy: float, device: str):
        output_report_location = os.path.join(self.output_location, "report")

        data = {'epochs': config["epochs"],
                'batch_size': config["batch_size"],
                'learning_rate': config["learning_rate"],
                'test_accuracy': test_accuracy,
                'training_time': time, 
                'device': device}

        with open(output_report_location, 'w') as file:
            yaml.dump(data, file)
    



