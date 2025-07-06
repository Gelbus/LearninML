import Perceptron
import load_data as data
import visualisation as vis

def stage_2_2():
    ppn: Perceptron = Perceptron.Perceptron(eta=0.1, n_iter=10)
    ppn.fit(data.X, data.y)

    vis.plot_errors(ppn.errors_)
    vis.plot_decision_regions(data.X, data.y, classifier=ppn)

stage_2_2()