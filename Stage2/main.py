import Perceptron
import Adaline
import load_data as data
import visualisation as vis
from Stage2.visualisation import plot_losses


def stage_2_2():
    # ppn: Perceptron = Perceptron.Perceptron(eta=0.1, n_iter=10)
    # ppn.fit(data.X, data.y)
    #
    # vis.plot_errors(ppn.errors_)
    # vis.plot_decision_regions(data.X, data.y, classifier=ppn)

    adl1: Adaline = Adaline.AdalineGD(eta=0.01, n_iter=15)
    adl1.fit(data.X, data.y)
    adl2: Adaline = Adaline.AdalineGD(eta=0.01, n_iter=15)
    adl2.fit(data.X, data.y)

    plot_losses(adl1, adl2)


stage_2_2()