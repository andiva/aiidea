import numpy as np
import matplotlib.pyplot as plt
from cntk import load_model

import physical_models as phys
import models
from SIR_Lie import SIR_Lie_Transform

def plot(title, train, test1, test2, pred, pred1, pred2):
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle(title, fontsize=13)
    fig.add_subplot(3, 1, 1)
    plt.ylabel('Training data')
    fig.add_subplot(3, 1, 2)
    plt.ylabel('Test1 and Prediction1')
    fig.add_subplot(3, 1, 3)
    plt.ylabel('Test2 and Prediction2')

    for ax, data in zip(plt.gcf().get_axes(),
                        [[train, pred],[test1, pred1],[test2,pred2]]):
        ax.grid() 
        # print()
        for sol, alpha in zip(data, [0.5, 1]):
            ax.plot(sol[:, 0], 'b-', alpha=alpha)
            ax.plot(sol[:, 1], 'g-', alpha=alpha)
            ax.plot(sol[:, 2], 'r-', alpha=alpha)


    for ax in plt.gcf().get_axes():
        ax.grid() 

    return


def main():
    # generate data from physical model
    T = 10 # time interval
    dt = 0.1 # integration step
    # train data:
    t, train = phys.simulate_epidemiology(0, np.array([0.99, 0.01, 0]), T, dt=dt)
    # test data 1st:
    _, test1 = phys.simulate_epidemiology(0, np.array([0.4, 0.1, 0]), T, dt=dt)
    # test data 2nd:
    _, test2 = phys.simulate_epidemiology(0, np.array([1, 0.0, 0]), T, dt=dt)


    N = 5 # size of the history window (for LSTM)

    # train lstm model
    model_lstm = models.train_lstm(train, N)
    model_lstm.save('bin_models/epidemiology_lstm')
    # or load pre-built one
    model_lstm = load_model('bin_models/epidemiology_lstm')


    # for training a polynomial neural network (matrix Lie transform) follow link
    # https://github.com/andiva/DeepLieNet/blob/master/demo/SIR_Identification.ipynb
    # load pre-built 3rd order Lie transform:
    model_linear =  SIR_Lie_Transform()


    for model, n, title in zip([model_lstm, model_linear],
                               [N, 1],
                               ["LSTM: non-physical predictions, training data are memorized",
                                "Linear map: physical behaviour generalization"]):
        # use model for prediction with training initial conditions
        pred = models.predict(model, train[:n], step_count=int(T/dt-n), N=n)
        # use model for prediction with initial condition of test1
        pred1 = models.predict(model, test1[:n], step_count=int(T/dt-n), N=n)
        # use lstm for prediction with initial condition of test2
        pred2 = models.predict(model, test2[:n], step_count=int(T/dt-n), N=n)

        plot(title, train, test1, test2, pred, pred1, pred2)
    
    plt.show()
    return 0

if __name__ == '__main__':
    main()