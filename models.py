import numpy as np
import time
import cntk as C
import cntk.tests.test_utils


def create_lstm_model(x, N, outputs=1):
    """Create the model for time series prediction"""
    with C.layers.default_options(initial_state = 0.1):
        m = C.layers.Recurrence(C.layers.LSTM(N))(x)
        m = C.sequence.last(m)
        # m = C.layers.Dropout(0.2, seed=1)(m)
        m = C.layers.Dense(outputs)(m)
        return m


def next_batch(x, y, batch_size):
    """get the next batch to process"""

    def as_batch(data, start, count):
        part = []
        for i in range(start, start + count):
            part.append(data[i])
        return np.array(part)

    for i in range(0, len(x)-batch_size, batch_size):
        yield as_batch(x, i, batch_size), as_batch(y, i, batch_size)


def rolling_window_1D(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_window(states, N):
    sample_count = states.shape[0] - N
    dim = states.shape[1]

    X = np.empty((sample_count, N, dim))
    Y = states[N:,:].copy()

    for i in range(dim):
        X[:,:,i] = rolling_window_1D(states[:, i], N)[:sample_count]
    
    return X, Y


def train_lstm(states, N=5, epochs = 500, batch_size = 10):
    X,Y = rolling_window(states, N)
    return train(create_lstm_model, X, Y, N, epochs, batch_size)


def train(create_model, X, Y, N=5, epochs = 500, batch_size = 10):
    dim = Y.shape[1]
    
    # input sequences
    x = C.sequence.input_variable(dim)
    # create the model
    z = create_model(x, N=dim, outputs=dim)

    # expected output (label), also the dynamic axes of the model output
    # is specified as the model of the label input
    l = C.input_variable(dim, dynamic_axes=z.dynamic_axes, name="y")

    # the learning rate
    learning_rate = 0.02
    lr_schedule = C.learning_parameter_schedule(learning_rate)

    # loss function
    loss = C.squared_error(z, l)
    # use squared error to determine error for now
    error = C.squared_error(z, l)

    # use fsadagrad optimizer
    momentum_schedule = C.momentum_schedule(0.9, minibatch_size=batch_size)
    learner = C.fsadagrad(z.parameters,
                          lr = lr_schedule,
                          momentum = momentum_schedule,
                          unit_gain = True)
    trainer = C.Trainer(z, (loss, error), [learner])

    # train
    loss_summary = []
    start = time.time()
    for epoch in range(0, epochs):
        for x1, y1 in next_batch(X, Y, batch_size):
            trainer.train_minibatch({x: x1, l: y1})
        if epoch % (epochs / 10) == 0:
            training_loss = trainer.previous_minibatch_loss_average
            loss_summary.append(training_loss)
            print("epoch: {}, loss: {:.5f}".format(epoch, training_loss))

    print("training took {0:.1f} sec".format(time.time() - start))

    return z


def predict(model, X0, step_count, N=1):
    X = np.empty((N+step_count, X0.shape[1]))
    X[:N] = X0

    for i in range(step_count):
        tmp = model.eval(X[i:i+N])[0].ravel()
        X[i+N:i+1+N] = tmp
    
    return X