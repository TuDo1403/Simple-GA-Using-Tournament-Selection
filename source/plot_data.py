import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import logistic_regression as lr
import linear_regression as lnr
import new_POPOP as popop
from collections import Counter

def load_data(file_name):
    data = []
    with open(file_name) as file:
        for line in file:
            data.append(line.replace("\n", "").split(","))
    data = np.array(data, dtype=np.float)
    return data

def get_training_examples(data, num_features=1):
    dt_size = np.shape(data)    # get data's shape

    # convert 1D array to 2D array
    X = np.reshape(data[:, :num_features], (dt_size[0], num_features))
    y = np.reshape(data[:, num_features:], (dt_size[0], dt_size[1]-num_features))
    return X, y




def scatter_plot(x, y, hold=False, x_label = None, y_label = None):
    positve = np.where(y == 1)[0]
    negative = np.where(y == 0)[0]

    plt.plot(x[positve, 0], x[positve, 1], "bo", label="Success")
    plt.plot(x[negative, 0], x[negative, 1], "rx", label="Fail")

    plt.xlabel(xlabel=x_label)
    plt.ylabel(ylabel=y_label)

    plt.xlim([0, 300])
    plt.ylim([0, 100])

    plt.xticks([2 ** i for i in range(4, 9)])
    plt.yticks([2 ** i for i in range(2, 7)])

    plt.legend()
    if not hold:
        plt.show()

def scatter_plot_3D(X, y, title=""):
    fig = plt.figure()
    ax = Axes3D(fig)

    positive = np.where(y == 1)[0]
    negative = np.where(y == 0)[0]


    ax.scatter(X[positive, 0], X[positive, 1], X[positive, 3], marker="^", label="Optimized")
    ax.scatter(X[negative, 0], X[negative, 1], X[negative, 3], marker="o", label="Unoptimized")

    ax.set_xlabel("Number of individuals")
    ax.set_ylabel("Number of parameters")
    ax.set_zlabel("Seeds type")

    ax.set_zlim([18521578, 18521587])
    ax.set_xlim([0, 300])
    ax.set_ylim([0, 100])

    ax.set_xticks([2 ** i for i in range(4, 9)])
    ax.set_yticks([2 ** i for i in range(2, 7)])

    ax.ticklabel_format(useOffset=False)
    ax.tick_params('z', labelsize=7)    # set z axis label size
    fig.suptitle(title)
    ax.legend()
    plt.show()

def plot_popsize_effect_on_eval_func_calls(X, y):
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    #fig.suptitle("Parameters/Individuals 's Effect On Number of OneMax Calls")

    ux = np.where(X[:, 2] == popop.Crossover.UX.value)[0]
    one_point = np.where(X[:, 2] == popop.Crossover.ONEPOINT.value)[0]

    X_ux, y_ux = get_smallest_success_individuals_case(X[ux], y[ux])
    X_1x, y_1x = get_smallest_success_individuals_case(X[one_point], y[one_point])

    # ## Individuals subplot
    # figure = plt.figure()
    # figure.suptitle("Individuals's Effect On Number of OneMax Calls")

    # plt.plot(X_ux[:, 0], y_ux[:, 0], "co", label="UX", alpha=0.5)
    # plt.plot(X_1x[:, 0], y_1x[:, 0], "bo", label="1X", alpha=0.5)
    # plt.plot(X_ux[:, 0], y_ux[:, 0], "c-", alpha=1)
    # plt.plot(X_1x[:, 0], y_1x[:, 0], "b-", alpha=1)

    # plt.xlabel("Number of individuals")
    # plt.ylabel("Number of function calls")

    # plt.xlim([0, 300])
    # plt.ylim([30, 2100])

    # plt.xticks([2 ** i for i in range(2, 9)])
    # plt.yticks([2 ** i for i in range(7, 12)])

    # plt.tick_params("y", labelsize=7)

    # plt.legend()

    # ##

    ## Parameters subplot
    figure = plt.figure()
    figure.suptitle("Problem Size's Effect On Number of OneMax Calls")

    plt.plot(X_ux[:, 1], y_ux[:, 0], "co", label="UX", alpha=0.5)
    plt.plot(X_1x[:, 1], y_1x[:, 0], "bo", label="1X", alpha=0.5)
    plt.plot(X_ux[:, 1], y_ux[:, 0], "c-", alpha=1)
    plt.plot(X_1x[:, 1], y_1x[:, 0], "b-", alpha=1)

    plt.xlabel("Problem size (d)")
    plt.ylabel("Number of function calls (n)")

    plt.xlim([0, 70])
    plt.ylim([30, 2100])

    plt.xticks([2 ** i for i in range(2, 7)])
    plt.yticks([2 ** i for i in range(7, 12)])

    plt.tick_params("y", labelsize=7)

    plt.legend()
    ##

    plt.show()

def plot_decision_boundary(theta, X, y, title=""):
    figure = plt.figure()
    scatter_plot(X[:, 1:3], y, hold=True, x_label="Number of individuals", y_label="Number of parameters")

    plot_x = np.array([np.amin(X, axis=0)[1]-2, np.amax(X, axis=0)[1]+2])
    plot_y = (-1 / theta[2]) * (theta[1]*plot_x + theta[0]) # 1 + x1 * theta1 + x2 * theta2 = 0 (x2 == y) 
    plt.plot(plot_x, plot_y, "--", c="red", label="Boundary")

    plt.legend()
    figure.suptitle(title)
    plt.show()


def get_training_per_10seeds_examples(X, y, num_seeds):
    num_examples = len(X)
    new_X = []
    new_y = []
    for i in range(0, num_examples, num_seeds):
        new_X.append([X[i, 0], X[i, 1], X[i, 2]])
        result = 1 if sum(y[i : i+num_seeds]) == 10 else 0
        new_y.append(result)

    new_X = np.reshape(new_X, (len(new_X), 3))
    new_y = np.reshape(new_y, (len(new_y), 1))
    return new_X, new_y

def get_summary_data_for_func_calls(X, y, num_seeds):
    num_examples = len(X)
    new_X = []
    new_y = []
    for i in range(0, num_examples, num_seeds):
        new_X.append([X[i, 0], X[i, 1], X[i, 2]])
        mean_calls = np.mean(y[i : i+num_seeds, 0], axis=0)
        result = 1 if sum(y[i : i+num_seeds, 1]) == 10 else 0
        new_y.append([mean_calls, result])

    new_X = np.reshape(new_X, (len(new_X), 3))
    new_y = np.reshape(new_y, (len(new_y), 2))
    return new_X, new_y


data = load_data("training_data.txt")
X, y = get_training_examples(data, num_features=4)

## Plot data with (x=num_individuals, y=num_params, z=seeds)
one_point = np.where(X[:, 2] == popop.Crossover.ONEPOINT.value)[0]
ux  = np.where(X[:, 2] == popop.Crossover.UX.value)[0]
X = X[ux]
y = y[ux]
#scatter_plot_3D(X, y, "3D GA OnePoint Crossover Mode Plot")

## Plot data with (x=num_individuals, y=num_params)
X1, y1 = get_training_per_10seeds_examples(X, y, 10)
#scatter_plot(X1, y1, x_label="num_individuals", y_label="num_params")

## Apply logistic regression to find decision boundary
X2 = np.reshape(X1[:, :2], (len(X1), 2))
X2 = np.hstack((np.ones((len(X2), 1)), X2))
theta = np.zeros((3, 1))
#cost = lr.cost_function(theta, X2, y1)
#print(cost)
theta = lr.fminunc(theta, X2, y1)
plot_decision_boundary(theta, X2, y1, "GA OnePoint Crossover Mode Plot")


# # Plot data with (x=num_individuals, y=num_eval_func_calls)
# data = load_data("training_data1.txt")
# X3, y3 = get_training_examples(data, num_features=3)
# X3, y3 = get_summary_data_for_func_calls(X3, y3, 10)

# success = np.where(y3[:, 1] == 1)[0]
# X3 = X3[success]
# y3 = y3[success]

# def get_smallest_success_individuals_case(X, y):
#     new_X = []
#     new_y = []
#     params = np.unique(X[:, 1])
#     for p in params:
#         p_idx = np.where(X[:, 1] == p)[0]
#         # y_temp = y[p_idx]
#         # X_temp = X[p_idx]
#         # min_idx = np.where(X[p_idx, 0] == np.min(X[p_idx, 0], axis=0))[0]
#         new_X.append(X[p_idx][0, :])
#         new_y.append(y[p_idx][0, :])

#     new_X = np.reshape(new_X, (len(new_X), 3))
#     new_y = np.reshape(new_y, (len(new_y), 2))
#     return new_X, new_y



# plot_popsize_effect_on_eval_func_calls(X3, y3)







