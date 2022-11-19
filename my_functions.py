import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def load_data(file_name):
    """
    Loads the data from a student dataset file_name and converts it to a training set (x_train, y_train).
    The input x_train includes the features ["sex", "age", "Pstatus", "Mjob", "Fjob", "higher", "activities"],
    the output y_train contains the final grade G3.

    Args:
        file_name (string): path to a student dataset

    Returns:
        x_train (ndarray): Shape(m, 7), m - number of training examples (students) Input to the model
        y_train (ndarray): Shape(m,) Output of the model
    """
    # importing the dataset
    data = pd.read_csv(file_name)

    # Editing the raw dataset to get x_train and y_train
    school_name = input("Choose the school (for student-mat.csv - GP or MS): ")

    data = data.loc[data["school"] == school_name]
    data = data[["sex", "age", "Pstatus", "Mjob", "Fjob", "higher", "activities", "G3"]]
    print("Hello")
    # Turning categorical features into numbers
    # Dummy matrices
    non_num = data.select_dtypes(include="object")
    for column in non_num.columns:
        if len(non_num[column].unique()) == 2:
            non_num = non_num.drop([column], axis=1)
        else:
            non_num[column] = non_num[column].apply(lambda x: column[0].lower() + "_" + x)
            dummies = pd.get_dummies(non_num[column])
            data = pd.concat([data, dummies], axis=1)
            data = data.drop([column], axis=1)

    # Binary label encoding
    non_num = data.select_dtypes(include="object")
    encoder = LabelEncoder()
    for column in non_num.columns:
        data[column] = encoder.fit_transform(data[column])

    # Extracting x_train and y_train from the table
    x_train = data.drop(["G3"], axis=1)
    y_train = data["G3"]

    # Normalizing the data
    scaler = StandardScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)

    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()

    return x_train, y_train


def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,n) Input to the model (features of students)
        y (ndarray): Shape (m,) Target (final grade G3 of students)
        w (ndarray): Shape(n,) Parameter of a model
        b (scalar): Parameter of a model

    Returns:
        total_cost (float): the cost of using w,b as the parameters for linear regression to fit the data points x and y
    """
    m = x.shape[0]
    n = x.shape[1]

    total_cost = 0

    sum = 0
    for i in range(m):
        func = np.dot(w, x[i]) + b
        cost = (func - y[i]) ** 2
        sum = sum + cost
    total_cost = sum / (2 * m)
    return total_cost


def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression.

    Args:
        x (ndarray): Shape (m,n) Input to the model (features of students)
        y (ndarray): Shape (m,) Target (final grade G3 of students)
        w (ndarray): Shape(n,) Parameter of a model
        b (scalar): Parameter of a model
    Returns:
        dj_dw (ndarray): Shape (n,) The gradient of the cost function depending on parameter w
        dj_db (scalar): The gradient of the cost function depending on parameter b
    """

    m = x.shape[0]
    n = x.shape[1]

    dj_dw = np.zeros(n)
    dj_db = 0

    dj_dw_sum = np.zeros(n)
    dj_db_sum = 0

    for j in range(n):
        for i in range(m):
            func = np.dot(w, x[i]) + b
            dj_dw_i_j = (func - y[i]) * x[i][j]
            dj_dw_sum[j] = dj_dw_sum[j] + dj_dw_i_j
        dj_dw[j] = dj_dw_sum[j] / m

    for i in range(m):
        func = np.dot(w, x[i]) + b
        dj_db = func - y[i]
        dj_db_sum = dj_db_sum + dj_db
    dj_db = dj_db / m

    return dj_dw, dj_db







def gradient_descent(x, y, w_init, b_init, cost_function, gradient_function, alpha, num_iters):
    """

    Performs gradient descent to compute parameters w, b of the model. Updated the parameters by taking
    num_iters gradient steps with the learning rate alpha.

    Args:
        x (ndarray): Shape (m,n) Input to the model (features of students)
        y (ndarray): Shape (m,) Target (final grade G3 of students)
        w_init (ndarray): Shape (n,) Initial parameter of the model
        b_init (scalar):  Initial parameter of the model
        cost_function: Function to compute cost
        gradient_function: Function to compute the gradient of cost
        alpha (float): Learning rate
        num_iters (int): Number of iterations to run gradient descent

    Returns:
        w (ndarray): Shape (n,) Updated values of parameters of the model after running gradient descent
        b (scalar): Updated value of parameter of the model after running gradient descent
    """

    m = x.shape[0]  # number of training examples
    n = x.shape[1]  # number of features

    J_history = []

    w = copy.deepcopy(w_init)  # Assigning w and b to initial values
    b = copy.deepcopy(b_init)

    for i in range(num_iters):

        # Calculate the gradient
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Update the parameters
        for j in range(n):
            w[j] = w[j] - alpha * dj_dw[j]
        b = b - alpha * dj_db

        # Save the cost J at each step
        if i < 10000:  # Prevent resource exhaustion
            cost = cost_function(x, y, w, b)
            J_history.append(cost)

    return w, b, J_history  # Return J history for graphing


def predict(x, w, b):
    """

    Predicts the final grade of a student by using the trained model with updates parameters w and b.

    Args:
        x (ndarray): Shape (m,n) Input to the model (features of students)
        w (ndarray): Shape(n,) Updated parameter of the trained model
        b (scalar): Updated parameter of the trained model

    Returns:
        y_pred (float): prediction of the final grade for the student
    """

    y_pred = np.dot(w, x) + b
    return y_pred


def learning_curve(num_iters, cost_history):
    """

    Generates a learning curve (cost function J vs iterations). If J decreases at each step, than the model works.

    Args:
        num_iters (int): Number of iterations to run gradient descent
        cost_history (list): Shape (num_iters,) Values of cost function at each itaeration step

    """
    iterations = np.arange(num_iters, dtype=int)  # Creating an array of iteration steps
    plt.scatter(iterations, cost_history)
    plt.title("Learning curve")
    plt.xlabel("# of iteration")
    plt.ylabel("Cost function")
    plt.show()

