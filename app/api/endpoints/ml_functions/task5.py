# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
 
def mean_squared_error(y_true, y_predicted):
     
    # Calculating the loss or cost
    cost = np.sum((y_true-y_predicted)**2) / len(y_true)
    return cost
 
# Gradient Descent Function
# Here iterations, learning_rate, stopping_threshold
# are hyperparameters that can be tuned
def gradient_descent(x, y, iterations = 1000, learning_rate = 0.0001,
                     stopping_threshold = 1e-6, identification = "test"):
     
    # Initializing weight, bias, learning rate and iterations
    current_weight = 0.1
    current_bias = 0.01
    iterations = iterations
    learning_rate = learning_rate
    n = float(len(x))
     
    costs = []
    weights = []
    previous_cost = None
     
    # Estimation of optimal parameters
    for i in range(iterations):
         
        # Making predictions
        y_predicted = (current_weight * x) + current_bias
         
        # Calculationg the current cost
        current_cost = mean_squared_error(y, y_predicted)
 
        # If the change in cost is less than or equal to
        # stopping_threshold we stop the gradient descent
        if previous_cost and abs(previous_cost-current_cost)<=stopping_threshold:
            break
         
        previous_cost = current_cost
 
        costs.append(current_cost)
        weights.append(current_weight)
         
        # Calculating the gradients
        weight_derivative = -(2/n) * sum(x * (y-y_predicted))
        bias_derivative = -(2/n) * sum(y-y_predicted)
         
        # Updating weights and bias
        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)
                 
        # Printing the parameters for each 1000th iteration
        print(f"Iteration {i+1}: Cost {current_cost}, Weight \
        {current_weight}, Bias {current_bias}")
     
     
    # Visualizing the weights and cost at for all iterations
    plt.figure(figsize = (8,6))
    plt.plot(weights, costs)
    plt.scatter(weights, costs, marker='o', color='red')
    plt.title("Cost vs Weights")
    plt.ylabel("Cost")
    plt.xlabel("Weight")
    plt.savefig(f"images/{identification}_task_5_2.png")
    plt.clf()
     
    return current_weight, current_bias
 
 
def task_5(parameters):
    if "iterations" in parameters and "learning_rate" in parameters:
        iterations = parameters["iterations"]
        learning_rate = parameters["learning_rate"]
        return __execute_task_5__(iterations, learning_rate, parameters["identifier"] or "test")
    else:
        return (False, 0, "Invalid parameters", None)


def __execute_task_5__(iterations = 25, learning_rate = 0.0001, identifier="test"):
     
    if iterations < 0 or learning_rate < 0:
        return (False, 0, "Invalid parameters", None)
    if iterations > 100 or learning_rate > 0.0001:
        return (False, 0, "Invalid parameters", None)

    X = np.array([32.50234527, 53.42680403, 61.53035803, 47.47563963, 59.81320787,
           55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444,
           45.41973014, 54.35163488, 44.1640495 , 58.16847072, 56.72720806,
           48.95588857, 44.68719623, 60.29732685, 45.61864377, 38.81681754])
    Y = np.array([31.70700585, 68.77759598, 62.5623823 , 71.54663223, 87.23092513,
           78.21151827, 79.64197305, 59.17148932, 75.3312423 , 71.30087989,
           55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216,
           60.72360244, 82.89250373, 97.37989686, 48.84715332, 56.87721319])
 
    # Estimating weight and bias using gradient descent
    estimated_weight, eatimated_bias = gradient_descent(X, Y, iterations=iterations, learning_rate=learning_rate, identification=identifier)
 
    # Making predictions using estimated parameters
    Y_pred = estimated_weight*X + eatimated_bias
 
    # Plotting the regression line
    plt.figure(figsize = (8,6))
    plt.scatter(X, Y, marker='o', color='red')
    plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='blue',markerfacecolor='red',
             markersize=10,linestyle='dashed')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(f"images/{identifier}_task_5.png")
    plt.clf()

    if min(Y_pred) >= 45.204 and max(Y_pred) >= 53.980:
        ## calculate slope
        slope = (max(Y_pred) - min(Y_pred)) / (max(X) - min(X))
        return (True, slope, "Task 5 completed", ["images/{}_task_5.png".format(identifier), "images/{}_task_5_2.png".format(identifier)])
    else:
        slope = (max(Y_pred) - min(Y_pred)) / (max(X) - min(X))
        return (False, slope, "Task 5 failed, try to modify the learning rate to steep the decent faster", None)
    # return (True, 1, "Success", {"weight": estimated_weight, "bias": eatimated_bias}, ["images/{}_task_5.png".format(identifier), "images/{}_task_5_2.png".format(identifier)])

def task_detail_5():
    return {
        "task_id": 5,
        "task_name": "Performing Gradient Descent iterations on given data, to observe the change in weights and bias",
        "parameters": {
            "iterations": {
                "type": "int",
                "description": "The number of iterations to perform gradient descent",
                "default": 25,
                "min": 1,
                "max": 100
            },
            "learning_rate": {
                "type": "float",
                "description": "The learning rate for gradient descent",
                "default": 0.0001,
                "min": 0.00001,
                "max": 0.0001,
            }

        }
    }