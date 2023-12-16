import numpy as np
import sys
import time

def loss_function(y_goal, x_curr, theta):
    return (y_goal - (x_curr + (-1 + 2/(1 + np.e ** (-20 * (theta[0] - 0.5)))))) ** 2

def calculate_gradient(y_goal, x_curr, p):
    f = x_curr + (-1 + 2.0/(1 + np.e**(-20*(p - 0.5))))
    f_d = (40 * np.e ** (-20 * (p - 0.5))) / (1 + np.e ** (-20 * (p - 0.5))**2)

    return 2 * f * f_d - 2 * y_goal * f_d

def progress(x, x_curr, theta, step):
    test = [str(b) for b in x]
    test = "".join(test)
    sys.stdout.write(test + " x_curr: " + str(x_curr) + " L_Prob: " + str(1 - theta[0])[:5] + " R_Prob:" + str(theta[0])[:5] + " Loss: " + str(step))
    sys.stdout.flush()
    print('', end='\r')

def run_simulation(grid: np.ndarray, x_curr: int, y_goal: int, lr: float, theta: np.ndarray = np.array([0.5])) -> None:
    steps = 0
    grid[y_goal] = 'G'
    grid[x_curr] = 'X'
    
    while x_curr != y_goal:
        next_step = np.random.binomial(1, theta[-1])

        grid[x_curr] = '#'
        
        theta[0] += -lr * calculate_gradient(y_goal, x_curr, theta)
        theta[0] = max(min(theta[0], 1), 0)

        x_curr = max(x_curr + (-1) ** (1 - next_step), 0)

        loss = loss_function(y_goal, x_curr, theta)



        grid[x_curr] = 'X'
        progress(grid, x_curr, theta, loss)

        time.sleep(0.1)

        steps += 1

    print('', end='\n')
    sys.stdout.flush()

if __name__ == "__main__":
    grids = [np.array(['#' for _ in range(50)]), np.array(['#' for _ in range(70)]), np.array(['#' for _ in range(100)])]
    y_goals = [50 - 1, 25, 2]
    x_currs = [1, 36, 50 - 1]
    lrs = [1e-3, 1e-3, 1e-3]
    thetas = [np.array([0.5]),np.array([0.5]), np.array([0.5])]

    for i in range(len(y_goals)):
        sys.stdout.flush()
        print("", end="\n")
        run_simulation(grids[i], x_currs[i], y_goals[i], lrs[i], thetas[i])
