import numpy as np
import os
from prettytable import PrettyTable

def cls():
    os.system("clear")

def main_menu():
    cls()
    print("-----Perceptron Training Simulation-----")
    print("1. Training")
    print("2. Import dataset")
    print("3. Exit")
    ans = int(input("Your answer >> "))
    return ans

def perceptron_input():
    cls()
    w = []
    while True:
        n_input = int(input("How many input do you want to have?(2/3) >> "))
        if n_input == 2 or n_input == 3:
            break
        else:
            print("Input given must 2 or 3")

    for i in range(n_input):
        w_temp = float(input("input weight of input n = {} >> ".format(i+1)))
        w.append(w_temp)

    b = float(input("Input bias of the perceptron (0...) >> "))
    alph = float(input("Input learning rate of the perceptron (0...) >> "))
    rows = int(input("Input how many data rows do you want (0...) >> "))
    print("Initialize Perceptron...")
    return Perceptron(n_input, w, b, rows, alph)

class Perceptron:
    def __init__(self, n_input, w, b, rows, alph):
        self.layer = {
            "input" : n_input,
            "output" : 1
        }
        self.w = w
        self.b = b
        self.alph = alph
        self.rows = rows
        self.dataset = []
        self.target = []

    def input_data(self):
        cls()
        outer_arr = []
        for i in range(self.rows):
            inner_arr = []
            for j in range(self.layer["input"] + self.layer["output"]):
                if j + 1 == self.layer["input"] + self.layer["output"]:
                    num = float(input("Enter target for row {} >> ".format(i+1)))
                else:
                    num = float(input("Enter value for row {} input {} >> ".format(i+1, j+1)))
                inner_arr.append(num)
            outer_arr.append(inner_arr)

        dataset = np.array(outer_arr)
        self.target = dataset[np.arange(len(dataset)), [self.layer["input"] + self.layer["output"] - 1]]
        self.dataset = np.delete(dataset, (self.layer["input"] + self.layer["output"]) - 1, 1)

    def generate_table(self):
        perceptron_table = PrettyTable()
        arr_temp = []
        for i in range(self.layer["input"]):
            field_names_str = "Input " + str(i)
            arr_temp.append(field_names_str)
        perceptron_table.field_names = arr_temp
        perceptron_table.add_rows(self.dataset)
        perceptron_table.add_column("Target", self.target)
        return perceptron_table

    def print_data(self):
        cls()
        print("------| ADALINE's Data |------")
        perceptron_table = self.generate_table()
        print(perceptron_table)
        print("Weight = {}".format(self.w))
        print("Bias = {}".format(self.b))

    def generate_2d_linear_function(self,w,b):
        w1 = w[0]
        w2 = w[1]
        function = "y = {:.2f}x + {:.2f}".format((-(b/w2) / (b/w1)), (-b/w2))
        return function

    def aftertraining_table(self, epoch_arr, w_arr, b_arr, error_arr, fx_arr):
        print("-----| Training Result |-----")
        table = PrettyTable()
        table.add_column("Epoch", epoch_arr)
        table.add_column("Weights", np.around(w_arr, decimals = 2))
        table.add_column("Biases", np.around(b_arr, decimals = 2))
        table.add_column("Error", np.around(error_arr, decimals = 2))
        table.add_column("f(x)", fx_arr)
        return table

    def train_perceptron(self):
        # assign epoch's input, initial epoch, and variables
        epoch = int(input("Input number of epoch >> "))
        num_epoch = 0
        w = self.w
        b = self.b
        alph = self.alph

        # array variable for table creation
        error_arr = []
        epoch_arr = []
        w_arr = []
        b_arr = []
        fx_arr = []
        mean_error = []

        # looping from given epoch
        for i in range(epoch):
            num_epoch += 1
            print("Weight = {}".format(w))
            print("Bias = {}".format(b))
            print("Epoch = {}".format(num_epoch))
            for j in range(self.rows):
                p = np.transpose(self.dataset[j])
                t = self.target[j]
                a = np.dot(w, p) + b
                E = np.square(t - a)
                e = t - a
                # print("Weight : {}".format(w))
                # print("Bias: {}".format(b))
                # print("Error : {}".format(E))
                # print("Target: {}".format(t))
                w = w + (2 * alph * e * p)
                b = b + (2 * alph * e)
                function = self.generate_2d_linear_function(w,b)

                # assign variables to array tables
                mean_error.append(E)
                epoch_arr.append(num_epoch)
                w_arr.append(w)
                b_arr.append(b)
                error_arr.append(E)
                fx_arr.append(function)
            print("Mean Error: {}".format(np.mean(mean_error)))
            mean_error = []
        print("Adaline Trained!")
        table = self.aftertraining_table(epoch_arr, w_arr, b_arr, error_arr, fx_arr)
        print(table)

# Main Program
ans = main_menu()
if ans == 1:
    perceptron = perceptron_input()
    print(perceptron.w)
    perceptron.input_data()
    perceptron.print_data()
    input("Press enter to train perceptron... >> ")
    perceptron.train_perceptron()

