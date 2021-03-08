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
    rows = int(input("Input how many data rows do you want (0...) >> "))
    print("Initialize Perceptron...")
    return Perceptron(n_input, w, b, rows)

class Perceptron:
    def __init__(self, n_input, w, b, rows):
        self.layer = {
            "input" : n_input,
            "output" : 1
        }
        self.w = w
        self.b = b
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
                    num = int(input("Enter target for row {} >> ".format(i+1)))
                else:
                    num = int(input("Enter value for row {} input {} >> ".format(i+1, j+1)))
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
        print("------| Perceptron's Data |------")
        perceptron_table = self.generate_table()
        print(perceptron_table)
        print("Weight = {}".format(self.w))
        print("Bias = {}".format(self.b))

    def aftertraining_table(self, expectation, result):
        print("-----| Training Result |-----")
        table = PrettyTable()
        table.add_column = ("Expectation", expectation)
        table.add_column = ("Result", result)
        return table
        

    def hardlim(self, n):
        if n >= 0:
            return 1
        else:
            return 0

    def train_perceptron(self):
        epoch = 1
        w = self.w
        b = self.b
        is_classified = []
        result = []
        while True:
            print("Weight = {}".format(w))
            print("Bias = {}".format(b))
            print("Epoch = {}".format(epoch))
            for i in range(self.rows):
                p = np.transpose(self.dataset[i])
                t = self.target[i]
                a = self.hardlim(np.matmul(w, p) + b)
                e = t - a
                if e == 1:
                    w = w + p
                    is_classified.append(0)
                elif e == -1:
                    w = w - p
                    is_classified.append(0)
                else:
                    is_classified.append(1)
                b = b + e
                result.append(a)
            if (0 not in is_classified) == True or epoch == 1000:
                break
            else:
                epoch += 1
                is_classified = []
                result = []
        print("Perceptron Trained!")
        print("Final Weight = {}".format(w))
        print("Final Bias = {}".format(b))
        print("Final Epoch = {}".format(epoch))
        table = aftertraining_table(result, self.target)

# Main Program
ans = main_menu()
if ans == 1:
    perceptron = perceptron_input()
    print(perceptron.w)
    perceptron.input_data()
    perceptron.print_data()
    input("Press enter to train perceptron... >> ")
    perceptron.train_perceptron()

