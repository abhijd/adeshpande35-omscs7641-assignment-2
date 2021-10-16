import time
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.compose import make_column_transformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
    validation_curve
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.model_selection import LearningCurve, ValidationCurve
import dataframe_image as dfi
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import mlrose_hiive
#import mlrose

m_depth = list(range(1, 40))
m_depth.append(None)

n_estimator = list(range(100, 2100, 100))
n_estimator.append(50)

cv = 10

oh = preprocessing.OneHotEncoder(sparse=False)

normalizer = preprocessing.MinMaxScaler()

scaler = preprocessing.StandardScaler()

idTransformer = preprocessing.FunctionTransformer(None)

# assumed ordinals
education = ["unknown", "primary", "secondary", "tertiary"]
credit_default = ["unknown", "yes", "no"]
bm_answer_ = ["no","yes"]
# job = ['unknown','unemployed',]

education_encoder = preprocessing.OrdinalEncoder(categories=[education])
credit_default_encoder = preprocessing.OrdinalEncoder(categories=[credit_default])
bm_answer_encoder = preprocessing.OrdinalEncoder(categories=[bm_answer_])

liv_train, liv_test, liv_ans_train, liv_ans_test = None, None, None, None

bm_train, bm_test, bm_ans_train, bm_ans_test = None, None, None, None

np.random.seed(65)

def run():
    #backprop using gradient descent
    create_intake()
    loop = [
      #  [liv_train, liv_test, liv_ans_train, liv_ans_test, "liver"],
        [bm_train, bm_test, bm_ans_train, bm_ans_test, "bank marketing"],
    ]
    for x in loop:
        X_train, X_test, y_train, y_test, name = x[0], x[1], x[2], x[3], x[4]

        print(name)
        print()
        classifier = mlrose_hiive.NeuralNetwork(hidden_nodes=[6], activation='relu',
                               algorithm = 'gradient_descent',early_stopping = True,
                               max_attempts = 200, max_iters = 10000,
                               bias = True, learning_rate = .001,
                               restarts=0, curve = True, random_state=65)
        stime = time.time()
        classifier.fit(X_train, y_train)
        #make_save_learning_curve_chart( "gradient descent",classifier, "neural_network", "accuracy", np.linspace(0.1, 1.0, 10), None, 8, name, X_train, y_train)
        #make_save_learning_curve_chart( "gradient descent",classifier, "neural_network", "f1", np.linspace(0.1, 1.0, 10), None, 8, name, X_train, y_train)
        etime = time.time()
        btime = etime-stime
        y_train_pred = classifier.predict(X_train)
        train_acc_score = accuracy_score(y_train, y_train_pred)
        train_f1_score = f1_score(y_train, y_train_pred)
        train_loss_score = classifier.loss
        y_pred = classifier.predict(X_test)
        pred_acc_score = accuracy_score(y_test, y_pred)
        pred_f1_score = f1_score(y_test, y_pred)
        pred_loss_score = classifier.loss
        fitness_curve = classifier.fitness_curve
        write_to_csv_2(name, "gradient descent", btime, pred_acc_score, pred_f1_score, train_acc_score, train_f1_score, train_loss_score)

        plt.plot(fitness_curve)
        plt.xlabel("Iters")
        plt.ylabel("Loss")
        plt.savefig("files/"+ name +"_neural_network_gradient_descent_loss.png")
        plt.clf()

        classifier = mlrose_hiive.NeuralNetwork(hidden_nodes=[6], activation='relu',
                               algorithm = 'simulated_annealing',early_stopping = True,
                               max_attempts = 200, max_iters = 10000,
                               bias = True, learning_rate = .001,
                               restarts=0, curve = True, random_state=65, schedule=mlrose_hiive.GeomDecay(init_temp=10000000000, decay=0.55, min_temp=0.0001))
        stime = time.time()
        classifier.fit(X_train, y_train)
        #make_save_learning_curve_chart( "simulated_annealing",classifier, "neural_network", "accuracy", np.linspace(0.1, 1.0, 10), None, 8, name, X_train, y_train)
        #make_save_learning_curve_chart( "simulated_annealingt",classifier, "neural_network", "f1", np.linspace(0.1, 1.0, 10), None, 8, name, X_train, y_train)
        etime = time.time()
        btime = etime-stime
        y_train_pred = classifier.predict(X_train)
        train_acc_score = accuracy_score(y_train, y_train_pred)
        train_f1_score = f1_score(y_train, y_train_pred)
        train_loss_score = classifier.loss
        y_pred = classifier.predict(X_test)
        pred_acc_score = accuracy_score(y_test, y_pred)
        pred_f1_score = f1_score(y_test, y_pred)
        pred_loss_score = classifier.loss
        fitness_curve = classifier.fitness_curve
        write_to_csv_2(name, "simulated_annealing", btime, pred_acc_score, pred_f1_score, train_acc_score, train_f1_score, train_loss_score)

        plt.plot(fitness_curve)
        plt.xlabel("Iters")
        plt.ylabel("Loss")
        plt.savefig("files/"+ name +"_neural_network_simulated_annealing_loss.png")
        plt.clf()

        classifier = mlrose_hiive.NeuralNetwork(hidden_nodes=[6], activation='relu',
                               algorithm = 'random_hill_climb',early_stopping = True,
                               max_attempts = 200, max_iters = 10000,
                               bias = True, learning_rate = .001,
                               restarts=10, curve = True, random_state=65)
        stime = time.time()
        classifier.fit(X_train, y_train)
        #make_save_learning_curve_chart( "random_hc",classifier, "neural_network", "accuracy", np.linspace(0.1, 1.0, 10), None, 8, name, X_train, y_train)
        #make_save_learning_curve_chart( "random_hc",classifier, "neural_network", "f1", np.linspace(0.1, 1.0, 10), None, 8, name, X_train, y_train)
        etime = time.time()
        btime = etime-stime
        y_train_pred = classifier.predict(X_train)
        train_acc_score = accuracy_score(y_train, y_train_pred)
        train_f1_score = f1_score(y_train, y_train_pred)
        train_loss_score = classifier.loss
        y_pred = classifier.predict(X_test)
        pred_acc_score = accuracy_score(y_test, y_pred)
        pred_f1_score = f1_score(y_test, y_pred)
        pred_loss_score = classifier.loss
        fitness_curve = classifier.fitness_curve
        write_to_csv_2(name, "random_hc", btime, pred_acc_score, pred_f1_score, train_acc_score, train_f1_score, train_loss_score)

        plt.plot(fitness_curve)
        plt.xlabel("Iters")
        plt.ylabel("Loss")
        plt.savefig("files/"+ name +"_neural_network_random_hc_loss.png")
        plt.clf()

        classifier = mlrose_hiive.NeuralNetwork(hidden_nodes=[6], activation='relu',
                               algorithm = 'genetic_alg',early_stopping = True,
                               max_attempts = 200, max_iters = 10000,
                               bias = True, learning_rate = .001,
                               restarts=0, curve = True, random_state=65, pop_size=400, mutation_prob=0.05)
        stime = time.time()
        classifier.fit(X_train, y_train)
        #make_save_learning_curve_chart( "genetic_algo",classifier, "neural_network", "accuracy", np.linspace(0.1, 1.0, 10), None, 8, name, X_train, y_train)
        #make_save_learning_curve_chart( "genetic_algo",classifier, "neural_network", "f1", np.linspace(0.1, 1.0, 10), None, 8, name, X_train, y_train)
        etime = time.time()
        btime = etime-stime
        y_train_pred = classifier.predict(X_train)
        train_acc_score = accuracy_score(y_train, y_train_pred)
        train_f1_score = f1_score(y_train, y_train_pred)
        train_loss_score = classifier.loss
        y_pred = classifier.predict(X_test)
        pred_acc_score = accuracy_score(y_test, y_pred)
        pred_f1_score = f1_score(y_test, y_pred)
        pred_loss_score = classifier.loss
        fitness_curve = classifier.fitness_curve
        write_to_csv_2(name, "genetic_algo", btime, pred_acc_score, pred_f1_score, train_acc_score, train_f1_score, train_loss_score)

        plt.plot(fitness_curve)
        plt.xlabel("Iters")
        plt.ylabel("Loss")
        plt.savefig("files/"+ name +"_neural_network_genetic_algo_loss.png")
        plt.clf()


def create_intake():
    global liv_train, liv_test, liv_ans_train, liv_ans_test, bm_train, bm_test, bm_ans_train, bm_ans_test

    liv_dataset = pd.read_csv("indian_liver_patient_dataset.csv")
    liv_answer = liv_dataset["class"]

    liv_dataset = liv_dataset.drop("class", axis=1)

    liv_transformer = make_column_transformer(
        (oh, ["gender"]),
        (
            idTransformer,
            ["age", "TB", "DB", "alkphos", "sgpt", "sgot", "TP", "ALB", "A_G"],
        ),
    )

    liv_full_set = liv_transformer.fit_transform(liv_dataset)

    liv_train, liv_test, liv_ans_train, liv_ans_test = train_test_split(
        liv_full_set, liv_answer, test_size=0.2, random_state=30
    )

    liv_train = scaler.fit_transform(liv_train)
    liv_test = scaler.fit_transform(liv_test)

    bm_dataset = pd.read_csv("bank_marketing_dataset.csv")
    bm_answer = pd.DataFrame(data=bm_dataset["y"])

    #print(bm_dataset)

    bm_transformer = make_column_transformer(
        (bm_answer_encoder, ["y"]),
    )

    bm_answer = bm_transformer.fit_transform(bm_answer)

    bm_dataset = bm_dataset.drop("y", axis=1)
    # drop predicted outcome
    bm_dataset = bm_dataset.drop("poutcome", axis=1)
    # drop day of month as this is assumed to be a noisy feature
    bm_dataset = bm_dataset.drop("day", axis=1)

    bm_transformer = make_column_transformer(
        (education_encoder, ["education"]),
        (credit_default_encoder, ["default"]),
        (oh, ["job", "marital", "housing", "loan", "contact", "month"]),
        (
            idTransformer,
            ["age", "balance", "duration", "campaign", "pdays", "previous"],
        ),
    )

    bm_full_set = bm_transformer.fit_transform(bm_dataset)

    bm_train, bm_test, bm_ans_train, bm_ans_test = train_test_split(
        bm_full_set, bm_answer, test_size=0.2, random_state=30
    )

    #print(bm_ans_train[6])

    bm_train = scaler.fit_transform(bm_train)
    bm_test = scaler.fit_transform(bm_test)
    print(np.unique(bm_ans_train))
    print(np.unique(bm_ans_test))


def make_save_learning_curve_chart(
    algo, model, classifier_n, scoring, sizes, cv, n_jobs, dataset_name, X_train, y_train
):
    v = LearningCurve(model, cv=cv, scoring=scoring, train_sizes=sizes, n_jobs=n_jobs)
    v.fit(X_train, y_train)
    v.show("files/{}_learning_curve_{}_{}.png".format(classifier_n, dataset_name, algo))
    plt.clf()

#write_to_csv_2(name, "genetic_algo", btime, pred_acc_score, pred_f1_score, train_acc_score, train_f1_score, train_loss_score)

def write_to_csv_2(
    dataset_name, fitness_fn, train_time, pred_acc_score, pred_f1_score, train_acc_score, train_f1_score, train_loss_score
):
    fname = "files/{}_neural_network_{}_metrics.csv".format(dataset_name, fitness_fn)
    try:
        f = open(fname)
    except IOError:
        f = open(fname, "a+")
        f.write(
            "Execution Time,Time to Train,Test Accuracy Score,Test F1 Score,Train Accuracy Score,Train F1 Score,Train Loss Score\n"
        )
    finally:
        f.close()
    with open(fname, "a+") as f:
        f.write(
            "{},{},{},{},{},{},{}\n".format(
                time.time(),
                train_time,
                pred_acc_score,
                pred_f1_score,
                train_acc_score,
                train_f1_score,
                train_loss_score

            )
        )


def main():
    run()
    return


if __name__ == "__main__":
    main()
