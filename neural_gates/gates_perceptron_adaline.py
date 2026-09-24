import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, learning_rate=19, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors_per_epoch = []

    def step_function(self, z):
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                linear_output = np.dot(xi, self.weights) + self.bias
                prediction = self.step_function(linear_output)
                update = self.learning_rate * (target - prediction)

                self.weights += update * xi
                self.bias += update

                errors += int(update != 0.0)

            self.errors_per_epoch.append(errors)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.step_function(linear_output)


class Adaline:
    def __init__(self, learning_rate=0.95, epochs=50):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_per_epoch = []

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def activation(self, X):
        return self.net_input(X)

    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            output = self.activation(X)
            errors = y - output

            self.weights += self.learning_rate * X.T.dot(errors)
            self.bias += self.learning_rate * errors.sum()

            loss = (errors ** 2).mean() / 2.0
            self.loss_per_epoch.append(loss)

    def predict(self, X):
        output = self.activation(X)
        return np.where(output >= 0.5, 1, 0)


def print_results(model_name, gate_name, X, y_true, y_pred):
    print(f"\n{model_name} on {gate_name} gate")
    print("Input -> Target -> Prediction")
    for xi, t, p in zip(X, y_true, y_pred):
        print(f"{xi} -> {t} -> {p}")


def main():
    # Truth table inputs
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    # Targets
    y_and = np.array([0, 0, 0, 1])
    y_or = np.array([0, 1, 1, 1])

    # Perceptron for AND
    perceptron_and = Perceptron(learning_rate=0.1, epochs=20)
    perceptron_and.fit(X, y_and)
    pred_and_p = perceptron_and.predict(X)
    print_results("Perceptron", "AND", X, y_and, pred_and_p)
    print("Perceptron AND weights:", perceptron_and.weights)
    print("Perceptron AND bias:", perceptron_and.bias)

    # Perceptron for OR
    perceptron_or = Perceptron(learning_rate=0.1, epochs=20)
    perceptron_or.fit(X, y_or)
    pred_or_p = perceptron_or.predict(X)
    print_results("Perceptron", "OR", X, y_or, pred_or_p)
    print("Perceptron OR weights:", perceptron_or.weights)
    print("Perceptron OR bias:", perceptron_or.bias)

    # Adaline for AND
    adaline_and = Adaline(learning_rate=0.01, epochs=50)
    adaline_and.fit(X, y_and)
    pred_and_a = adaline_and.predict(X)
    print_results("Adaline", "AND", X, y_and, pred_and_a)
    print("Adaline AND weights:", adaline_and.weights)
    print("Adaline AND bias:", adaline_and.bias)

    # Adaline for OR
    adaline_or = Adaline(learning_rate=0.01, epochs=50)
    adaline_or.fit(X, y_or)
    pred_or_a = adaline_or.predict(X)
    print_results("Adaline", "OR", X, y_or, pred_or_a)
    print("Adaline OR weights:", adaline_or.weights)
    print("Adaline OR bias:", adaline_or.bias)

    # Plot learning curves
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(perceptron_and.errors_per_epoch, label="AND")
    plt.plot(perceptron_or.errors_per_epoch, label="OR")
    plt.title("Perceptron Errors per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Number of Misclassifications")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(adaline_and.loss_per_epoch, label="AND")
    plt.plot(adaline_or.loss_per_epoch, label="OR")
    plt.title("Adaline Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error / 2")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()