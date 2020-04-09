# Todo:
#       derivative of softmax.
#       convert all y into matrices.
#       gradient.
#       dropout w/ probability.
#       shuffle input X.

import torch


class TwoLayerMLP:
    def __init__(self, n_iter=1000, batch_size=32, hidden_layer_width=100,
                 hidden_activation='relu', output_activation='softmax'):
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.hidden_layer_width = hidden_layer_width
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.V = torch.tensor([])
        self.W = torch.tensor([])
        self.X = torch.tensor([])
        self.Y = torch.tensor([])

    @staticmethod
    def hyperbolic_tangent(Z):
        return Z.tanh()

    @staticmethod
    def hyperbolic_tangent_derivative(Z):
        return 1 - Z.tanh() ** 2

    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + torch.exp(-Z))

    @staticmethod
    def relu(Z):
        return torch.clamp(Z, min=0)

    @staticmethod
    def relu_subgradient(Z):
        return torch.clamp(torch.sign(Z), min=0)

    @staticmethod
    def softmax(Z):
        # Transposing is to allow for row-wise operations (with built-in broadcasting)
        Z_0_transpose = Z.t() - Z.max()
        Y_hat_transpose = torch.exp(Z_0_transpose)
        Y_hat_transpose /= Y_hat_transpose.sum(0)
        Y_hat = Y_hat_transpose.t()
        return Y_hat

    @staticmethod
    def cross_entropy_loss(y_hat, y):
        return -torch.dot(y, torch.log(y_hat))

    @staticmethod
    def tensor_width(tensor):
        if len(list(tensor.shape)) == 2:
            return tensor.size(1)
        elif len(list(tensor.shape)) == 1:
            return 1
        else:
            raise ValueError('Tensor must be one or two dimensions.')

    @staticmethod
    def matrixify(y):
        num_categories = int(y.max()) + 1
        num_entries = len(y)
        y = y.long()
        Y = torch.zeros(num_entries, num_categories)
        for i in range(num_entries):
            Y[i, y[i]] = 1
        return Y

    def sigmoid_derivative(self, Z):
        return self.sigmoid(Z) * (1 - self.sigmoid(Z))

    def activation(self, Z, output=False, derivative=False):
        if not output:
            function = self.hidden_activation
        else:
            function = self.output_activation

        if not derivative:
            if function == 'relu':
                return self.relu(Z)
            elif function == 'tanh':
                return self.hyperbolic_tangent(Z)
            elif function == 'sigmoid':
                return self.sigmoid(Z)
            elif function == 'softmax':
                return self.softmax(Z)
            else:
                raise ValueError("Invalid activation function.")
        else:
            if function == 'relu':
                return self.relu_subgradient(Z)
            elif function == 'tanh':
                return self.hyperbolic_tangent_derivative(Z)
            elif function == 'sigmoid':
                return self.sigmoid_derivative(Z)
            elif function == 'softmax':
                return
            else:
                raise ValueError("Invalid activation function.")

    def verify_args(self):
        if self.batch_size > self.X.size(0):
            raise ValueError('Mini-batch size cannot be larger than data set.')

    def initialize_weights(self):
        y_width = self.tensor_width(self.Y)
        self.W = torch.rand(self.hidden_layer_width, self.X.size(1))
        self.V = torch.rand(y_width, self.hidden_layer_width)

    def predict_one(self, x):
        z1 = torch.mv(self.W, x)
        a = self.activation(z1)
        z2 = torch.mv(self.V, a)
        y_hat = self.activation(z2, output=True)
        return y_hat

    def ith_batch(self, i):
        """Selects a mini-batch of samples allowing for wraparound.
        """

        first_sample = i * self.batch_size % self.X.size(0)

        if self.batch_size <= (self.X.size(0) - first_sample):
            return (self.X[first_sample:first_sample + self.batch_size, :],
                    self.Y[first_sample:first_sample + self.batch_size, :])
        else:
            input_end = self.X[first_sample:, :]
            label_end = self.Y[first_sample:, :]
            input_wraparound = self.X[:(self.batch_size - len(input_end)), :]
            label_wraparound = self.Y[:(self.batch_size - len(label_end)), :]
            inputs = torch.cat((input_end, input_wraparound))
            labels = torch.cat((label_end, label_wraparound))
            return inputs, labels

    def batch_cost(self, inputs, labels):
        cost = 0
        for r in range(self.batch_size):
            y_hat = self.predict_one(inputs[r, :])
            cost += self.cross_entropy_loss(y_hat, labels[r])
        return cost/self.batch_size

    def train_one_batch(self, inputs, labels):
        cost = self.batch_cost(inputs, labels)
        W, V = self.update_weights()
        return

    def train(self, X, y):
        self.X = X
        self.Y = self.matrixify(y)
        self.verify_args()
        self.initialize_weights()

        for i in range(self.n_iter):
            inputs, labels = self.ith_batch(i)
            self.train_one_batch(inputs, labels)


if __name__ == '__main__':
    mlp = TwoLayerMLP(hidden_activation='softmax', output_activation='softmax')
    X = torch.arange(100).repeat(3, 1).t()
    y = (torch.rand(30)*10).round()

    # mlp.train(X, y)
    # print(mlp.ith_batch(4))
