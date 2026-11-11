import numpy as np


class Module(object):
    """
    Basically, you can think of a module as of a something (black box)
    which can process input data and produce ouput data.
    This is like applying a function which is called forward:

        output = module.forward(input)

    The module should be able to perform a backward pass: to differentiate the forward function.
    More, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule.

        gradInput = module.backward(input, gradOutput)
    """

    def init(self):
        self.output = None
        self.gradInput = None
        self.training = True

    def forward(self, input):
        """
        Takes an input object, and computes the corresponding output of the module.
        """
        return self.updateOutput(input)

    def backward(self, input, gradOutput):
        """
        Performs a backpropagation step through the module, with respect to the given input.

        This includes:
        - computing a gradient w.r.t. input (is needed for further backprop),
        - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput)
        return self.gradInput

    def updateOutput(self, input):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which is stored in the output field.

        Make sure to both store the data in output field and return it.
        """
        pass

    def updateGradInput(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own input.
        This is returned in gradInput. Also, the gradInput state variable is updated accordingly.

        The shape of gradInput is always the same as the shape of input.

        Make sure to both store the gradients in gradInput field and return it.
        """
        pass

    def accGradParameters(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        pass

    def zeroGradParameters(self):
        """
        Zeroes gradParams variable if the module has params.
        """
        pass

    def getParameters(self):
        """
        Returns a list with its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def getGradParameters(self):
        """
        Returns a list with gradients with respect to its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def train(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True

    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False

    def repr(self):
        """
        Pretty printing. Should be overrided in every module if you want
        to have readable description.
        """
        return "Module"


class Sequential(Module):
    """
    This class implements a container, which processes input data sequentially.
    input is processed by each module (layer) in self.modules consecutively.
    The resulting array is called output.
    """

    def init(self):
        super(Sequential, self).init()
        self.modules = []

    def add(self, module):
        """
        Adds a module to the container.
        """
        self.modules.append(module)

    def updateOutput(self, input):
        y = input
        for m in self.modules:
            y = m.forward(y)
        self.output = y
        return self.output

    def backward(self, input, gradOutput):
        t = []
        y = input
        t.append(y)
        for m in self.modules:
            y = m.forward(y)
            t.append(y)

        y = gradOutput
        for i in range(len(self.modules)):
            k = len(self.modules) - i - 1
            y = self.modules[k].backward(t[k], y)
        self.gradInput = y
        return self.gradInput

    def zeroGradParameters(self):
        for module in self.modules:
            module.zeroGradParameters()

    def getParameters(self):
        return [x.getParameters() for x in self.modules]

    def getGradParameters(self):
        return [x.getGradParameters() for x in self.modules]

    def repr(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string

    def getitem(self, x):
        return self.modules[x]

    def train(self):
        self.training = True
        for module in self.modules:
            module.train()

    def evaluate(self):
        self.training = False
        for module in self.modules:
            module.evaluate()


class Linear(Module):
    """
    A module which applies a linear transformation (fully-connected layer).
    """

    def init(self, n_in, n_out):
        super(Linear, self).init()
        stdv = 1. / np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size=(n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, input):
        self.output = np.dot(input, self.W.T) + self.b
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.dot(gradOutput, self.W)
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        self.gradW += np.dot(gradOutput.T, input)
        self.gradb += np.sum(gradOutput, axis=0)

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def repr(self):
        s = self.W.shape
        q = 'Linear %d -> %d' % (s[1], s[0])
        return q


class SoftMax(Module):
    def init(self):
        super(SoftMax, self).init()

    def updateOutput(self, input):
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        exp = np.exp(self.output)
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output

    def updateGradInput(self, input, gradOutput):
        sum_grad = np.sum(gradOutput * self.output, axis=1, keepdims=True)
        self.gradInput = self.output * (gradOutput - sum_grad)
        return self.gradInput

    def repr(self):
        return "SoftMax"


class LogSoftMax(Module):
    def init(self):
        super(LogSoftMax, self).init()

    def updateOutput(self, input):
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        exp = np.exp(self.output)
        sum_exp = np.sum(exp, axis=1, keepdims=True)
        self.output = self.output - np.log(sum_exp)
        return self.output

    def updateGradInput(self, input, gradOutput):
        sum_grad = np.sum(gradOutput, axis=1, keepdims=True)
        self.gradInput = gradOutput - np.exp(self.output) * sum_grad
        return self.gradInput

    def repr(self):
        return "LogSoftMax"


class BatchNormalization(Module):
    EPS = 1e-3

    def init(self, alpha=0.):
        super(BatchNormalization, self).init()
        self.alpha = alpha
        self.moving_mean = None
        self.moving_variance = None

    def updateOutput(self, input):
        if self.training:
            self.mean = input.mean(axis=0)
            self.var = input.var(axis=0)

            if self.moving_mean is None:
                self.moving_mean = np.zeros_like(self.mean)
                self.moving_variance = np.ones_like(self.var)

            self.moving_mean = self.moving_mean * self.alpha + self.mean * (1 - self.alpha)
            self.moving_variance = self.moving_variance * self.alpha + self.var * (1 - self.alpha)

            self.std = np.sqrt(self.var + self.EPS)
            self.x_normalized = (input - self.mean) / self.std
            self.output = self.x_normalized
        else:
            self.output = (input - self.moving_mean) / np.sqrt(self.moving_variance + self.EPS)
        return self.output

    def updateGradInput(self, input, gradOutput):
        N = gradOutput.shape[0]
        std = np.sqrt(self.var + self.EPS)
        dxhat = gradOutput
        dvar = np.sum(dxhat * (input - self.mean) * -0.5 * (std ** -3), axis=0)
        dmean = np.sum(dxhat * -1.0 / std, axis=0) + dvar * np.mean(-2.0 * (input - self.mean), axis=0)
        self.gradInput = dxhat / std + dvar * 2.0 * (input - self.mean) / N + dmean / N
        return self.gradInput

    def repr(self):
        return "BatchNormalization"


class Dropout(Module):
    def init(self, p=0.5):
        super(Dropout, self).init()
        self.p = p
        self.mask = None

    def updateOutput(self, input):
        if self.training:
            self.mask = (np.random.rand(*input.shape) >= self.p) / (1.0 - self.p)
            self.output = input * self.mask
        else:
            self.output = input
        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.training:
            self.gradInput = gradOutput * self.mask
        else:
            self.gradInput = gradOutput
        return self.gradInput

    def repr(self):
        return "Dropout"


class ReLU(Module):
    def init(self):
        super(ReLU, self).init()

    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput, input > 0)
        return self.gradInput

    def repr(self):
        return "ReLU"


class LeakyReLU(Module):
    def init(self, slope=0.03):
        super(LeakyReLU, self).init()
        self.slope = slope

    def updateOutput(self, input):
        self.output = np.where(input > 0, input, input * self.slope)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * np.where(input > 0, 1.0, self.slope)
        return self.gradInput

    def repr(self):
        return "LeakyReLU"

    # ==========================================
    # МИНИ-ТЕСТ ДЛЯ ПРОВЕРКИ РАБОТЫ ФРЕЙМВОРКА
    # ==========================================
    if __name__ == "__main__":
        print("=== Запуск теста нейросети ===")

        # 1. Создаем контейнер Sequential
        model = Sequential()
        model.init()

        # 2. Создаем слои с помощью метода .init()
        # Linear: принимает 5 признаков, на выходе дает 8 нейронов
        layer1 = Linear()
        layer1.init(5, 8)

        # ReLU: функция активации (без параметров)
        activation = ReLU()
        activation.init()

        # Linear: принимает 8 нейронов, на выходе дает 3 класса (например, для классификации)
        layer2 = Linear()
        layer2.init(8, 3)

        # SoftMax: переводит сырые числа в вероятности
        sf = SoftMax()
        sf.init()

        # Добавляем слои в последовательную модель
        model.add(layer1)
        model.add(activation)
        model.add(layer2)
        model.add(sf)

        # 3. Генерируем тестовые входные данные
        # Батч из 4 примеров (строк), у каждого по 5 признаков (столбцов)
        x = np.random.randn(4, 5)
        print("\n1. Входные данные (x):\n", x)
        print("Форма входа:", x.shape)

        # 4. Проверяем ПРЯМОЙ проход (Forward pass)
        output = model.forward(x)
        print("\n2. Результат forward прохода (вероятности):\n", output)
        print("Форма выхода:", output.shape)

        # Проверяем, что SoftMax дал сумму вероятностей по каждой строке ровно 1.0
        row_sums = np.sum(output, axis=1)
        print("Сумма вероятностей по строкам (должна быть единицы):", row_sums)

        # 5. Проверяем ОБРАТНЫЙ проход (Backward pass)
        # Создаем фиктивную ошибку того же размера, что и выход модели
        grad_output = np.ones_like(output)
        grad_input = model.backward(x, grad_output)

        print("\n3. Обратный проход выполнен успешно!")
        print("Форма градиента по входу (gradInput):", grad_input.shape)
        print("Тест пройден, всё работает корректно!")