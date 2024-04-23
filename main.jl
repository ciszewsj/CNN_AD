using MLDatasets: MNIST
include("model.jl")

train_data = MNIST(split=:train)

x_train = train_data.features
y_train = train_data.targets
x_train = reshape(x_train, size(x_train, 1), size(x_train, 2), 1, size(x_train, 3))
y_train = y_train .== 5
x_train = x_train[:, :, :, 1:100]
y_train = y_train[1:100, :]

x = x_train
y = y_train

train_model(x, y, 0.1, 100, false)