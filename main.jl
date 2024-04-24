using MLDatasets: MNIST
using Flux
include("model2.jl")

train_ds = MNIST(:train)
test_data  = MNIST(split=:test)

x_train = reshape(train_ds.features, 28, 28, :)
y_train  = Flux.onehotbatch(train_ds.targets, 0:9)

x_test = reshape(test_data.features, 28, 28, :)
y_test  = Flux.onehotbatch(test_data.targets, 0:9)

do_magic_trick_2(x_train, y_train', x_test, y_test')