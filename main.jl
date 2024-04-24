using MLDatasets: MNIST
using Flux
include("model.jl")

conv_op = NNlib.conv

train_ds = MNIST(:train)

x = reshape(train_ds.features, 28, 28, :)
y  = Flux.onehotbatch(train_ds.targets, 0:9) 

y = Int32.(Array(y))
do_magic_trick(x, y)