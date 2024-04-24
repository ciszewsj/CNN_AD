using MLDatasets: MNIST
using Flux
include("model.jl")

train_ds = MNIST(:train)

x = reshape(train_ds.features, 28, 28, :)#[:, :, 1:800]
y  = Flux.onehotbatch(train_ds.targets, 0:9)#[:, 1:800]

println(typeof(x), " " , typeof(y))
println(size(x), " " , size(y))
do_magic_trick(x, y')