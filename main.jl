# using MLDatasets: MNIST
# using Flux
include("model.jl")

conv_op = NNlib.conv

# train_ds = MNIST(:train)

#  x = reshape(train_ds.features, 28, 28, :)
#  y  = Flux.onehotbatch(train_ds.targets, 0:9) 
# println(typeof(x) , " " , typeof(y))
# println(size(x) , " " , size(y))

# y = Int32.(Array(y))
# x= [0.96900620424131; 0.11711241495755131;;;
#  0.1932924381964406; 0.6678911560504353;;;
#   0.877333879469451; 0.3855336441180791]
x = rand(28,28)
y = rand(28,10)
# y= [1.0 0.0;
#     0.0 0.0;
#     1.0 1.0]

println(typeof(x), " " , typeof(y))
println(size(x), " " , size(y))
do_magic_trick(x, y)