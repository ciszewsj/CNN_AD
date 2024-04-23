include("scalar_operators.jl")
include("structure.jl")
include("graph.jl")
include("forward.jl")
include("backward.jl")

x = Variable(5.0, 0.0)
t = Variable(3.0, 0.0)

two = Constant(2.0)
four = Constant(4.0)

squared = relu(sin(max(log(x)/two/four * four, log(t / two) * four)))

order = topological_sort(squared)

y = forward!(order)

backward!(order)

println(x.gradient)
println(t.gradient)