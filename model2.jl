using Statistics
include("structure.jl")
include("backward.jl")
include("forward.jl")
include("scalar_operators.jl")
include("brodcasted_operators.jl")
include("graph.jl")
include("utils.jl")

update_weight!(node, learning_rate) = node.output -= learning_rate .* node.gradient

function update_weights!(graph::Vector, lr::Float64, batch_size::Int64)
    # print("\n\n\n======UPDATE HERE ======\n\n\n")

    for node in graph
        if isa(node, Variable) && hasproperty(node, :__gradient)
            # node.__gradient ./= batch_size
            node.output -= lr * node.gradient
            # node.__gradient .= 0
        end
    end
end

function build_graph(
    x::Constant,
    y::Constant,
    cnn::CNN2
)
    z0 = convolution(x, cnn.c1) |> relu |> maxpool2d
    z1 = convolution(z0, cnn.c2) |> relu |> maxpool2d |>flatten
    z2 = dense(z1, cnn.wd1, cnn.wb1) |> relu
	z3 = dense(z2, cnn.wd2 , cnn.wb2)
	e = cross_entropy_loss(z3, y)
	return topological_sort(e)
end


function do_train(cnn::CNN2,
    trainx::Any,
    trainy::Any,
    batch_size = 1)
	epoch_loss = 0.0
    for i=1:size(trainx, 3)
        x = Constant(trainx[:, :, i])
        y = Constant(trainy[i, :])
        graph = build_graph(x, y, cnn)

		epoch_loss += forward!(graph)
		backward!(graph)
        # if i % batch_size == 0
            update_weights!(graph, 1e-2, batch_size)
            # println(cnn.wb2)
        # end
    end
    return epoch_loss / size(trainx, 3)
end


function do_magic_trick_2(x_train::Any, y_train::Any)
	c1 = Variable(create_kernel(1, 6))
	c2 = Variable(create_kernel(6, 16))

    k1 =  Variable(randn(84, 400), name = "wh")
    k2 = Variable(randn(10, 84), name = "wo")

    b1 = Variable(initialize_uniform_bias(400, 84));
    b2 = Variable(initialize_uniform_bias(84, 10));

    c = CNN2(c1, c2, k1, k2, b1, b2)

    i = 1

    for i=1:200
        epoch_loss = do_train(c, x_train, y_train)
        println("Epoch " ,i," : ", epoch_loss)
        global suma2 = 0
        global poprawne = 0
    end
end