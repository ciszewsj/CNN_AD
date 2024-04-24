using Statistics
include("structure.jl")
include("backward.jl")
include("forward.jl")
include("scalar_operators.jl")
include("brodcasted_operators.jl")
include("graph.jl")
include("utils.jl")

update_weight!(node, learning_rate) = node.output -= learning_rate .* node.gradient

function update_weights!(graph::Vector, lr::Float64)
    for node in graph
        if isa(node, Variable) && hasproperty(node, :gradient)
            # node._gradient ./= batch_size
            # println(node.output)
            # println(node.gradient)
            # println(lr)
            # println(size(node.output), " " , size(node.gradient))
            node.output -= lr * node.gradient
            node.gradient .= 0
        end
    end
end

function build_graph(
    x::Constant,
    y::Constant,
    cnn::CNN
)
	# z1 = convolution(x, cnn.wk1) |> relu |> maxpool2d |> flatten
	# z2 = dense(z1, cnn.wd1, cnn.wb1)# |>relu
	# z3 = dense(z2, cnn.wd2, cnn.wb2)

	# println(typeof(z1))
	# println(typeof(z2))
	# println(typeof(z3))

    
    # k1 = Variable(create_kernel(1, 6));
    # k2 = Variable(create_kernel(13*13*6, 84));
    # k3 = Variable(create_kernel(84, 10));

    # b1 = Variable(initialize_uniform_bias(84, 13*13*6));
    # b2 = Variable(initialize_uniform_bias(10, 84));

    #     z1 = convolution(x, k1) |> relu |> maxpool2d |> flatten
    #     z2 = dense(z1, k2, b1) |> relu
    #     z3 = dense(z2, k3, b2)

    z0 = convolution(x, cnn.wk1) |> linear |> maxpool2d |> flatten
    z1 = dense(z0, cnn.wd1, cnn.wb1) |> linear
	# z2 = dense(z1, cnn.wd2 , cnn.wb2) |> linear
	e = cross_entropy_loss(z1, y)
	return topological_sort(e)
end

function do_train(cnn::CNN,
    trainx::Any,
    trainy::Any)
	epoch_loss = 0.0
    for i=1:size(trainx, 3)
        # println(i, " / ", size(trainx, 3))
        x = Constant(trainx[:, :, i])
        y = Constant(trainy[i, :])
        graph = build_graph(x, y, cnn)

		epoch_loss += forward!(graph)
		backward!(graph)
        update_weights!(graph, 1e-4)
    end
    return epoch_loss / size(trainx, 3)
end


function do_magic_trick(x_train::Any, y_train::Any)
	wk1 = Variable(create_kernel(1, 16))

    k1 =  Variable(randn(10, 2704), name = "wh")
    k2 = Variable(randn(2, 10), name = "wo")

    b1 = Variable(initialize_uniform_bias(2704, 10));
    b2 = Variable(initialize_uniform_bias(10, 2));

    c = CNN(wk1, k1, b1, k2, b2)

    i = 1
    # x = Variable(x_train, name="x")
    # y = Variable(y_train, name="y")

    for i=1:100
        epoch_loss = do_train(c, x_train, y_train)
        println("Epoch " ,i," : ", epoch_loss)
    end
end