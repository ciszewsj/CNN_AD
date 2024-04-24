using Statistics
include("structure.jl")
include("backward.jl")
include("forward.jl")
include("scalar_operators.jl")
include("brodcasted_operators.jl")
include("graph.jl")
include("utils.jl")

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

    
k1 = Variable(create_kernel(1, 6));
k2 = Variable(create_kernel(13*13*6, 84));
k3 = Variable(create_kernel(84, 10));

b1 = Variable(initialize_uniform_bias(84, 13*13*6));
b2 = Variable(initialize_uniform_bias(10, 84));

    z1 = convolution(x, k1) |> relu |> maxpool2d |> flatten
    z2 = dense(z1, k2, b1) |> relu
    z3 = dense(z2, k3, b2)


	e = cross_entropy_loss(z3, y)
	return topological_sort(e)
end

function do_train(cnn::CNN,
    trainx::Array{Float32, 3},
    trainy::Array{Int32, 2})
	epoch_loss = 0.0
    # iter = ProgressBar(1:size(trainx, 3), printing_delay = 0.1)
    for i in axes(1, size(trainx, 3))
        x = Constant(add_dim(trainx[:, :, i]))
        y = Constant(trainy[i, :])

        graph = build_graph(x, y, cnn)
		loss = forward!(graph)
		backward!(graph)
        println("loss:", loss)
		# if i % batch_size == 0
        #     step!(graph, lr, batch_size)
        # end
        # set_description(iter, string(@sprintf("Train Loss: %.3f", epoch_loss / i)))
    end
    return epoch_loss / size(trainx, 3)
end


function do_magic_trick(x_train::Array{Float32, 3}, y_train::Array{Int32})
	wk1 = Variable(create_kernel(1, 16))
    wd1 = Variable(kaiming_normal_weights(128, 64))
    wb1 = Variable(initialize_uniform_bias(64, 128))
    wd2 = Variable(kaiming_normal_weights(10, 128))
    wb2 = Variable(initialize_uniform_bias(128, 10))

    c = CNN(wk1, wd1, wb1, wd2, wb2)

    i = 1
    # x = Variable(x_train, name="x")
    # y = Variable(y_train, name="y")

	do_train(c, x_train, y_train)
end