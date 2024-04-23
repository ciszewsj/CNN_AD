using Statistics
include("structure.jl")
include("backward.jl")
include("forward.jl")
include("scalar_operators.jl")
include("brodcasted_operators.jl")
include("graph.jl")

function save_param_gradients!(graph, gradients_in_batch)
	for (idx, node) in enumerate(graph)
		# if has_name(node) && is_parameter(node)
		# 	if is_bias(node)
		# 		average_bias_gradient!(node)
		# 	end
		# 	if !haskey(gradients_in_batch, node.name)
		# 		gradients_in_batch[node.name] = Vector{AbstractArray{<:Real, 4}}()
		# 	end
		# 	push!(gradients_in_batch[node.name], node.gradient)
		# 	# println("save_param_gradients!: ", typeof(node.gradient))
		# end
	end
end

function build_graph(
    x::Constant,
    y::Constant,
    cnn::CNN
)
    # input_size = 28
	# kernel_size = 3
	# input_channels = 1
	# out_channels = 4
	# x = Variable(randn(input_size, input_size, input_channels, 1)::Array{Float64, 4}, 0.0)
	# wh = Variable(randn(kernel_size, kernel_size, input_channels, out_channels), 0.0)
	# bh = Variable(randn(1, 1, out_channels, 1)::Array{Float64, 4}, 0.0)
	# wo = Variable(randn((input_size - 2) * (input_size - 2) * out_channels, 1)::Matrix{Float64}, 0.0)
	# bo = Variable(randn(1, 1)::Matrix{Float64}, 0.0)
	# y = Variable(randn(1)::Vector{Float64}, 0.0)

	# print(typeof(x.output))
	# print(typeof(wh.output))
	# print(typeof(bh.output))
	# print(typeof(wo.output))
	# print(typeof(bo.output))
	# print(typeof(y.output))

	z1 = convolution(x, k1) |> relu |> maxpool2d |> flatten
	z2 = dense(z1, cnn.wd1, cnn.wb1) |>relu
	z3 = dense(z2, cnn.wd2, cnn.wb2)

	e = mean_squared_loss(y, z3)
	return topological_sort(e)
end

function train_model(x, y, learning_rate, n_iterations, if_print)
    graph, x_node, y_node = build_graph()
    avg_losses = Vector{Float64}()
    times = Vector{Float64}()
    for iter ∈ 1:n_iterations
        losses_in_batch = Vector{Float64}()
		gradients_in_batch = Dict{String, Vector{AbstractArray}}()
        mean_gradients = Dict{String, AbstractArray}()
        println("Batch $iter")
        for i ∈ 1:size(x, 4)
            curr_x = x[:, :, :, i]
			curr_x = reshape(curr_x, size(curr_x, 1), size(curr_x, 2), size(curr_x, 3), 1)
            curr_y = y[i, :]
            x_node.output = curr_x
			y_node.output = curr_y

            forward!(graph)
            backward!(graph)

            save_param_gradients!(graph, gradients_in_batch)
            loss = graph[end].output[1]
			push!(losses_in_batch, loss)
        end

        for (key, value) in gradients_in_batch
            matrixes = gradients_in_batch[key]
            sum = matrixes[1]
            n_matrixes = length(matrixes)
            for i ∈ 2:n_matrixes
				sum .+= matrixes[i]
			end
			mean = sum ./ n_matrixes
			mean_gradients[key] = mean
        end

        for (idx, node) in enumerate(graph)
            # node.output -= learning_rate .* mean_gradients[node.name]
        end

        avg_loss = mean(losses_in_batch)
        println(avg_loss)
		push!(avg_losses, avg_loss)
    end
    return avg_losses, graph, times
end  


