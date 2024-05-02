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
    for node in graph
        if isa(node, Variable) && hasproperty(node, :__gradient)
            node.__gradient ./= batch_size
            node.output -= lr * node.__gradient
            node.__gradient .= 0
        end
    end
end

poprawne = 0
suma2 = 0
function do_test(cnn::CNN2,
    x_data::Any,
    y_data::Any)
    for i=1:size(x_data, 3)
        x = Constant(x_data[:, :, i])
        y = Constant(y_data[i, :])
        graph = build_graph(x, y, cnn)
		forward!(graph)
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
    batch_size,
    lr)
	epoch_loss = 0.0
    for i=1:size(trainx, 3)
        x = Constant(trainx[:, :, i])
        y = Constant(trainy[i, :])
        graph = build_graph(x, y, cnn)

		epoch_loss += forward!(graph)
		backward!(graph)
        if i % batch_size == 0
            update_weights!(graph, lr, batch_size)
        end
    end
    return epoch_loss / size(trainx, 3)
end


function do_magic_trick_2(x_train::Any, y_train::Any, x_test::Any, y_test::Any,  batch_size, lr, epoch)

    println("RUN CNN")
    println("   INPUT SIZE  : ", size(x_train))
    println("   OUTPUT SIZE : ", size(y_train))
    println("   BATCH SIZE  : ", batch_size)
    println("   lr          : ", lr)
    println("   epoch       : ", epoch)


	c1 = Variable(create_kernel(1, 6))
	c2 = Variable(create_kernel(6, 16))

    k1 =  Variable(randn(84, 400), name = "wh")
    k2 = Variable(randn(10, 84), name = "wo")

    b1 = Variable(initialize_uniform_bias(400, 84));
    b2 = Variable(initialize_uniform_bias(84, 10));

    c = CNN2(c1, c2, k1, k2, b1, b2)

    i = 1
    global poprawne
    global suma2
    poprawne = 0
    suma2 = 0

    @time for i=1:epoch
        epoch_loss = do_train(c, x_train, y_train, batch_size, lr)
        println("Epoch " ,i," : ", epoch_loss)
        println("   ACCURACY : ", poprawne/suma2)
        poprawne = 0
        suma2 = 0
    end

    do_test(c, x_train, y_train)
    println("TRAIN DATA")
    println("   ACCURACY : ", poprawne/suma2)    
    poprawne = 0
    suma2 = 0
    do_test(c, x_test, y_test)
    println("TEST DATA")
    println("   ACCURACY : ", poprawne/suma2)
    poprawne = 0
    suma2 = 0
end