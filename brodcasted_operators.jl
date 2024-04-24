include("structure.jl")
import Base: ^, sin, sum, *, +, -, max, reshape
import LinearAlgebra: mul!, diagm
import NNlib

reshape(x::GraphNode, new_size::GraphNode) = let 
    # println(x.name, " = ", x.size, ">>>", new_size.name)
    BroadcastedOperator(reshape, x, new_size)
end


^(x::GraphNode, n::Number) = BroadcastedOperator(^, x, n)
forward(::BroadcastedOperator{typeof(^)}, x, n) = x .^ n
backward(::BroadcastedOperator{typeof(^)}, x, n, g) = tuple(g .* n .* x .^ (n - 1), nothing)


*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)


relu(x::GraphNode) = BroadcastedOperator(relu, x)
forward(::BroadcastedOperator{typeof(relu)}, x) = return max.(x, zero(x))
backward(::BroadcastedOperator{typeof(relu)}, x, g) = return tuple(g .* (x .> 0))


logistic(x::GraphNode) = BroadcastedOperator(logistic, x)
forward(::BroadcastedOperator{typeof(logistic)}, x) = 1 ./ (1 .+ exp.(-x))
backward(::BroadcastedOperator{typeof(logistic)}, x, g) = tuple(g .* exp.(x) ./ (1 .+ exp.(x)) .^ 2)


flatten(x::GraphNode) = BroadcastedOperator(flatten, x)
forward(::BroadcastedOperator{typeof(flatten)}, x) = reshape(x, length(x))
backward(::BroadcastedOperator{typeof(flatten)}, x, g) = tuple(reshape(g, size(x)))


Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x, y) = x .* y
backward(node::BroadcastedOperator{typeof(*)}, x, y, g) =
    let
        𝟏 = ones(length(node.output))
        Jx = diagm(vec(y .* 𝟏))
        Jy = diagm(vec(x .* 𝟏))
        tuple(Jx' * g, Jy' * g)
    end


Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = tuple(g, -g)


Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) = x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = tuple(g, g)


sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x) = sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, g) =
    let
        𝟏 = 
        J = 𝟏'
        tuple(ones(length(x))'' * g)
    end


Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x, y) = x ./ y
function backward(node::BroadcastedOperator{typeof(/)}, x, y::Real, g)
    let
        𝟏 = ones(length(node.output))
        Jx = diagm(𝟏 ./ y)
        Jy = (-x ./ y .^ 2)
        tuple(Jx' * g, Jy' * g)
    end
end


Base.Broadcast.broadcasted(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x, y) = max.(x, y)
backward(::BroadcastedOperator{typeof(max)}, x, y, g) =
    let
        Jx = diagm(isless.(y, x))
        Jy = diagm(isless.(x, y))
        tuple(Jx' * g, Jy' * g)
    end


cross_entropy_loss(y_hat::GraphNode, y::GraphNode) = BroadcastedOperator(cross_entropy_loss, y_hat, y)
forward(::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y) =
    let
        println(y_hat, " <> ", y)
        y_hat = y_hat .- maximum(y_hat)
        y_hat = exp.(y_hat) ./ sum(exp.(y_hat))
        loss = sum(log.(y_hat) .* y) * -1.0
        return loss
    end
backward(node::BroadcastedOperator{typeof(cross_entropy_loss)}, y_hat, y, g) =
    let
        y_hat = y_hat .- maximum(y_hat)
        y_hat = exp.(y_hat) ./ sum(exp.(y_hat))
        return tuple(g .* (y_hat - y))
    end



add_dim(x::Array) = reshape(x, (size(x)..., 1))


convolution(x::GraphNode, kernel::GraphNode) = BroadcastedOperator(convolution, x, kernel)
forward(::BroadcastedOperator{typeof(convolution)}, x, kernel) =
    let
        input = @view x[:, :, :, 1:1]
        output = @view conv_op(input, kernel, flipped = true)[:, :, :, 1]
        return output
    end

backward(::BroadcastedOperator{typeof(convolution)}, x, kernel, g) =
    let
        x = add_dim(x)
        if size(g)[end] != 1
            g = add_dim(g)
        end

        kernel_gradient = permutedims(
            conv_op(
                permutedims(x, (1, 2, 4, 3)),
                permutedims(g, (1, 2, 4, 3)),
                flipped = true,
            ),
            (1, 2, 4, 3),
        )
        input_gradient =
            conv_op(g, permutedims(kernel, (1, 2, 4, 3)), pad = 2, flipped = false)
        return tuple(input_gradient, kernel_gradient)
    end

linear(x::GraphNode) = BroadcastedOperator(linear, x)
forward(::BroadcastedOperator{typeof(linear)}, x) = x
backward(::BroadcastedOperator{typeof(linear)}, x, g) = tuple(g)


dense(x::GraphNode, w::GraphNode, b::GraphNode) = BroadcastedOperator(dense, x, w, b)
forward(::BroadcastedOperator{typeof(dense)}, x, w, b) = let 
    println("w: ",typeof(w), " x: ", typeof(x))
    println("w: ",size(w), " x: ", size(x))
    # println("w",w)
    # println("x",x)
    # a,b = size(w)
    # x = reshape(x, b, :)
    w * x
end
backward(::BroadcastedOperator{typeof(dense)}, x, w, b, g) = let
    # println("x: ", typeof(x), "w: ", typeof(w), "b: ", typeof(b), "g: ", typeof(g))
    # println("x: ", size(x), "w: ", size(w), "b: ", size(b), "g: ", size(g))
    # a,b = size(w)
    # x = reshape(x, b, :)
    tuple(w' * g, g * x', g)
end
    

maxpool2d(x::GraphNode) = BroadcastedOperator(maxpool2d, x)
forward(node::BroadcastedOperator{typeof(maxpool2d)}, x) =
    let
        h, w, c = size(x)
        output = zeros(h ÷ 2, w ÷ 2, c)
        indices = CartesianIndex{3}[]
        for i = 1:c
            for j = 1:h÷2
                for k = 1:w÷2
                    val, ids = findmax(@view x[2*j-1:2*j, 2*k-1:2*k, i])
                    output[j, k, i] = val

                    idx, idy = ids[1] + 2 * j - 1 - 1, ids[2] + 2 * k - 1 - 1
                    push!(indices, CartesianIndex(idx, idy, i))
                end
            end
        end
        node.cache = indices
        output
    end

backward(node::BroadcastedOperator{typeof(maxpool2d)}, x, g) =
    let
        output = zeros(size(x))
        output[node.cache] = vcat(g...)
        tuple(output)
    end
