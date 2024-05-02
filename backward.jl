include("structure.jl")

update!(node::Constant, gradient) = nothing

update!(node::GraphNode, gradient) = begin
    node.gradient = gradient
    if typeof(node) == Variable
        if isnothing(node.__gradient)
            node.__gradient = gradient * 0
        else
            node.__gradient .+= gradient
        end
    end
end


function backward!(order::Vector; seed=1.0)
    result = last(order)
    result.gradient = seed
    for node in reverse(order)
        backward!(node)
    end
end

backward!(node::Constant) = nothing
backward!(node::Variable) = nothing


function backward!(node::Operator)
    inputs = node.inputs
    input_gradients = backward(node, [input.output for input in inputs]..., node.gradient)
    for (input, input_gradient) in zip(inputs, input_gradients)
        update!(input, input_gradient)
    end
    return nothing
end
