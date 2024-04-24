include("structure.jl")

reset!(node::Constant) = nothing
reset!(node::Variable) = node.gradient = nothing
reset!(node::Operator) = node.gradient = nothing

compute!(node::Constant) = node.output
compute!(node::Variable) = node.output


pr(node::Any)=begin
    println(typeof(node))
end

pr(node::BroadcastedOperator)=begin
    println(typeof(node), "   ", node.name, "   ", typeof(node.inputs), typeof(node.output))
end


function compute!(node::Operator)
    inputs = [input.output for input in node.inputs]
    pr(node)

    println(typeof(node), typeof(inputs))
    node.output = forward(node, inputs...)
    return node.output
end

function forward!(order::Vector{<:GraphNode})
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end