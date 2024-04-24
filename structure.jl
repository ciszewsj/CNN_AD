abstract type GraphNode end
abstract type Operator <: GraphNode end

struct Constant{T} <: GraphNode
    output :: T
end

mutable struct Variable <: GraphNode
    output :: Any
    gradient :: Any
    name::String
    Variable(output; name = "?") = new(output, nothing, name)
end

mutable struct ScalarOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name::String
    function ScalarOperator(fun, inputs...; name = "?")
		return new{typeof(fun)}(inputs, nothing, nothing, name)
	end
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name::String
    cache::Any
    function BroadcastedOperator(fun, inputs...; name = "?")
       return new{typeof(fun)}(inputs, nothing, nothing, name, nothing) 
    end
end


struct CNN
    wk1
    wd1
    wb1
    wd2
    wb2
end