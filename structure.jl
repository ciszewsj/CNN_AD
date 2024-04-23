abstract type GraphNode end
abstract type Operator <: GraphNode end

struct Constant{T} <: GraphNode
    output :: T
end

mutable struct Variable <: GraphNode
    output :: Any
    gradient :: Any
end

mutable struct ScalarOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    function ScalarOperator(fun, inputs...)
		return new{typeof(fun)}(inputs, nothing, nothing)
	end
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    function BroadcastedOperator(fun, inputs...)
       return new{typeof(fun)}(inputs, nothing, nothing) 
    end
end


struct CNN
    wd1::Variable,
    wb1::Variable,
    wd2::Variable,
    wb2::Variable,
end
