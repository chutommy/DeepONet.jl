"""
A dense layer with multiple activation functions.
"""
struct ParallelDense{T, F}
	W::Matrix{T}
	b::Vector{T}
	act::Vector{F}
end

"""
Constructs a `ParallelDense` layer.
"""
function ParallelDense(in_dim::Int, out_dim::Int, act::Vector{<:Function})
	W = Flux.glorot_normal(out_dim, in_dim)
	b = Flux.glorot_normal(out_dim)
	return ParallelDense{Float32, Function}(W, b, act)
end

"""
Applies the parallel dense layer to the input `x`.
"""
function (dense::ParallelDense)(x::AbstractArray)
	pre = dense.W * x .+ dense.b
	act = [f.(pre) for f in dense.act]
	return cat(act..., dims = 1)
end

"""
A neural network composed of a sequence of `ParallelDense` layers.
"""
struct OperatorNet
	layers::Chain
end

"""
	OperatorNet(in_dim::Int, out_dim::Int, act::Vector{<:Function}, sizes::Vector{Int})

Constructs an `OperatorNet`.

# Arguments
- `in_dim::Int`: Input dimension.
- `out_dim::Int`: Output dimension.
- `act::Vector{<:Function}`: Activation functions applied for each layer.
- `sizes::Vector{Int}`: Sizes of each layer in the network.

# Returns
- An `OperatorNet` instance.
"""
function OperatorNet(in_dim::Int, out_dim::Int, act::Vector{<:Function}, sizes::Vector{Int})
	layers = []
	from = in_dim
	for to in sizes
		push!(layers, ParallelDense(from, to, act))
		from = to * length(act)
	end
	push!(layers, Dense(from => out_dim, identity))
	return OperatorNet(Chain(layers))
end

"""
Applies the operator network to the input `x`.
"""
function (opnet::OperatorNet)(x::AbstractArray)
	return opnet.layers(x)
end

"""
	struct DeepONetModel

An extended DeepONet model (https://arxiv.org/abs/1910.03193). This version implementation
allows for flexible sizes of the branch, trunk, and projection, as well as multiple
activation functions.
"""
struct DeepONetModel
	branch::OperatorNet
	trunk::OperatorNet
	project::OperatorNet
end

"""
	DeepONetModel(
		branch_dim::Int, trunk_dim::Int, output_dim::Int, act::Array{<:Function};
		branch_sizes::Vector{Int}, trunk_sizes::Vector{Int}, output_sizes::Vector{Int}
	)

Constructs a `DeepONetModel`.

# Arguments
- `branch_dim::Int`: Input dimension of the branch network.
- `trunk_dim::Int`: Input dimension of the trunk network.
- `output_dim::Int`: Shared output dimension of the branch and trunk.
- `act::Array{<:Function}`: Activation functions for all networks.

# Keywords
- `branch_sizes::Vector{Int}`: Sizes of the branch network layers.
- `trunk_sizes::Vector{Int}`: Sizes of the trunk network layers.
- `output_sizes::Vector{Int}`: Sizes of the projection network layers.

# Returns
- A `DeepONetModel` instance.
"""
function DeepONetModel(
	branch_dim::Int, trunk_dim::Int, output_dim::Int, act::Array{<:Function};
	branch_sizes::Vector{Int}, trunk_sizes::Vector{Int}, output_sizes::Vector{Int},
)
	branch = OperatorNet(branch_dim, output_dim, act, branch_sizes)
	trunk = OperatorNet(trunk_dim, output_dim, act, trunk_sizes)
	project = OperatorNet(output_dim, 1, [identity], output_sizes)
	return DeepONetModel(branch, trunk, project)
end

"""
	(model::DeepONetModel)(u, x)

Applies the model to the inputs `u` and `x`.

# Arguments
- `u::AbstractArray`: Input to the branch network.
- `x::AbstractArray`: Input to the trunk network.

# Returns
- The combined output of the model.
"""
function (model::DeepONetModel)(u::AbstractArray, x::AbstractArray)
	branch = model.branch(u)
	trunk = model.trunk(x)
	output = model.project(branch .* trunk)
	return output
end
