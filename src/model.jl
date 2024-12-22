struct ParallelDense{T, F}
	W::Matrix{T}
	b::Vector{T}
	act::Vector{F}
end

function ParallelDense(in_dim::Int, out_dim::Int, act::Vector{<:Function})
	W = Flux.glorot_normal(out_dim, in_dim)
	b = Flux.glorot_normal(out_dim)
	return ParallelDense{Float32, Function}(W, b, act)
end

function (dense::ParallelDense)(x)
	pre = dense.W * x .+ dense.b
	act = [f.(pre) for f in dense.act]
	return cat(act..., dims = 1)
end


struct OperatorNet
	layers::Chain
end

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

function (opnet::OperatorNet)(x)
	return opnet.layers(x)
end


struct Model
	branch::OperatorNet
	trunk::OperatorNet
	project::OperatorNet
end

function Model(
	branch_dim::Int, trunk_dim::Int, output_dim::Int, act::Array{<:Function};
	branch_sizes::Vector{Int}, trunk_sizes::Vector{Int}, output_sizes::Vector{Int},
)
	branch = OperatorNet(branch_dim, output_dim, act, branch_sizes)
	trunk = OperatorNet(trunk_dim, output_dim, act, trunk_sizes)
	project = OperatorNet(output_dim, 1, [identity], output_sizes)
	return Model(branch, trunk, project)
end

function (model::Model)(u, x)
	branch = model.branch(u)
	trunk = model.trunk(x)
	output = model.project(branch .* trunk)
	return output
end
