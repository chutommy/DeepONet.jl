using GaussianRandomFields
using Plots
using NumericalIntegration
using Flux
# using Statistics
using ProgressMeter
# using LinearAlgebra
# using ArgCheck
# using ConcreteStructs
# using LuxCore











struct ParallelDense
	W::AbstractMatrix
	b::AbstractVector
	act::Array{Function}
end

function ParallelDense(in::Int, out::Int, act::Array{<:Function})
	ParallelDense(Flux.glorot_normal(out, in), zeros(Float32, out), act)
end

function (d::ParallelDense)(x)
	preactivation = d.W * x .+ d.b
	activation = [act(preactivation) for act in d.act]
	return cat(activation..., dims = 1)
end

model = let neurons = 40, in1 = M, in2 = 1, output_neurons = 20
	act = [gelu, tanh, identity]
	actsize = length(act)
	branch1 = ParallelDense(in1, neurons, act)
	branch2 = ParallelDense(actsize*neurons, neurons, act)
	branch3 = ParallelDense(actsize*neurons, output_neurons, act)

	trunk1 = ParallelDense(in2, neurons, act)
	trunk2 = ParallelDense(actsize*neurons, neurons, act)
	trunk3 = ParallelDense(actsize*neurons, neurons, act)
	trunk4 = ParallelDense(actsize*neurons, output_neurons, act)

	adjoint = Dense(actsize * output_neurons => 1, [identity])

	function fwd(u, x)
		branch = branch3(branch2(branch1(u)))
		trunk = trunk4(trunk3(trunk2(trunk1(x))))
		output = adjoint(branch .* trunk)
		return output
	end
end















EPOCHS = 30
M = 100
N = 110
D = 1
A, B = 0, 1

μ = 0.0
λ = 0.3
σ = 3

gauss = Gaussian(λ, σ = σ)
covf = CovarianceFunction(D, gauss)
pts = range(A, stop = B, length = M)
mean = fill(μ, (M));
generator = CirculantEmbedding()
grf = GaussianRandomField(mean, covf, generator, pts)

U = zeros(Float32, (M, N))
for i in 1:N
	U[:, i] = sample(grf)
end
show = 20
plot(pts, U[:, 1:show], label = "")
heatmap(U[:, 1:show]')

S = cumul_integrate(pts, U', dims = 1)
heatmap(S[:, 1:show]')

i = 10
plot(pts, U[:, i], label = "f(pts)", lw = 2)
plot!(pts, S[:, i], label = "F(pts)", lw = 2)

Us = zeros(Float32, (M, M * N))
xs = zeros(Float32, (1, M * N))
Ss = zeros(Float32, (1, M * N))

for i in 1:N
	a = (i - 1) * M + 1
	b = i * M
	Us[:, a:b] .= U[:, i]
	xs[:, a:b] = collect(pts)[1:M]
	Ss[:, a:b] = S[:, i]
end

test_train_split = 0.99
n_train = Int(floor(test_train_split * M * N))

U_train = Us[:, 1:n_train]
x_train = xs[:, 1:n_train]
S_train = Ss[:, 1:n_train]

U_test = Us[:, 1+n_train:end]
x_test = xs[:, 1+n_train:end]
S_test = Ss[:, 1+n_train:end]

BATCH_SIZE = 512
train_loader = Flux.DataLoader((U_train, x_train, S_train), batchsize = BATCH_SIZE, shuffle = true);
test_loader = Flux.DataLoader((U_test, x_test, S_test), batchsize = BATCH_SIZE, shuffle = true);

for (u_, x_, s_) in train_loader
	println(size(u_), size(x_), size(s_))
	break
end

for (u_, x_, s_) in test_loader
	println(size(u_), size(x_), size(s_))
	break
end

# struct Model
# 	branch_net::Chain
# 	trunk_net::Chain
# 	joint::Dense
# end

# function create_layers(input_dim::Int, output_dim::Int, layer_sizes::Tuple{Vararg{Int}})
# 	layers = []
# 	in_features = input_dim
# 	for out_features in layer_sizes
# 		push!(layers, Dense(in_features => out_features, relu))
# 		in_features = out_features
# 	end
# 	push!(layers, Dense(in_features => output_dim))
# 	return layers
# end

# function (model::Model)(branch_input::AbstractArray, trunk_input::AbstractArray)
# 	branch_output = model.branch_net(branch_input)
# 	trunk_output = model.trunk_net(trunk_input)
# 	joint_output = model.joint(branch_output .* trunk_output)
# 	return joint_output
# end

# function DeepNet(
# 	branch_input_dim::Int,
# 	trunk_input_dim::Int,
# 	output_dim::Int,
# 	branch_layer_sizes::Tuple{Vararg{Int}},
# 	trunk_layer_sizes::Tuple{Vararg{Int}},
# )
# 	branch_layers = Chain(create_layers(branch_input_dim, output_dim, branch_layer_sizes))
# 	trunk_layers = Chain(create_layers(trunk_input_dim, output_dim, trunk_layer_sizes))
#     joint_layer = Dense(output_dim => 1)
# 	return Model(branch_layers, trunk_layers, joint_layer)
# end

# in1, in2, out12 = M, 1, 20
# branch1_sizes = (40,)
# branch2_sizes = (40, 40)
# model = DeepNet(in1, in2, out12, branch1_sizes, branch2_sizes)


opt_state = Flux.setup(Flux.Adam(0.003), model)

losses = []
@showprogress for epoch in 1:EPOCHS
	train_loss = 0
	for (u_, x_, s_) in train_loader
		loss, grads = Flux.withgradient(model) do m
			x_ = reshape(x_, (1, length(x_)))
			s_ = reshape(s_, (1, length(s_)))
			y_hat = m(u_, x_)
			Flux.Losses.mse(y_hat, s_)
		end
		Flux.update!(opt_state, model, grads[1])
		train_loss += loss
	end
	push!(losses, train_loss / BATCH_SIZE)
end

plot(losses; yaxis = "loss", label = "per batch")

# f(pts) = sin(-exp(-pts + 2)) * exp(-pts + 2)
# F(pts) = -cos(-exp(-pts + 2)) - -cos(-exp(2))
f(x) = -cos(3 * x) * 3
F(x) = -sin(3 * x)
fx = f.(pts)
Fx = F.(pts)
Gx = zeros(Float32, (M))
u_ = zeros(Float32, ((M, M)))
x_ = zeros(Float32, ((1, M)))
for i in 1:M
	u_[:, i] = fx
	x_[1, i] = pts[i]
end
Gx = model(u_, x_)
plot(pts, fx, label = "f", lw = 2)
plot!(pts, Fx, label = "F", lw = 2)
plot!(pts, Gx', label = "G", lw = 2)

