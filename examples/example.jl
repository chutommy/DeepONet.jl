using GaussianRandomFields
using Plots
using NumericalIntegration
using Flux
using Random
# using Statistics
using ProgressMeter
# using LinearAlgebra
# using ArgCheck
# using ConcreteStructs










```
ParallelDense
```

struct ParallelDense
	W::AbstractMatrix
	b::AbstractVector
	act::Array{Function}
end

function ParallelDense(in_dim::Int, out_dim::Int, act::Array{<:Function})
	W = Flux.glorot_normal(out_dim, in_dim)
	b = Flux.glorot_normal(out_dim)
	return ParallelDense(W, b, act)
end

function (dense::ParallelDense)(x)
	pre = dense.W * x .+ dense.b
	act = [act.(pre) for act in dense.act]
	return cat(act..., dims = 1)
end


```
OperatorNet
```

struct OperatorNet
	layers::Chain
end

function OperatorNet(
	in_dim::Int, out_dim::Int,
	act::Array{<:Function}, sizes::Array{Int},
)
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


```
DeepONet
```

struct DeepONet
	branch::OperatorNet
	trunk::OperatorNet
	fuser::OperatorNet
end

function DeepONet(
	branch_dim::Int, trunk_dim::Int, output_dim::Int, act::Array{<:Function};
	branch_sizes::Array{Int}, trunk_sizes::Array{Int}, output_sizes::Array{Int},
)
	branch = OperatorNet(branch_dim, output_dim, act, branch_sizes)
	trunk = OperatorNet(trunk_dim, output_dim, act, trunk_sizes)
	fuser = OperatorNet(output_dim, 1, [identity], output_sizes)
	return DeepONet(branch, trunk, fuser)
end

function (model::DeepONet)(u, x)
	branch = model.branch(u)
	trunk = model.trunk(x)
	output = model.fuser(branch .* trunk)
	return output
end









```
RandomField
```

struct RandomFieldGenerator
	mean::AbstractVector
	cov::AbstractCovarianceFunction
	generator::GaussianRandomFieldGenerator
	grf::GaussianRandomField
end

function RandomFieldGenerator(
	points::AbstractVector, dim::Int, size::Int;
	μ::Float64, σ::Float64, λ::Float64,
)
	model = Gaussian(λ, σ = σ)
	mean = fill(μ, (size))
	covf = CovarianceFunction(dim, model)
	generator = CirculantEmbedding()
	grf = GaussianRandomField(mean, covf, generator, points)
	return RandomFieldGenerator(mean, covf, generator, grf)
end

(generator::RandomFieldGenerator)() = sample(generator.grf)






# integrals

EPOCHS = 10
M = 200
K = 10
D = 1
A, B = 0, 1

μ = 0.0
λs = range(0.1, stop = 0.5, length = 10)
σs = range(1.0, stop = 10, length = 5)

points = range(A, stop = B, length = M)
generators = [
	RandomFieldGenerator(points, D, M; μ = 0.0, σ = σ, λ = λ)
	for λ in λs, σ in σs
]

G = length(generators)
N = G * K
U = zeros(Float32, (M, N))
for i in 1:K, j in 1:G
	id = (i - 1) * G + j
	U[:, id] = generators[j]()
end
show = 30
plot(points, U[:, 1:show], label = "", lw = 2)
heatmap(U[:, 1:show]')

S = cumul_integrate(points, U', dims = 1)
heatmap(S[:, 1:show]')

i = 3
plot(points, U[:, i], label = "f(points)", lw = 2)
plot!(points, S[:, i], label = "F(points)", lw = 2)

Us = zeros(Float32, (M, M * N))
xs = zeros(Float32, (1, M * N))
Ss = zeros(Float32, (1, M * N))

for i in 1:N
	a = (i - 1) * M + 1
	b = i * M
	Us[:, a:b] .= U[:, i]
	xs[:, a:b] = collect(points)[1:M]
	Ss[:, a:b] = S[:, i]
end

n_train = Int(floor(test_train_split * M * N))
shuffle_view = shuffle(1:M*N)
train_view = shuffle_view[1:n_train]
test_view = shuffle_view[n_train+1:end]

U_train = Us[:, train_view]
x_train = xs[:, train_view]
S_train = Ss[:, train_view]

U_test = Us[:, test_view]
x_test = xs[:, test_view]
S_test = Ss[:, test_view]

BATCH_SIZE = 1024
train_loader = Flux.DataLoader((U_train, x_train, S_train), batchsize = BATCH_SIZE, shuffle = true);
test_loader = Flux.DataLoader((U_test, x_test, S_test), batchsize = BATCH_SIZE, shuffle = true);



model = DeepONet(M, 1, 32, [gelu],
	branch_sizes = [64, 64, 64],
	trunk_sizes = [128, 128, 128, 128],
	output_sizes = [8, 8],
)
opt_state = Flux.setup(Flux.Adam(0.0003), model)

function evaluate(model, loader)
	loss = 0
	for (u, pts, s) in loader
		s_hat = model(u, pts)
		loss += Flux.Losses.mse(s_hat, s)
	end
	return loss
end

train_losses = []
test_losses = []
@showprogress for epoch in 1:EPOCHS
	test_loss = evaluate(model, test_loader)
	train_loss = 0
	for (u, pts, s) in train_loader
		loss, grads = Flux.withgradient(model) do m
			s_hat = m(u, pts)
			Flux.Losses.mse(s_hat, s)
		end
		Flux.update!(opt_state, model, grads[1])
		train_loss += loss
	end
	push!(train_losses, train_loss / BATCH_SIZE)
	push!(test_losses, test_loss / BATCH_SIZE)
end

plot(train_losses; yaxis = "loss", label = "train")
plot!(test_losses; yaxis = "loss", label = "test")

# f(points) = sin(-exp(-points + 2)) * exp(-points + 2)
# F(points) = -cos(-exp(-points + 2)) - -cos(-exp(2))
# f(x) = -cos(3 * x) * 3
# F(x) = -sin(3 * x)
# a = 17
# f(x) = cos(a * x) * a
# F(x) = sin(a * x)
# fx = f.(points)
# Fx = F.(points)

fx = U[:, 203]
Fx = S[:, 203]

Gx = zeros(Float32, (M))
u_ = zeros(Float32, ((M, M)))
x_ = zeros(Float32, ((1, M)))
for i in 1:M
	u_[:, i] = fx
	x_[1, i] = points[i]
end
Gx = model(u_, x_)
plot(points, fx, label = "f", lw = 2)
plot(points, Fx, label = "F", lw = 2)
plot!(points, Gx', label = "G", lw = 2)





# burger's equation
