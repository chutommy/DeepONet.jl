using GaussianRandomFields
using Plots
using NumericalIntegration
using Flux
using Random
using ProgressMeter

















# model.jl

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


struct DeepONet
	branch::OperatorNet
	trunk::OperatorNet
	project::OperatorNet
end

function DeepONet(
	branch_dim::Int, trunk_dim::Int, output_dim::Int, act::Array{<:Function};
	branch_sizes::Vector{Int}, trunk_sizes::Vector{Int}, output_sizes::Vector{Int},
)
	branch = OperatorNet(branch_dim, output_dim, act, branch_sizes)
	trunk = OperatorNet(trunk_dim, output_dim, act, trunk_sizes)
	project = OperatorNet(output_dim, 1, [identity], output_sizes)
	return DeepONet(branch, trunk, project)
end

function (model::DeepONet)(u, x)
	branch = model.branch(u)
	trunk = model.trunk(x)
	output = model.project(branch .* trunk)
	return output
end



















# random_field.jl

struct RandomFieldGenerator
	size::Int
	mean::Vector
	cov::AbstractCovarianceFunction
	generator::GaussianRandomFieldGenerator
	grf::GaussianRandomField
end

function RandomFieldGenerator(
	points::AbstractVector, dim::Int, size::Int;
	mean::Real, std::Real, param::Real,
)
	model = Gaussian(param, σ = std)
	means = fill(mean, size)
	covf = CovarianceFunction(dim, model)
	generator = CirculantEmbedding()
	grf = GaussianRandomField(means, covf, generator, points)
	return RandomFieldGenerator(length(points), means, covf, generator, grf)
end

function (generator::RandomFieldGenerator)()
	sample(generator.grf)
end
function (generator::RandomFieldGenerator)(n::Int)
	fields = zeros(Float32, (generator.size, n))
	for i in 1:n
		fields[:, i] = generator()
	end
	return fields
end

function generate_random_fields(
	points::AbstractVector, dim::Int, size::Int;
	means::Vector, stds::Vector, params::Vector,
	K::Int,
)
	generators = [RandomFieldGenerator(points, dim, size; mean = m, std = s, param = p)
				  for p in params, s in stds, m in means]
	gsize = length(generators)
	total = gsize * K
	out = zeros(Float32, (size, total))
	for g in 1:gsize
		gi = (g - 1) * K
		a, b = gi + 1, gi + K
		out[:, a:b] = generators[g](K)
	end
	return out
end














# train.jl

function evaluate(model::DeepONet, loader::Flux.DataLoader; loss_fn = Flux.Losses.mse)
	return sum(loss_fn(model(u, x), s) for (u, x, s) in loader)
end

function train!(
	model::DeepONet,
	train_loader::Flux.DataLoader,
	test_loader::Flux.DataLoader;
	epochs = 20, loss_fn = Flux.Losses.mse,
)
	train_losses = zeros(Float32, epochs)
	test_losses = zeros(Float32, epochs)
	@showprogress for e in 1:epochs
		for (u, pts, s) in train_loader
			loss, grads = Flux.withgradient(model) do m
				loss_fn(m(u, pts), s)
			end
			Flux.update!(opt_state, model, grads[1])
			train_losses[e] += loss
		end
		test_losses[e] = evaluate(model, test_loader, loss_fn = loss_fn)
	end
	return train_losses, test_losses
end


















# integrals

EPOCHS = 30
M, K, D = 100, 60, 1
A, B = 0, 1

points = range(A, stop = B, length = M)
means = [0]
params = collect(range(0.1, stop = 0.3, length = 5))
stds = collect(range(1, stop = 5, length = 3))

U = generate_random_fields(points, D, M; means = means, stds = stds, params = params, K = K)
S = cumul_integrate(points, U', dims = 1)
N = size(U, 2)

show = 30
plot(points, U[:, 1:show], label = "", lw = 2)
heatmap(U[:, 1:show]')
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

test_train_split = 0.8
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
train_loader = Flux.DataLoader((U_train, x_train, S_train), batchsize = BATCH_SIZE);
test_loader = Flux.DataLoader((U_test, x_test, S_test), batchsize = BATCH_SIZE);

model = DeepONet(M, 1, 32, [gelu, tanh],
	branch_sizes = [32, 32],
	trunk_sizes = [64, 64, 64],
	output_sizes = [4],
)
opt_state = Flux.setup(Flux.Adam(0.0003), model)
train_losses, test_losses = train!(model, train_loader, test_loader; epochs = EPOCHS)

plot(train_losses; yaxis = "loss", label = "train")
plot!(test_losses; yaxis = "loss", label = "test")

fx = U[:, 293]
Fx = S[:, 293]
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
using NPZ

burgers = npzread("data/npz/burgers_equation.npz")

x_train = burgers["x_train"]
y_train = burgers["y_train"]

x_train = PermutedDimsArray(x_train, (2, 1))
y_train = PermutedDimsArray(y_train, (3, 2, 1))
M, T, N = size(y_train)

plot(y_train[:, :, 1])

X = zeros(Float32, (M, N * T * M))
Y = zeros(Float32, (1, N * T * M))
g = zeros(Float32, (2, N * T * M))

for n in 1:N, t in 1:T
	ni = (n - 1) * M * T
	ti = (t - 1) * M
	a = ni + ti + 1
	b = ni + ti + M

	X[:, a:b] .= x_train[:, n]
	Y[:, a:b] = y_train[:, t, n]

	g[1, a:b] .= collect(1:M)
	g[2, a:b] .= t
end


BATCH_SIZE = 1024
train_loader = Flux.DataLoader((X, g, Y), batchsize = BATCH_SIZE, shuffle = false);


model = DeepONet(size(x_data, 1), 2, 32, [gelu, tanh],
	branch_sizes = [ntuple(Returns(32), 5)...],
	trunk_sizes = [ntuple(Returns(32), 5)...],
	output_sizes = [1],
)
opt_state = Flux.setup(Flux.AdamW(0.0003), model)

EPOCHS = 30
train_losses = []
@showprogress for epoch in 1:EPOCHS
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
end


plot(train_losses; yaxis = "loss", label = "train")


using CairoMakie

i = 0
preds = []
t = 10
for i in 1:16
	x = x_train[:, i]
	y = zeros(M)
	for pt in 1:M
		y[pt] = model(x, [pt, t])[1]
	end
	push!(preds, (y, y_train[:, t, i]))
end

begin
	fig = Figure(; size = (1024, 1024))

	axs = [Axis(fig[i, j]) for i in 1:4, j in 1:4]
	for i in 1:4, j in 1:4
		idx = i + (j - 1) * 4
		ax = axs[i, j]
		l1 = lines!(ax, vec(preds[idx][1]))
		l2 = lines!(ax, vec(preds[idx][2]))

		i == 4 && (ax.xlabel = "x")
		j == 1 && (ax.ylabel = "u(x)")

		if i == 1 && j == 1
			axislegend(ax, [l1, l2], ["Predictions", "Ground Truth"])
		end
	end
	linkaxes!(axs...)

	fig[0, :] = Label(fig, "Burgers Equation using DeepONet"; tellwidth = false, font = :bold)

	fig
end






# darcy's flow

using NPZ

darcy = npzread("data/npz/darcys_flow.npz")

x_train = darcy["x_train"]
y_train = darcy["y_train"]

x_train = PermutedDimsArray(x_train, (2, 3, 1))
y_train = PermutedDimsArray(y_train, (2, 3, 1))
a, b, N = size(y_train)
a != b && error("a != b")
M = a

heatmap(x_train[:, :, 1])
heatmap(y_train[:, :, 1])

X = zeros(Float32, (M * M, N * M * M))
Y = zeros(Float32, (1, N * M * M))
g = zeros(Float32, (2, N * M * M))

matrix_embd = [i for i in 1:M, j in 1:M]

for n in 1:N
	ni = (n - 1) * M * M
	a = ni + 1
	b = ni + M * M

	X[:, a:b] .= x_train[:, :, n][:]
	Y[:, a:b] = y_train[:, :, n][:]

	g[1, a:b] = matrix_embd[:]
	g[2, a:b] = matrix_embd'[:]
end

BATCH_SIZE = 1024
train_loader = Flux.DataLoader((X, g, Y), batchsize = BATCH_SIZE, shuffle = false);

model = DeepONet(M * M, 2, 32, [gelu, tanh],
	branch_sizes = [ntuple(Returns(32), 5)...],
	trunk_sizes = [ntuple(Returns(32), 5)...],
	output_sizes = [1],
)
opt_state = Flux.setup(Flux.AdamW(0.0003), model)

EPOCHS = 30
train_losses = []
@showprogress for epoch in 1:EPOCHS
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
end


plot(train_losses; yaxis = "loss", label = "train")

i = 0
preds = []
for i in 1:4
	x = x_train[:, :, i]
	y = zeros(M, M)
	for i in 1:M, j in 1:M
		y[i, j] = model(x[:], [i, j])[1]
	end
	push!(preds, (x, y, y_train[:, :, i]))
end

i = 3
heatmap(preds[i][1])
heatmap(preds[i][2])
heatmap(preds[i][3])
