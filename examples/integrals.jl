using Revise
using DeepONet
using Flux
using NumericalIntegration
using Plots
using Random

Random.seed!(1)

M, K, D = 80, 50, 1
A, B = 0, 1

points = range(A, stop = B, length = M)
means = [0]
params = collect(range(0.1, stop = 0.4, length = 5))
stds = collect(range(1, stop = 5, length = 4))

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

train_uxs, test_uxs = uxs_split(Us, xs, Ss; split = 0.75, toshuffle = true)
U_train, X_train, S_train = train_uxs
U_test, X_test, S_test = test_uxs

BATCH_SIZE = 256
train_loader = Flux.DataLoader((U_train, X_train, S_train), batchsize = BATCH_SIZE);
test_loader = Flux.DataLoader((U_test, X_test, S_test), batchsize = BATCH_SIZE);

model = DeepONetModel(M, 1, 20, [gelu, tanh],
	branch_sizes = [40],
	trunk_sizes = [40, 40],
	output_sizes = [4],
)
opt_state = Flux.setup(Flux.Adam(0.0003), model)
train_losses, test_losses = train!(model, opt_state, train_loader, test_loader)
plot(train_losses; yaxis = "loss", label = "train", lw = 2)
plot!(test_losses; yaxis = "loss", label = "test", lw = 2)

fx = U[:, 100]
Fx = S[:, 100]
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
