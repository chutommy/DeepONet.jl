using DeepONet
using Flux
using NPZ
using Plots

darcys = npzread("examples/data/npz/darcys_flow.npz")

X_train, y_train = darcys["x_train"], darcys["y_train"]
X_test, y_test = darcys["x_test"], darcys["y_test"]

X_train = PermutedDimsArray(X_train, (2, 3, 1))
y_train = PermutedDimsArray(y_train, (2, 3, 1))
X_test = PermutedDimsArray(X_test, (2, 3, 1))
y_test = PermutedDimsArray(y_test, (2, 3, 1))

heatmap(X_train[:, :, 1])
heatmap(y_train[:, :, 1])

X = cat(X_train, X_test, dims = 3)
y = cat(y_train, y_test, dims = 3)
M, M2, N = size(X)
@assert M == M2

Us = zeros(Float32, (M * M, N * M * M))
Xs = zeros(Float32, (2, N * M * M))
Ss = zeros(Float32, (1, N * M * M))
matrix_embd = [i for i in 1:M, j in 1:M]
for n in 1:N
	ni = (n - 1) * M * M
	a = ni + 1
	b = ni + M * M

	Us[:, a:b] .= X[:, :, n][:]
	Ss[:, a:b] = y[:, :, n][:]
	Xs[1, a:b] = matrix_embd[:]
	Xs[2, a:b] = matrix_embd'[:]
end

uxs_train, uxs_test = uxs_split(Us, Xs, Ss; split = 0.5, toshuffle = true)
U_train, x_train, S_train = uxs_train
U_test, x_test, S_test = uxs_test

BATCH_SIZE = 256
train_loader = Flux.DataLoader((U_train, x_train, S_train), batchsize = BATCH_SIZE);
test_loader = Flux.DataLoader((U_test, x_test, S_test), batchsize = BATCH_SIZE);

model = Model(M * M, 2, 20, [gelu, tanh],
	branch_sizes = [30, 30],
	trunk_sizes = [30, 30],
	output_sizes = [1],
)
opt_state = Flux.setup(Flux.AdamW(0.0003), model)
train_losses, test_losses = train!(model, opt_state, train_loader, test_loader)
plot(train_losses; yaxis = "loss", label = "train", lw = 2)
plot!(test_losses; yaxis = "loss", label = "test", lw = 2)

i = 0
preds = []
for i in 1:4
	x = X_train[:, :, i]
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
