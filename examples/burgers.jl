burgers = npzread("data/npz/burgers_equation.npz")

X_train, y_train = burgers["x_train"], burgers["y_train"]
X_test, y_test = burgers["x_test"], burgers["y_test"]

X_train = PermutedDimsArray(X_train, (2, 1))
y_train = PermutedDimsArray(y_train, (3, 2, 1))
X_test = PermutedDimsArray(X_test, (2, 1))
y_test = PermutedDimsArray(y_test, (3, 2, 1))

X = cat(X_train, X_test, dims = 2)
y = cat(y_train, y_test, dims = 3)
M, T, N = size(y)

Us = zeros(Float32, (M, N * T * M))
Xs = zeros(Float32, (2, N * T * M))
Ss = zeros(Float32, (1, N * T * M))
for n in 1:N, t in 1:T
	ni = (n - 1) * M * T
	ti = (t - 1) * M
	a = ni + ti + 1
	b = ni + ti + M

	Us[:, a:b] .= X[:, n]
	Ss[:, a:b] = y[:, t, n]
	Xs[1, a:b] .= collect(1:M)
	Xs[2, a:b] .= t
end

plot(y_train[:, :, 1])

uxs_train, uxs_test = uxs_split(Us, Xs, Ss; split = 0.5, toshuffle = true)
U_train, x_train, S_train = uxs_train
U_test, x_test, S_test = uxs_test

BATCH_SIZE = 1024
train_loader = Flux.DataLoader((U_train, x_train, S_train), batchsize = BATCH_SIZE);
test_loader = Flux.DataLoader((U_test, x_test, S_test), batchsize = BATCH_SIZE);

model = Model(size(U_train, 1), 2, 32, [gelu, tanh],
	branch_sizes = [32 for _ in 1:5],
	trunk_sizes = [32 for _ in 1:5],
	output_sizes = [1],
)
opt_state = Flux.setup(Flux.AdamW(0.0003), model)
train_losses, test_losses = train!(model, train_loader, test_loader)
plot(train_losses; yaxis = "loss", label = "train", lw = 2)
plot!(test_losses; yaxis = "loss", label = "test", lw = 2)


using CairoMakie

i = 0
preds = []
t = 13
for i in 1:16
	x = X_test[:, i]
	y = zeros(M)
	for pt in 1:M
		y[pt] = model(x, [pt, t])[1]
	end
	push!(preds, (y, y_test[:, t, i]))
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
