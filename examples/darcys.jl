#!/usr/bin/env -S julia --project=examples

using DeepONet
using Flux
using NPZ
using Plots

darcys = npzread("examples/data/npz/darcys_flow.npz")

# Data preparation

X_train = PermutedDimsArray(darcys["x_train"], (2, 3, 1))
y_train = PermutedDimsArray(darcys["y_train"], (2, 3, 1))
X_test = PermutedDimsArray(darcys["x_test"], (2, 3, 1))
y_test = PermutedDimsArray(darcys["y_test"], (2, 3, 1))

@info """Data shapes
X_train $(size(X_train))
y_train $(size(y_train))
X_test  $(size(X_test))
y_test  $(size(y_test))
"""

X = cat(X_train, X_test, dims = 3)
y = cat(y_train, y_test, dims = 3)
M, M2, N = size(X)
@assert M == M2

@info "M=$(M), N=$(N)"

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

# Data splitting

uxs_train, uxs_test = uxs_split(Us, Xs, Ss; split = 0.5, toshuffle = true)
U_train, x_train, S_train = uxs_train
U_test, x_test, S_test = uxs_test

@info """Dataset sizes
Train: $(size(U_train, 2))
Test:  $(size(U_test, 2))
"""

train_bs = 256
test_bs = 1024
train_loader = Flux.DataLoader((U_train, x_train, S_train), batchsize = train_bs);
test_loader = Flux.DataLoader((U_test, x_test, S_test), batchsize = test_bs);

@info """
Batch sizes
Train: $(length(train_loader))
Test:  $(length(test_loader))
"""

# Model training

activations = [gelu, tanh]
branch_sizes = [30, 30]
trunk_sizes = [30, 30]
output_sizes = [1]
model = DeepONetModel(
	M * M, 2, 20, activations;
	branch_sizes = branch_sizes,
	trunk_sizes = trunk_sizes,
	output_sizes = output_sizes,
)

@info """
Model summary
Activations:  $(activations)
Branch_sizes: $(branch_sizes)
Trunk_sizes:  $(trunk_sizes)
Output_sizes: $(output_sizes)
"""

@info "Training model"

opt_state = Flux.setup(Flux.AdamW(0.0003), model)
train_losses, test_losses = train!(model, opt_state, train_loader, test_loader)
train_losses ./= train_bs
test_losses ./= test_bs

@info "Training complete"

# Plot training and test losses

p = plot(xtickfont = font(8), ytickfont = font(8), dpi = 300)
plot!(p, train_losses, label = "Train Loss", lw = 2)
plot!(p, test_losses, label = "Test Loss", lw = 2)
savefig(p, "docs/assets/darcys_losses.png")

@info "Loss plot saved"

# Plot darcys

C = 16
p = plot(
	layout = C,
	dpi = 300,
	size = (850, 800),
	axis = false,
	legend = false,
	tickfontsize = 1,
	guidefontsize = 12,
	xtickfontcolor = :white,
	ytickfontcolor = :white,
)

for c in 1:4
	x = X_test[:, :, c]
	y_hat = zeros(M, M)
	for i in 1:M, j in 1:M
		y_hat[i, j] = model(x[:], [i, j])[1]
	end

	if c == 1
		heatmap!(p, X_test[:, :, c], subplot = c, ylabel = "Input", color = :amp)
		heatmap!(p, y_test[:, :, c], subplot = c + 4, ylabel = "Ground Truth")
		heatmap!(p, y_hat, subplot = c + 8, ylabel = "Prediction")
		heatmap!(p, abs.(y_test[:, :, c] - y_hat), subplot = c + 12, ylabel = "Abs. Error",
			clim = (0, maximum(abs.(y_test[:, :, c]))), color = :amp)
		continue
	end

	heatmap!(p, X_test[:, :, c], subplot = c, color = :amp)
	heatmap!(p, y_test[:, :, c], subplot = c + 4)
	heatmap!(p, y_hat, subplot = c + 8)
	heatmap!(p, abs.(y_test[:, :, c] - y_hat), subplot = c + 12,
		clim = (0, maximum(abs.(y_test[:, :, c]))), color = :amp)
end
savefig(p, "docs/assets/darcys_predictions.png")

@info "Darcys predictions plot saved"
