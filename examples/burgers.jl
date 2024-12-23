#!/usr/bin/env -S julia --project=examples

using DeepONet
using Flux
using NPZ
using Plots

burgers = npzread("examples/data/npz/burgers_equation.npz")

# Data preparation

X_train = PermutedDimsArray(burgers["x_train"], (2, 1))
y_train = PermutedDimsArray(burgers["y_train"], (3, 2, 1))
X_test = PermutedDimsArray(burgers["x_test"], (2, 1))
y_test = PermutedDimsArray(burgers["y_test"], (3, 2, 1))

@info """Data shapes
X_train $(size(X_train))
y_train $(size(y_train))
X_test  $(size(X_test))
y_test  $(size(y_test))
"""

X = cat(X_train, X_test, dims = 2)
y = cat(y_train, y_test, dims = 3)
M, T, N = size(y)

@info "M=$(M), T=$(T), N=$(N)"

Us = zeros(Float32, (M, N * T * M))
Xs = zeros(Float32, (2, N * T * M))
Ss = zeros(Float32, (1, N * T * M))

for n in 1:N, t in 1:T
	ni = (n - 1) * M * T
	ti = (t - 1) * M
	a = ni + ti + 1
	b = ni + ti + M

	Us[:, a:b] .= X[:, n]
	Xs[1, a:b] .= collect(1:M)
	Xs[2, a:b] .= t
	Ss[:, a:b] = y[:, t, n]
end

# Data splitting

uxs_train, uxs_test = uxs_split(Us, Xs, Ss; split = 0.6, toshuffle = true)
U_train, x_train, S_train = uxs_train
U_test, x_test, S_test = uxs_test

@info """Dataset sizes
Train: $(size(U_train, 2))
Test:  $(size(U_test, 2))
"""

train_bs = 256
test_bs = 1024
train_loader = Flux.DataLoader((U_train, x_train, S_train), batchsize = train_bs)
test_loader = Flux.DataLoader((U_test, x_test, S_test), batchsize = test_bs)

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
	M, 2, 20, activations;
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
savefig(p, "docs/assets/burgers_losses.png")

@info "Loss plot saved"

# Plot predictions

C = 12
y_hat = zeros(Float32, (M, T, C))
for c in 1:C, t in 1:T, m in 1:M
	x = Float32.(X_test[:, c])
	y_hat[m, t, c] = model(x, [m, t])[1]
end
anim = @animate for t in 1:T+4
	(t > T) && (t = T)
	p = plot(
		layout = C,
		dpi = 300,
		size = (700, 400),
		legend = false,
		legs = :bottom,
		legendfontsize = 10,
		tickfontsize = 1,
		xtickfontcolor = :white,
		ytickfontcolor = :white,
		foreground_color_legend = nothing,
	)
	annotate!(p[1], 0.2, 0.15, text("t = $(t)", :black, :left, 12))

	plot!(p[1], zeros(0) .* 1.2, color = :dimgrey, lw = 2, linestyle = :dot, label = " Input", legend = true, axis = ([], false))
	plot!(p[1], zeros(0), color = :dodgerblue2, lw = 2, label = " Ground Truth", legend = true, axis = ([], false))
	plot!(p[1], zeros(0), color = :orangered, lw = 2, label = " Prediction", legend = true, axis = ([], false))
	for c in 2:C
		plot!(p[c], X_test[:, c] .* 1.2, color = :dimgrey, lw = 1, linestyle = :dot)
		plot!(p[c], y_test[:, t, c], color = :dodgerblue2, lw = 1.2)
		plot!(p[c], y_hat[:, t, c], color = :orangered, lw = 1.2)
	end
end
gif(anim, "docs/assets/burgers_predictions.gif", fps = 10)

@info "Predictions plot saved"
