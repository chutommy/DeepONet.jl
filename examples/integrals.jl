#!/usr/bin/env -S julia --project=examples

using DeepONet
using Flux
using NumericalIntegration
using Plots
using Random

# Data preparation

M, K, D = 80, 50, 1
A, B = 0, 1

points = range(A, stop = B, length = M)
params = collect(range(0.1, stop = 0.4, length = 5))
stds = collect(range(1, stop = 5, length = 4))
means = [0]

X = generate_random_fields(points, D, M; means = means, stds = stds, params = params, K = K)
Y = cumul_integrate(points, X', dims = 1)
N = size(X, 2)

@info """Data shapes
X $(size(X))
y $(size(Y))
"""

@info "M=$(M), N=$(N), K=$(K)"

Us = zeros(Float32, (M, M * N))
xs = zeros(Float32, (1, M * N))
Ss = zeros(Float32, (1, M * N))

for i in 1:N
	a = (i - 1) * M + 1
	b = i * M
	Us[:, a:b] .= X[:, i]
	xs[:, a:b] = collect(points)[1:M]
	Ss[:, a:b] = Y[:, i]
end

# Data splitting

train_uxs, test_uxs = uxs_split(Us, xs, Ss; split = 0.75, toshuffle = true)
U_train, x_train, S_train = train_uxs
U_test, x_test, S_test = test_uxs

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
branch_sizes = [30]
trunk_sizes = [30, 30]
output_sizes = [1]
model = DeepONetModel(
	M, 1, 20, activations;
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
savefig(p, "assets/integrals_losses.png")

@info "Loss plot saved"

# Plot predictions

C = 12
p = plot(
	layout = C,
	dpi = 300,
	size = (800, 500),
	legend = false,
	legendfontsize = 12,
	tickfontsize = 1,
	guidefontsize = 12,
	xtickfontcolor = :white,
	ytickfontcolor = :white,
	foreground_color_legend = nothing,
)

for c in 1:C
	fx = X[:, c]
	Fx = Y[:, c]
	Gx = zeros(Float32, (M))
	u_ = zeros(Float32, ((M, M)))
	x_ = zeros(Float32, ((1, M)))
	for i in 1:M
		u_[:, i] = fx
		x_[1, i] = points[i]
	end
	Gx = model(u_, x_)

	if c == 1
		plot!(p[c], zeros(0), linestyle = :dash, label = " Input", axis = ([], false))
		plot!(p[c], zeros(0), color = :grays, label = " Ground Truth", axis = ([], false))
		plot!(p[c], zeros(0), color = :orangered, label = " Prediction", axis = ([], false),
			legend = :inside)
		continue
	end

	r = maximum(abs.(Fx)) / maximum(abs.(fx))
	plot!(p[c], points, fx .* r, lw = 1.2, linestyle = :dash)
	plot!(p[c], color = :grays, points, Fx, lw = 1.5)
	plot!(p[c], color = :orangered, points, Gx', lw = 1.5)
end
savefig(p, "assets/integrals_predictions.png")

@info "Integrals predictions plot saved"

# Plot fields

C = 20
layout = @layout [a; b c]
p = plot(
	points, X[:, 1:C],
	layout = layout,
	dpi = 300,
	size = (800, 600),
	legend = false,
	tickfontsize = 1,
	xtickfontcolor = :white,
	ytickfontcolor = :white,
	subplot = 1,
)
heatmap!(p, X[:, 1:C]', subplot = 2)
heatmap!(p, Y[:, 1:C]', subplot = 3)
savefig(p, "assets/integrals_fields.png")

@info "Fields plot saved"
