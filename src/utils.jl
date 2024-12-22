"""
	uxs_split(U::Matrix, X::Matrix, S::Matrix; split::Real, toshuffle::Bool = false)

Splits operator dataset into training and testing datasets.

# Arguments
- `U::Matrix`: Matrix for `u` initial condition data.
- `X::Matrix`: Matrix for `x` input grid/mesh data.
- `S::Matrix`: Matrix for `s` solution function data.

# Keywords
- `split::Real`: Fraction of the data to use for training.
- `toshuffle::Bool`: Whether to shuffle the data before splitting (default: `false`).

# Returns
- `train_uxs`: Training dataset (U, X, S).
- `test_uxs`: Testing dataset (U, X, S).
"""
function uxs_split(U::Matrix, X::Matrix, S::Matrix; split::Real, toshuffle::Bool = false)
	N = size(U, 2)
	indices = toshuffle ? shuffle(1:N) : 1:N
	n_train = Int(floor(split * N))
	train_view = indices[1:n_train]
	test_view = indices[n_train+1:end]
	train_uxs = U[:, train_view], X[:, train_view], S[:, train_view]
	test_uxs = U[:, test_view], X[:, test_view], S[:, test_view]
	return train_uxs, test_uxs
end

"""
Evaluates the model on a dataset using a specified loss function.
"""
function evaluate(model::DeepONetModel, loader::Flux.DataLoader; loss_fn = Flux.Losses.mse)
	loss = 0.0
	for (u, x, s) in loader
		s_hat = model(u, x)
		loss += loss_fn(s_hat, s)
	end
	return loss
end

"""
	train!(
		model::DeepONetModel,
		optimizer::NamedTuple,
		train_loader::Flux.DataLoader,
		test_loader::Flux.DataLoader;
		epochs::Int = 30,
		loss_fn::Function = Flux.Losses.mse
	)

Trains the model using the provided data loaders and optimizer.

# Arguments
- `model::DeepONetModel`: The model to train.
- `optimizer::NamedTuple`: Optimizer state for training.
- `train_loader::Flux.DataLoader`: DataLoader containing training data.
- `test_loader::Flux.DataLoader`: DataLoader containing testing data.

# Keywords
- `epochs::Int`: Number of training epochs (default: `30`).
- `loss_fn::Function`: Loss function to use (default: `Flux.Losses.mse`).

# Returns
- `train_losses`: Array of training losses for each epoch.
- `test_losses`: Array of testing losses for each epoch.
"""
function train!(
	model::DeepONetModel,
	optimizer::NamedTuple,
	train_loader::Flux.DataLoader,
	test_loader::Flux.DataLoader;
	epochs::Int = 30,
	loss_fn::Function = Flux.Losses.mse,
)
	train_losses = zeros(Float32, epochs)
	test_losses = zeros(Float32, epochs)
	@showprogress for e in 1:epochs
		for (u, x, s) in train_loader
			loss, grads = Flux.withgradient(model) do m
				s_hat = m(u, x)
				loss_fn(s_hat, s)
			end
			Flux.update!(optimizer, model, grads[1])
			train_losses[e] += loss
		end
		test_losses[e] = evaluate(model, test_loader, loss_fn = loss_fn)
	end
	return train_losses, test_losses
end
