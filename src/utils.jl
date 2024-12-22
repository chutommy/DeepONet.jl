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

function evaluate(model::Model, loader::Flux.DataLoader; loss_fn = Flux.Losses.mse)
	loss = 0.0
	for (u, x, s) in loader
		s_hat = model(u, x)
		loss += loss_fn(s_hat, s)
	end
	return loss
end

function train!(
	model::Model,
	opt_state::NamedTuple,
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
			Flux.update!(opt_state, model, grads[1])
			train_losses[e] += loss
		end
		test_losses[e] = evaluate(model, test_loader, loss_fn = loss_fn)
	end
	return train_losses, test_losses
end
