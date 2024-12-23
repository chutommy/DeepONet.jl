@testset "Utilities" verbose = true begin
	@testset "UXS Split" begin
		params = [
			(U = (5, 100), X = (3, 100), S = (2, 100), split = 0.8),
			(U = (5, 200), X = (3, 200), S = (2, 200), split = 0.6),
			(U = (6, 200), X = (4, 200), S = (3, 200), split = 0.5),
		]
		@testset "U=$(p.U) X=$(p.X) S=$(p.S) split=$(p.split)" for p in params
			U, X, S = rand(Float32, p.U), rand(Float32, p.X), rand(Float32, p.S)
			train, test = uxs_split(U, X, S; split = p.split, toshuffle = true)
			U_train, X_train, S_train = train
			U_test, X_test, S_test = test
			@test size(U_train) == (size(U, 1), round(Int, size(U, 2) * p.split))
			@test size(X_train) == (size(X, 1), round(Int, size(X, 2) * p.split))
			@test size(S_train) == (size(S, 1), round(Int, size(S, 2) * p.split))
			@test size(U_test) == (size(U, 1), round(Int, size(U, 2) * (1 - p.split)))
			@test size(X_test) == (size(X, 1), round(Int, size(X, 2) * (1 - p.split)))
			@test size(S_test) == (size(S, 1), round(Int, size(S, 2) * (1 - p.split)))
		end
	end

	@testset "Evaluation" begin
		params = [
			(uin = 3, xin = 1, pin = 3, act = [tanh]),
			(uin = 5, xin = 2, pin = 6, act = [tanh]),
			(uin = 5, xin = 2, pin = 6, act = [tanh]),
		]
		sizes = [1, 10, 100]
		@testset "$(p)" for p in params, s in sizes
			model = DeepONetModel(p.uin, p.xin, p.pin, p.act;
				branch_sizes = [3], trunk_sizes = [3], output_sizes = [1])
			u = rand(Float32, (p.uin, s))
			x = rand(Float32, (p.xin, s))
			s = rand(Float32, (1, s))
			loader = Flux.DataLoader((u, x, s), batchsize = 5)
			loss = evaluate(model, loader; loss_fn = Flux.Losses.mse)
			@test loss >= 0
		end
	end

	@testset "Training" begin
		p = (uin = 3, xin = 1, pin = 3, act = [tanh], s = 10)
		model = DeepONetModel(p.uin, p.xin, p.pin, p.act;
			branch_sizes = [3], trunk_sizes = [3], output_sizes = [1])
		U = rand(Float32, (p.uin, p.s))
		X = rand(Float32, (p.xin, p.s))
		S = rand(Float32, (1, p.s))

		train_loader = Flux.DataLoader((U, X, S), batchsize = 5)
		test_loader = Flux.DataLoader((U, X, S), batchsize = 5)
		opt = Flux.setup(Flux.AdamW(0.001), model)
		train_losses, test_losses = train!(model, opt, train_loader, test_loader; epochs = 3)

		@test length(train_losses) == 3
		@test length(test_losses) == 3
		# Check that the losses are decreasing
		@test train_losses[end] < train_losses[1]
		@test test_losses[end] < test_losses[1]
	end
end
