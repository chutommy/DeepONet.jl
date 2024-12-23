@testset "DeepONetModel" verbose = true begin
	@testset "ParallelDense" verbose = true begin
		params = [
			(in = 10, out = 10, act = [relu, tanh, sigmoid]),
			(in = 20, out = 10, act = [relu, tanh, sigmoid]),
			(in = 10, out = 20, act = [relu, tanh, sigmoid]),
			(in = 20, out = 20, act = [relu]),
		]
		@testset "Glorot Initialization" begin
			@testset "in_dim=$(p.in) out_dim=$(p.out) act=$(p.act)" for p in params
				layer = ParallelDense(p.in, p.out, p.act)
				@test size(layer.W) == (p.out, p.in)
				@test size(layer.b) == (p.out,)
				@test length(layer.act) == length(p.act)
				@test layer.W != zeros(p.out, p.in)
				@test layer.b != zeros((p.out,))
				@test layer.act == p.act
			end
		end

		bs = [1, 10, 100]
		@testset "Forward Pass" begin
			@testset "in_dim=$(p.in) out_dim=$(p.out) act=$(p.act)" for p in params, b in bs
				layer = ParallelDense(p.in, p.out, p.act)
				input = rand(Float32, p.in, b)
				output = layer(input)
				@test size(output) == (length(p.act) * p.out, b)
			end
		end
	end

	@testset "OperatorNet" begin
		b = 100
		params = [
			(in = 3, out = 3, act = [relu, tanh, sigmoid], sizes = [6, 6, 6]),
			(in = 6, out = 3, act = [relu, tanh, sigmoid], sizes = [6, 6, 6]),
			(in = 3, out = 6, act = [relu, tanh, sigmoid], sizes = [6, 6, 6]),
			(in = 6, out = 6, act = [relu, tanh, sigmoid], sizes = [6]),
			(in = 6, out = 6, act = [relu, tanh], sizes = [6, 6, 6]),
		]
		@testset "in_dim=$(p.in) out_dim=$(p.out) act=$(p.act) sizes=$(p.sizes)" for p in params
			opnet = OperatorNet(p.in, p.out, p.act, p.sizes)
			@test length(opnet.layers) == length(p.sizes) + 1

			input = rand(Float32, p.in, b)
			output = opnet(input)
			@test size(output) == (p.out, b)
		end
	end

	@testset "Forward Pass" begin
		dparams = [
			(branch_dim = 2, trunk_dim = 3, output_dim = 4, act = [relu, sigmoid]),
			(branch_dim = 5, trunk_dim = 6, output_dim = 7, act = [relu, sigmoid]),
			(branch_dim = 5, trunk_dim = 6, output_dim = 7, act = [relu]),
		]
		sparams = [
			(branch_sizes = [2], trunk_sizes = [2], output_sizes = [2]),
			(branch_sizes = [2, 2], trunk_sizes = [2, 2], output_sizes = [2, 2]),
			(branch_sizes = [2, 3, 2], trunk_sizes = [2, 3, 2], output_sizes = [2, 3, 2]),
		]
		batches = [1, 10]
		@testset "dims=$(dp) sizes=$(sp) bs=$(b)" for dp in dparams, sp in sparams, b in batches
			model = DeepONetModel(dp.branch_dim, dp.trunk_dim, dp.output_dim, dp.act;
				branch_sizes = sp.branch_sizes,
				trunk_sizes = sp.trunk_sizes,
				output_sizes = sp.output_sizes,
			)
			@test length(model.branch.layers) == length(sp.branch_sizes) + 1
			@test length(model.trunk.layers) == length(sp.trunk_sizes) + 1
			@test length(model.project.layers) == length(sp.output_sizes) + 1

			u = rand(Float32, dp.branch_dim)
			x = rand(Float32, dp.trunk_dim)
			output = model(u, x)
			@test size(output) == (1,)
		end
	end
end
