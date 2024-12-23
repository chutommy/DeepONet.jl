@testset "Random Field Generator" begin
	points, resolution = 1:100, 100
	gparams = [
		(mean = 0.0, std = 1.0, par = 0.1),
		(mean = 0.0, std = 1.0, par = 0.2),
		(mean = 0.0, std = 2.0, par = 0.1),
		(mean = 2.0, std = 1.0, par = 0.1),
	]
	@testset "Single Field" begin
		@testset "mean=$(p.mean), std=$(p.std), par=$(p.par)" for p in gparams
			generator = DeepONet.RandomFieldGenerator(points, 1, resolution;
				mean = p.mean, std = p.std, param = p.par)
			@test size(generator()) == (resolution,)
		end
	end

	fs = [1, 10, 100]
	@testset "Multiple Fields" begin
		@testset "mean=$(p.mean), std=$(p.std), par=$(p.par)" for p in gparams, s in fs
			generator = DeepONet.RandomFieldGenerator(points, 1, resolution;
				mean = p.mean, std = p.std, param = p.par)
			@test size(generator(s)) == (resolution, s)
		end
	end

	gfparams = [
		(means = [0, 1, 2], stds = [1, 3, 5], pars = [0.1, 0.3, 0.5], K = 10),
		(means = [0, 1, 2], stds = [1, 3, 5], pars = [0.1, 0.3, 0.5], K = 20),
		(means = [0, 1], stds = [1, 3, 5], pars = [0.1, 0.3, 0.5], K = 10),
		(means = [0, 1, 2], stds = [1, 3, 5], pars = [0.1, 0.3, 0.5], K = 10),
		(means = [0, 1, 2], stds = [1, 3, 5], pars = [0.1, 0.3], K = 10),
	]
	@testset "Multiple Fields with Different Parameters" begin
		@testset "mean=$(p.means) std=$(p.stds) par=$(p.pars), K=$(p.K)" for p in gfparams
			field = generate_random_fields(points, 1, resolution;
				means = p.means, stds = p.stds, params = p.pars, K = p.K)
			total = length(p.means) * length(p.stds) * length(p.pars) * p.K
			@test size(field) == (resolution, total)
		end
	end
end
