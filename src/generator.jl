struct RandomFieldGenerator
	size::Int
	mean::Vector
	cov::AbstractCovarianceFunction
	generator::GaussianRandomFieldGenerator
	grf::GaussianRandomField
end

function RandomFieldGenerator(
	points::AbstractVector, dim::Int, size::Int;
	mean::Real, std::Real, param::Real,
)
	model = Gaussian(param, σ = std)
	means = fill(mean, size)
	covf = CovarianceFunction(dim, model)
	generator = CirculantEmbedding()
	grf = GaussianRandomField(means, covf, generator, points)
	return RandomFieldGenerator(length(points), means, covf, generator, grf)
end

function (generator::RandomFieldGenerator)()
	sample(generator.grf)
end
function (generator::RandomFieldGenerator)(n::Int)
	fields = zeros(Float32, (generator.size, n))
	for i in 1:n
		fields[:, i] = generator()
	end
	return fields
end

function generate_random_fields(
	points::AbstractVector, dim::Int, size::Int;
	means::Vector, stds::Vector, params::Vector,
	K::Int,
)
	generators = Vector{RandomFieldGenerator}()
	for p in params, s in stds, m in means
		@suppress begin
			push!(generators, RandomFieldGenerator(points, dim, size; mean = m, std = s, param = p))
		end
	end
	gsize = length(generators)
	total = gsize * K
	out = zeros(Float32, (size, total))
	for g in 1:gsize
		gi = (g - 1) * K
		a, b = gi + 1, gi + K
		out[:, a:b] = generators[g](K)
	end
	return out
end
