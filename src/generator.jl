"""
	struct RandomFieldGenerator

A struct for generating Gaussian random fields.
"""
struct RandomFieldGenerator
	resolution::Int
	mean::Vector
	cov::AbstractCovarianceFunction
	generator::GaussianRandomFieldGenerator
	grf::GaussianRandomField
end

"""
	RandomFieldGenerator(
		points::AbstractVector, dim::Int, resolution::Int;
		mean::Real, std::Real, param::Real,
	)

Constructs a `RandomFieldGenerator` struct.

# Arguments
- `points::AbstractVector`: The spatial points for the random field.
- `dim::Int`: The dimensionality of the random field.
- `resolution::Int`: The resolution of the realizations.

# Keywords
- `mean::Real`: Mean value of the random field.
- `std::Real`: Standard deviation of the random field.
- `param::Real`: Parameter of the covariance function.

# Returns
- A `RandomFieldGenerator` instance.
"""
function RandomFieldGenerator(
	points::AbstractVector, dim::Int, resolution::Int;
	mean::Real, std::Real, param::Real,
)
	model = Gaussian(param, σ = std)
	means = fill(mean, resolution)
	covf = CovarianceFunction(dim, model)
	generator = CirculantEmbedding()
	grf = GaussianRandomField(means, covf, generator, points)
	return RandomFieldGenerator(length(points), means, covf, generator, grf)
end

"""
	(generator::RandomFieldGenerator)()
	(generator::RandomFieldGenerator)(n::Int)

Generates a single or `n` random fields using the `RandomFieldGenerator`.
"""
function (generator::RandomFieldGenerator)()
	return sample(generator.grf)
end
function (generator::RandomFieldGenerator)(n::Int)
	fields = zeros(Float32, (generator.resolution, n))
	for i in 1:n
		fields[:, i] = generator()
	end
	return fields
end

"""
	generate_random_fields(
		points::AbstractVector, dim::Int, resolution::Int;
		means::Vector, stds::Vector, params::Vector, K::Int
	)

Generates multiple random fields with varying parameters.

# Arguments
- `points::AbstractVector`: Spatial points for the random fields.
- `dim::Int`: Dimensionality of the random fields.
- `resolution::Int`: Number of realizations per random field.

# Keywords
- `means::Vector`: Mean values for each random field.
- `stds::Vector`: Standard deviations for each random field.
- `params::Vector`: Parameters for the covariance functions.
- `K::Int`: Number of random fields to generate.

# Returns
- An array of `K` sets of random field realizations.
"""
function generate_random_fields(
	points::AbstractVector, dim::Int, resolution::Int;
	means::Vector, stds::Vector, params::Vector, K::Int,
)
	generators = Vector{RandomFieldGenerator}()
	for p in params, s in stds, m in means
		@suppress begin
			g = RandomFieldGenerator(
				points, dim, resolution;
				mean = m, std = s, param = p,
			)
			push!(generators, g)
		end
	end
	gsize = length(generators)
	total = gsize * K
	out = zeros(Float32, (resolution, total))
	for g in 1:gsize
		gi = (g - 1) * K
		a, b = gi + 1, gi + K
		out[:, a:b] = generators[g](K)
	end
	return out
end
