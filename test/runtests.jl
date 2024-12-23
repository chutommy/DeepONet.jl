using Aqua
using Flux
using DeepONet
using Test

@testset "DeepONet.jl" verbose = true begin
	include("aqua.jl")
	include("generator.jl")
	include("model.jl")
	include("utils.jl")
end
