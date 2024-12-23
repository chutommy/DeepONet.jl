using Aqua
using DeepONet
using Test

@testset "DeepONet.jl" begin
	@testset "Code quality (Aqua.jl)" begin
		Aqua.test_all(DeepONet)
	end
end
