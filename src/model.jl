struct ParallelDense
	W::AbstractMatrix
	b::AbstractVector
	act::Tuple{Vararg{Function}}
end

function (d::ParallelDense)(x)
	preactivation = d.W * x .+ d.b
	activation = [act(preactivation) for act in d.act]
	return cat(activation..., dims = 1)
end

function ParallelDense(in::Int, out::Int)
	ParallelDense(glorot_normal(out, in), zeros(Float32, out))
end

model = let neurons = 40, in1 = M, in2 = 1, output_neurons = 20
	branch1 = Layer(in1, neurons)
	branch2 = Layer(neurons, neurons)
	branch3 = Layer(neurons, output_neurons)

	trunk1 = Layer(in2, neurons)
	trunk2 = Layer(neurons, neurons)
	trunk3 = Layer(neurons, neurons)
	trunk4 = Layer(neurons, output_neurons)

	adjoint = Layer(output_neurons, 1)

	function fwd(u, x)
		branch = branch3(gelu(branch2(gelu(branch1(u)))))
		trunk = trunk4(gelu(trunk3(gelu(trunk2(gelu(trunk1(x)))))))
		output = adjoint(branch .* trunk)
		return output
	end
end
