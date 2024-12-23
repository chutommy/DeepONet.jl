module DeepONet

export DeepONetModel
export ParallelDense
export RandomFieldGenerator
export generate_random_fields
export uxs_split, evaluate, train!

using Flux
using GaussianRandomFields
using ProgressMeter
using Random
using Suppressor

include("generator.jl")
include("model.jl")
include("utils.jl")

end # module DeepONet
