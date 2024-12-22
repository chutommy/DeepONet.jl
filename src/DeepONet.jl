module DeepONet

export DeepONetModel
export generate_random_fields
export uxs_split, evaluate, train!

using Flux
using GaussianRandomFields
using ProgressMeter
using Random
using Suppressor

include("include.jl")

end # module DeepONet
