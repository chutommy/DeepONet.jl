module DeepONet

export Model
export generate_random_fields
export uxs_split
export evaluate
export train!

using Flux
using GaussianRandomFields
using ProgressMeter
using Random
using Suppressor

include("include.jl")

end # module DeepONet
