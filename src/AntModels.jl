module AntModels

using Requires

export AntModelParameters, Domain, AntModel, run_ant_model, mm, s, rad

include("core.jl")


function DrWatson_glue()

end

function __init__()
    @require DrWatson="634d3b9d-ee7a-5ddf-bec9-22491ea816e1" begin
        @eval DrWatson.default_allowed(p::AntModelParameters) = (Symbol, Domain, Unitful.Quantity, SpeedProcess)
        @eval begin
            using .DrWatson
            function run_ant_model_DrWatson(p::AntModelParameters; kwargs...)
                ret = Dict{String,Any}([string(k) => v for (k, v) in struct2dict(p)]...)
                df, _ = run_ant_model(p; kwargs...)
                ret["data"] = df
                return ret
            end
        end
    end
end

end # module
