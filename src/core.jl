using Agents
using ProgressLogging
using Parameters
using Unitful
import Unitful: 𝐋, 𝐓
using DataFrames
using Distributions

include("pheromone.jl")

mutable struct Ant <: AbstractAgent
    id::Int64
    pos::NTuple{2,Float64}
    theta::Float64
end

const mm = u"mm"
const s = u"s"
const rad = u"rad"

@with_kw struct AntModelParameters
    width::typeof(1.0mm) = 100.0mm
    height::typeof(1.0mm) = 10.0mm
    speed::typeof(1.0mm/s) = 15.5mm/s
    angle_diffusion_coefficient::typeof(1.0rad^2/s) = 0.38rad^2/s
    gamma::typeof(1.0rad*mm/s) = 0.0rad*mm/s
    kappa::typeof(1.0mm^2/s) = 0.0mm^2/s
    pheromone_diffusion_coefficient::typeof(1.0mm^2/s) = 0.0mm^2/s
    pheromone_dx::typeof(1.0mm) = 0.1mm
    interaction_strength::typeof(1.0mm^2/s) = 129.0mm^2/s
    interaction_scale::typeof(1.0mm) = 1.98mm
    spawn_rate::typeof(1.0/s) = 0.1/s
    number_of_ants::Int = 0  # Number of ants present at the start
    time_step::typeof(1.0s) = 0.04s
    end_time::typeof(1.0s) = 100s
    pheromone_model::Symbol = :gradient_nonlocal
    pheromone_λ::typeof(1.0mm) = 1.0mm
    pheromone_ρ::typeof(1.0mm) = 1.0mm
    domain_type::Symbol = :bridge
end
const AntModel = AgentBasedModel{K,Ant} where {K}

struct AntKilledException <: Exception
end

"""
    ant_model(parameters)

Create and return a new ant model object with the given `parameters`.
"""
function ant_model(parameters::AntModelParameters)::AntModel
    # Extract parameters into variable (can't use unpack unfortunately)
    width = uconvert(mm, parameters.width).val
    height = uconvert(mm, parameters.height).val
    speed = uconvert(mm/s, parameters.speed).val
    angle_diffusion_coefficient = uconvert(rad^2/s, parameters.angle_diffusion_coefficient).val
    gamma = uconvert(rad*mm/s, parameters.gamma).val
    pheromone_diffusion_coefficient = uconvert(mm^2/s, parameters.pheromone_diffusion_coefficient).val
    interaction_strength = uconvert(mm^2/s, parameters.interaction_strength).val
    interaction_scale = uconvert(mm, parameters.interaction_scale).val
    number_of_ants = parameters.number_of_ants
    time_step = uconvert(s, parameters.time_step).val
    kappa = uconvert(mm^2/s, parameters.kappa).val
    spawn_rate = uconvert(s^-1, parameters.spawn_rate).val
    pheromone_dx = uconvert(mm, parameters.pheromone_dx).val

    # Create space and pheromone
    space2d::ContinuousSpace = ContinuousSpace((width, height), 10.0, periodic=true)
    if parameters.pheromone_model == :gradient
        pheromone = Pheromone_model1(width, height, pheromone_diffusion_coefficient, pheromone_dx)
    elseif parameters.pheromone_model == :gradient_attraction
        pheromone = Pheromone_model2(width, height, pheromone_diffusion_coefficient, pheromone_dx)
    elseif parameters.pheromone_model == :perna
        pheromone = Pheromone_model3(width, height, pheromone_diffusion_coefficient, pheromone_dx, uconvert(mm, parameters.pheromone_λ).val, uconvert(mm, parameters.pheromone_ρ).val)
    elseif parameters.pheromone_model == :gradient_nonlocal
        pheromone = Pheromone_model4(width, height, pheromone_diffusion_coefficient, pheromone_dx, uconvert(mm, parameters.pheromone_λ).val)
    else
        throw(ArgumentError("Unknown pheromone interaction model"))
    end
    ## Only for debugging!
    #pheromone = Pheromone(width, height, 0.1)

    if parameters.domain_type == :bridge
        boundarycheck = boundarycheck_bridge!
        add_ant = add_ant_bridge!
    elseif parameters.domain_type == :arena
        boundarycheck = boundarycheck_arena!
        add_ant = add_ant_arena!
    else
        throw(ArgumentError("Unknown domain type"))
    end

    # Create model
    model::AntModel = ABM(
        Ant,
        space2d,
        properties = Dict(
            :dt => time_step,
            :spawn_distribution => Poisson(time_step*spawn_rate),
            :speed => speed,
            :angle_stepsize => sqrt(2*angle_diffusion_coefficient*time_step),
            :gammadt => gamma*time_step,
            :kappadt => kappa*time_step,
            :interaction_strength => interaction_strength,
            :interaction_scale => interaction_scale,
            :pheromone => pheromone,
            :time => 0.0,
            :boundarycheck => boundarycheck,
            :extent => space2d.extent,
            :add_ant => add_ant,
        )
    )
    model.properties[:neighbors] = (pos::NTuple{2,Float64}, r::Float64)->(model[id].pos for id in nearby_ids(pos, model, r))

    # Add the first ant.
    add_ant(model)

    # # Add some ants - this is mostly for debugging, so usually number_of_ants = 0
    # for i in 1:number_of_ants
    #     pos = (width/2, 1.0) #(width, height) .* Tuple(rand(2))
    #     theta = π/4 #2π * rand()
    #     vel = (0.0, 0.0) #sincos(theta) .* speed
    #     add_agent!(pos, model, vel, theta, 0.0)
    # end

    return model
end

"""
    add_ant_bridge!(model)

Add a single ant at a random y position at the left or right end of the bridge.
"""
function add_ant_bridge!(model::AntModel)
    e::NTuple{2,Float64} = model.space.extent
    side = rand() > 0.5
    x = (e[1]*0.99*side, e[2]*rand())
    θ = π*side
    add_agent!(x, model, θ)
end
function add_ant_arena!(model::AntModel)
    e::NTuple{2,Float64} = model.space.extent
    x = e ./ 2 .+ (rand() - 0.5, rand() - 0.5)
    θ = 2π*rand()
    add_agent!(x, model, θ)
end

"""
    model_step!(model)

Perform a single ant model step.
"""
function model_step!(model::AntModel)
    # Perform pheromone diffusion
    model.time += model.dt
    steppheromone!(model.pheromone, model.time)

    # Add ants on the ends of the bridge according to a Poisson process
    for i in 1:rand(model.spawn_distribution)
        model.add_ant(model)
    end
end

function boundarycheck_bridge!(state::Vector{Float64}, extent::NTuple{2,Float64})
    # Kill the ant if it moves outside the bridge bounds left and right
    if (state[1] < 0.0) | (state[1] > extent[1])
        throw(AntKilledException())
    end

    # Take care of reflection at the upper and lower bridge edge
    if state[2] < 0
        state[2] *= -1
        state[3] = π*round(state[3]/π)
    elseif pos[2] > extent[2]
        state[2] = 2*e[2] - state[2]
        state[3] = π*round(state[3]/π)
    end
end
function boundarycheck_arena!(state::Vector{Float64}, extent::NTuple{2,Float64})
    # Kill the ant if it leaves the arena
    if any(state[1:2] .< 0) | any(state[1:2] .> extent)
        throw(AntKilledException())
    end
end

function ant_new_position(state::Vector{Float64}, parameters::Dict{Symbol,Any})::Vector{Float64}
    neighbors::Function = parameters[:neighbors]
    boundarycheck!::Function = parameters[:boundarycheck]
    η::Float64 = parameters[:angle_stepsize]
    γ::Float64 = parameters[:gammadt]
    κ::Float64 = parameters[:kappadt]
    S::Float64 = parameters[:interaction_strength]
    L::Float64 = parameters[:interaction_scale]
    v0::Float64 = parameters[:speed]
    dt::Float64 = parameters[:dt]
    extent::NTuple{2,Float64} = parameters[:extent]
    ph::Pheromone = parameters[:pheromone]

    pos::NTuple{2,Float64} = (state[1], state[2])

    # Call the appropriate pheromone interaction model method (determined by T in the ant type)
    gradient_force, attraction = pheromone_interaction(state, ph)

    # Update the velocity vector with the constant speed in the ant's direction
    v = (cos(θ), sin(θ)) .* v0

    # Calculate ant-ant interaction forces
    if S != 0.0
        for apos in neighbors(pos, 10.0*L)
            d = apos .- pos
            r = hypot(d[1], d[2])
            if r > 0
                force = d .* (-S*exp(-r/L)/(L*r))
                v = v .+ force
            end
        end
    end

    # Update the ant's position with the current velocity vector plus an (optional) pheromone attraction
    pos = (state[1], state[2]) .+ (dt .* v) .+ (κ * attraction)
    # Update the direction angle with rotational Brownian motion
    θ = state[3] + η * randn() + γ * gradient_force

    state[1:2] .= pos
    state[3] = θ

    # Take care of boundaries.
    boundarycheck!(state, extent)

    return state
end

"""
    ant_step!(ant, model)

Perform a single ant's step. This needs to be called *after* `model_step!`.
"""
function ant_step!(ant::Ant, model::AntModel)
    try
        state = ant_new_position([ant.pos[1], ant.pos[2], ant.theta], model.properties)
        pos = (state[1], state[2])
        ant.theta = state[3]

        # Add pheromone between the last and current ant positions
        addpheromone!(model.pheromone, ant.pos, pos)
        # Perform the move with the new position
        move_agent!(ant, pos, model)

    catch AntKilledException
        kill_agent!(ant, model)
    end

    return ant.pos
end

"""
    run_ant_model(parameters, iteration_callbacks = [])

Run the ant model with the given parameters and return the collected ant trajectories.
Optionally, a list of callback functions can be given which are then called at each time step.
"""
function run_ant_model(parameters::AntModelParameters; iteration_callbacks::Vector{T} = Function[]) where {T <: Function}
    # Create new ant (agents) model
    model = ant_model(parameters)
    number_of_steps::Int64 = ceil(Int64, parameters.end_time / parameters.time_step)

    # Prepare data collection
    position_x(a) = a.pos[1]
    position_y(a) = a.pos[2]
    adata = [position_x, position_y, :theta]

    # This strange construct allows us to display a progress bar when using Agents' `run!` function.
    # The `when_model` function argument to `run!` is called at each iteration, so we just use it to
    # update the progress bar. We can also call an optional call back function to e.g. animate our simulation.
    df = ProgressLogging.progress() do id
        function when_model(m::AntModel, s::Int64)::Bool
            # Update progress bar
            @info "stepping ants" progress=s/number_of_steps _id=id
            # Perform the call backs
            for f in iteration_callbacks
                f(m, s)
            end
            # Always return false - we don't want to collect model data.
            return false
        end
        df, _ = run!(model, ant_step!, model_step!, number_of_steps, adata = adata, when_model = when_model)
        return df
    end

    return df
end
