using Agents
import Agents: AbstractSpace
using ProgressLogging
using Parameters
using Unitful
import Unitful: 𝐋, 𝐓, mm, s, rad
using DataFrames
using Distributions
using LinearAlgebra

include("pheromone.jl")
include("speedprocess.jl")

# Ant struct - this describes a single agent and inherits from Agents' AbstractAgent
@agent struct Ant(ContinuousAgent{2, Float64})
    # Orientation angle
    theta::Float64
    # This is so we can have ants that move at a continuous pace, or vary their
    # velocity according to some process.
    speedprocess::SpeedProcess
    # Is the ant stopped (i.e. an immovable obstacle)?
    stopped::Bool
    function Ant(id::Int64, pos::SVector{2,Float64}, theta::Real, speedprocess::SpeedProcess, time::Float64)
        new(id, pos, (0.0, 0.0), theta, copy(speedprocess, time), false)
    end
end

# Shortcuts for the units we use most.
# const mm = u"mm"
# const s = u"s"
# const rad = u"rad"

# This describes the environment, size and type (i.e. :arena or :bridge).
struct Domain
    width::typeof(1.0mm)
    height::typeof(1.0mm)
    type::Symbol
end
function Base.show(io::IO, d::Domain)
    print(io, "$(d.width)×$(d.height)[$(d.type)]")
end
Base.isless(a::Domain, b::Domain) = a.height*a.width < b.height*b.width
Base.isequal(a::Domain, b::Domain) = (a.height == b.height) && (a.width == b.width) && (a.type == b.type)

"""
    AntModelParameters(domain, v0, Dθ, γ, κ, Dc, Δx, S, L, β, Δt, T, pmodel, η, λ, ρ)

Create a parameter struct for our simulations. The parameters are as follows:

 * `domain`: a `Domain` instance. (default `Domain(100.0mm, 100.0mm, :bridge)`)
 * `v0`: either the `SpeedProcess` the ants should use or a Unitful value of unit `mm/s` which assumes a constant speed process (default `15.5mm/s`).
 * `Dθ`: orientation angle diffusion coefficient (default `0.38rad^2/s`).
 * `γ`: pheromone-orientation interaction strength constant (default `0.0rad*mm/s`, i.e. pheromone is ignored).
 * `κ`: pheromone-position interaction strength constant (default `0.0mm^2/s`, i.e. pheromone is ignored).
 * `Dc`: pheromone diffusion coefficient (default `0.0mm^2/s`).
 * `Δx`: pheromone PDE spatial discretization (default `0.1mm`).
 * `S`: Ant-ant Morse force interaction strength (default `129.0mm^2/s`).
 * `L`: Ant-ant Morse force interaction length (default `1.98mm`).
 * `β`: Ant spawn rate - `add_ant!` is called according to a Poisson process with this rate (default `0.1/s`).
 * `Δt`: simulation time step (default `0.04s`).
 * `T`: simulation duration (default `100s`).
 * `pmodel`: pher
"""
@with_kw struct AntModelParameters
    domain::Domain = Domain(100.0mm, 100.0mm, :bridge)
    v0::Union{SpeedProcess,typeof(1.0mm/s)} = 15.5mm/s
    Dθ::typeof(1.0rad^2/s) = 0.38rad^2/s
    γ::typeof(1.0rad*mm/s) = 0.0rad*mm/s
    κ::typeof(1.0mm^2/s) = 0.0mm^2/s
    Dc::typeof(1.0mm^2/s) = 0.0mm^2/s
    Δx::typeof(1.0mm) = 0.1mm
    S::typeof(1.0mm^2/s) = 129.0mm^2/s
    L::typeof(1.0mm) = 1.98mm
    β::typeof(1.0/s) = 0.1/s
    Δt::typeof(1.0s) = 0.04s
    T::typeof(1.0s) = 100s
    pmodel::Symbol = :gradient_nonlocal
    η::typeof(1.0mm) = 1.0mm
    λ::typeof(1.0mm) = 1.0mm
    ρ::typeof(1.0mm) = 1.0mm
    stop_when_no_ants_left::Bool = false
end

struct Pillar
    pos::SVector{2,Float64}
    radius::Float64
    strength::Float64
end

struct AntKilledException <: Exception
end

"""
    ant_model(parameters)

Create and return a new ant model object with the given `parameters`.
"""
function ant_model(parameters::AntModelParameters)
    # Extract parameters into variable (can't use unpack unfortunately)
    width = uconvert(mm, parameters.domain.width).val
    height = uconvert(mm, parameters.domain.height).val
    #speed = uconvert(mm/s, parameters.speed).val
    angle_diffusion_coefficient = uconvert(rad^2/s, parameters.Dθ).val
    gamma = uconvert(rad*mm/s, parameters.γ).val
    interaction_strength = uconvert(mm^2/s, parameters.S).val
    interaction_scale = uconvert(mm, parameters.L).val
    time_step = uconvert(s, parameters.Δt).val
    kappa = uconvert(mm^2/s, parameters.κ).val
    spawn_rate = uconvert(s^-1, parameters.β).val

    # Create space and pheromone
    space2d::ContinuousSpace = Agents.ContinuousSpace((width, height), spacing=1.0, periodic=false)

    D = uconvert(mm^2/s, parameters.Dc).val
    η = uconvert(mm, parameters.η).val
    λ = uconvert(mm, parameters.λ).val
    ρ = uconvert(mm, parameters.ρ).val
    Δx = uconvert(mm, parameters.Δx).val
    pheromone = Pheromone{parameters.pmodel}(width, height, D, Δx, η, [λ, ρ])

    # Create model
    model = StandardABM(
        Ant,
        space2d;
        properties = Dict(
            :dt => time_step,
            :spawn_distribution => Poisson(time_step*spawn_rate),
            :speed => isa(parameters.v0, Unitful.Quantity) ? ConstantSpeedProcess(uconvert(mm/s,parameters.v0).val) : parameters.v0,
            :angle_stepsize => sqrt(2*angle_diffusion_coefficient*time_step),
            :gammadt => gamma*time_step,
            :kappadt => kappa*time_step,
            :interaction_strength => interaction_strength,
            :interaction_scale => interaction_scale,
            :pheromone => pheromone,
            :time => 0.0,
            :pillars => Pillar[],
            :stop_when_no_ants_left => parameters.stop_when_no_ants_left,
        ),
        agent_step! = ant_step!,
        model_step! = model_step!,
    )

    # Add the first ant.
    if spawn_rate > 0
        add_ant!(model)
    end

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
function add_ant!(model)
    e::SVector{2,Float64} = spacesize(model)
    side = rand() > 0.5
    x = SVector(e[1]*0.99*side, e[2]*rand())
    θ = π*side
    add_agent!(x, model, θ, model.speed, model.time)
end

"""
    add_pillar!(model, pos, radius)

Add a pillar (obstacle) at the given position and radius
"""
function add_pillar!(model, pos::SVector{2,Float64}, radius::Float64, strength::Float64)
    push!(model.pillars, Pillar(pos, radius, strength))
end

"""
    model_step!(model)

Perform a single ant model step.
"""
function model_step!(model)
    # Perform pheromone diffusion
    model.time += model.dt
    steppheromone!(model.pheromone, model.time)

    # Add ants on the ends of the bridge according to a Poisson process
    for _ in 1:rand(model.spawn_distribution)
        add_ant!(model)
    end

    for a in allagents(model)
        a.vel = (0., 0.)
    end

    S::Float64 = model.interaction_strength
    L::Float64 = model.interaction_scale

    for (a1, a2) in interacting_pairs(model, 10.0*L, :all)
        d::SVector{2,Float64} = a2.pos .- a1.pos
        r::Float64 = hypot(d[1], d[2])
        fr::Float64 = -S*exp(-r/L)/(L*r)
        force::SVector{2,Float64} = d .* fr
        a1.vel = a1.vel .+ force
        a2.vel = a2.vel .- force
    end

    pillars::Vector{Pillar} = model.pillars
    for pillar in pillars
        R::Float64 = pillar.radius
        Sp::Float64 = pillar.strength
        pos::SVector{2,Float64} = pillar.pos
        for a in allagents(model)
            d::SVector{2,Float64} = a.pos .- pos
            r::Float64 = hypot(d[1], d[2])
            if r < 10.0*R
                fr::Float64 = -Sp*exp(-r/R)/(R*r)
                force::SVector{2,Float64} = d .* fr
                a.vel = a.vel .- force
            end
        end
    end
end

function boundarycheck!(ant, pos::SVector{2,Float64}, model)::SVector{2,Float64}
    extent::SVector{2,Float64} = spacesize(model)

    # Kill the ant if it moves outside the bridge bounds left and right
    if (pos[1] < 0.0) | (pos[1] > extent[1])
        remove_agent!(ant, model)
        throw(AntKilledException())
    end

    # Take care of reflection at the upper and lower bridge edge
    if pos[2] < 0
        pos = (pos[1], -pos[2])
        ant.theta = π*round(ant.theta/π)
    elseif pos[2] > extent[2]
        pos = (pos[1], 2*extent[2] - pos[2])
        ant.theta = π*round(ant.theta/π)
    end

    return pos
end

"""
    ant_step!(ant, model)

Perform a single ant's step. This needs to be called *after* `model_step!`.
"""
function ant_step!(ant, model)
    η::Float64 = model.angle_stepsize
    γ::Float64 = model.gammadt
    κ::Float64 = model.kappadt
    dt::Float64 = model.dt
    ph::Pheromone = model.pheromone

    v0proc::SpeedProcess = model.speed
    v0::Float64  = speed(v0proc, model.time)

    pos = ant.pos
    θ = ant.theta

    # Call the appropriate pheromone interaction model method (determined by T in the pheromone type)
    gradient_force, attraction = pheromone_interaction(pos, θ, ph)

    # Update the velocity vector with the constant speed in the ant's direction
    ant.vel = ant.vel .+ (cos(θ), sin(θ)) .* v0

    # Update the ant's position with the current velocity vector plus an (optional) pheromone attraction
    if !ant.stopped
        pos = pos .+ (dt .* ant.vel) .+ (κ .* attraction)
    end

    # Update the direction angle with rotational Brownian motion
    θ = θ + η * randn() + γ * gradient_force
    if θ > π
        θ -= 2π
    elseif θ < -π
        θ += 2π
    end
    ant.theta = θ

    # Take care of boundaries.
    try
        pos = boundarycheck!(ant, pos, model)
    catch exc
        if isa(exc, AntKilledException)
            return pos
        end
    end

    # Add pheromone between the last and current ant positions
    addpheromone!(ph, ant.pos, pos)

    # Perform the move with the new position
    move_agent!(ant, pos, model)

    return ant.pos
end

"""
    run_ant_model(parameters, iteration_callbacks = [])

Run the ant model with the given parameters and return the collected ant trajectories.
Optionally, a list of callback functions can be given which are then called at each time step.
"""
function run_ant_model(parameters::AntModelParameters; initialization_callbacks::Vector{S} = Function[], iteration_callbacks::Vector{T} = Function[], showprogress = true) where {T <: Function, S <: Function}
    # Create new ant (agents) model
    model = ant_model(parameters)
    number_of_steps::Int64 = ceil(Int64, parameters.T / parameters.Δt)
    local stop_condition::Union{Int, Function}
    if model.stop_when_no_ants_left
        function until(m::StandardABM, t::Int)
            stopped = map((a) -> !a.stopped, allagents(m))
            return (sum(stopped) == 0) | (t >= number_of_steps)
        end
        stop_condition = until
    else
        stop_condition = number_of_steps
    end

    # Prepare data collection
    position_x(a) = a.pos[1]
    position_y(a) = a.pos[2]
    adata = [position_x, position_y, :theta, :stopped]

    # Run initialization callbacks
    for f in initialization_callbacks
        f(model, s)
    end

    # This strange construct allows us to display a progress bar when using Agents' `run!` function.
    # The `when_model` function argument to `run!` is called at each iteration, so we just use it to
    # update the progress bar. We can also call an optional call back function to e.g. animate our simulation.
    df = ProgressLogging.progress() do id
        function when_model(m, s::Int64)::Bool
            # Update progress bar
            if showprogress
                @info "stepping ants" progress=s/number_of_steps _id=id
            end

            # Perform the call backs
            for f in iteration_callbacks
                f(m, s)
            end
            # Always return false - we don't want to collect model data.
            return false
        end
        df, _ = run!(model, stop_condition, adata = adata, when_model = when_model)
        return df
    end

    return df, model
end
