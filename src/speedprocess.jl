abstract type SpeedProcess end
Broadcast.broadcastable(sp::SpeedProcess) = Ref(sp)

#### Constant ####
struct ConstantSpeedProcess <: SpeedProcess
    μ::Float64
end
Base.copy(sp::ConstantSpeedProcess, time::Float64) = ConstantSpeedProcess(sp.μ)
function Base.show(io::IO, p::ConstantSpeedProcess)
    print(io, "Const(μ=$(p.μ))")
end
function speed(sp::ConstantSpeedProcess, time::Float64)::Float64
    return sp.μ
end

#### Sinusoidal ####
struct SinusoidalSpeedProcess <: SpeedProcess
    μ::Float64
    A::Float64
    f::Float64
    ϕ::Float64
end
Base.copy(sp::SinusoidalSpeedProcess, time::Float64) = SinusoidalSpeedProcess(sp.μ, sp.A, sp.f, sp.ϕ)
function Base.show(io::IO, p::SinusoidalSpeedProcess)
    print(io, "Sinus(μ=$(p.μ),A=$(p.A),f=$(p.f),ϕ=$(p.ϕ))")
end
function speed(sp::SinusoidalSpeedProcess, t::Float64)::Float64
    @unpack μ, A, f, ϕ = sp
    return μ + A * sin(2π*f*t + ϕ)
end

#### Ornstein-Uhlenbeck ####
mutable struct OrnsteinUhlenbeckSpeedProcess <: SpeedProcess
    μ::Float64
    θ::Float64
    σ::Float64

    x::Float64
    t::Float64
    function OrnsteinUhlenbeckSpeedProcess(μ::Float64, θ::Float64, σ::Float64; time::Float64 = 0.0)
        new(μ, θ, σ, μ, time)
    end
end
Base.copy(sp::OrnsteinUhlenbeckSpeedProcess, time::Float64) = OrnsteinUhlenbeckSpeedProcess(sp.μ, sp.θ, sp.σ, time = time)
function Base.show(io::IO, p::OrnsteinUhlenbeckSpeedProcess)
    print(io, "OU(μ=$(p.μ),θ=$(p.θ),σ=$(p.σ))")
end
function speed(sp::OrnsteinUhlenbeckSpeedProcess, time::Float64)::Float64
    @unpack μ, θ, σ, x, t = sp

    Δt = time - t
    @assert Δt >= 0 "time needs to increase with consecutive calls!"
    sp.t = time
    x += θ*(μ - x)*Δt + σ*√(Δt)*randn()
    sp.x = x
    return x
end