using SpecialFunctions

mutable struct Pheromone
    amount::Vector{Array{Float64,2}}
    time::Float64
    D::Float64
    Δx::Float64
    Δt::Float64
    interaction_model::Function
    interaction_parameters::Vector{Float64}
    function Pheromone(width::Float64, height::Float64, D::Float64, Δx::Float64, interaction_model::Function, interaction_parameters::Vector{Float64})
        Nw = trunc(Int64, width/Δx) + 3
        Nh = trunc(Int64, height/Δx) + 3
        Δt = 0.1Δx^2/D # One fifth of stability condition
        new([zeros(Float64,Nw,Nh), zeros(Float64,Nw,Nh)], 0.0, D, Δx, Δt, interaction_model, interaction_parameters)
    end
end
function Pheromone(width::Float64, height::Float64, Δx::Float64) # For testing only!
    ph = Pheromone(width, height, 0.0, Δx, (_::Vector{Float64})->(0.0, (0.0, 0.0)), [])
    Nw = size(ph.amount[1],1)
    Nh = size(ph.amount[1],2)
    for s=16:15:Nw
        drawgaussianline!(ph.amount[1], (s-15, (Nh-3)÷2-2), (s, (Nh-3)÷2-2), 4.0)
        drawgaussianline!(ph.amount[2], (s-15, (Nh-3)÷2-2), (s, (Nh-3)÷2-2), 4.0)
    end
    ph
end

function drawgaussianline!(img::Array{Float64,2}, a::NTuple{2,Int64}, b::NTuple{2,Int64}, σ::Float64)
    left::Int64 = max(2, min(a[1]-3σ, b[1]-3σ))
    right::Int64 = min(size(img,1)-1, max(a[1]+3σ, b[1]+3σ))
    bottom::Int64 = max(2, min(a[2]-3σ, b[2]-3σ))
    top::Int64 = min(size(img,2)-1, max(a[2]+3σ, b[2]+3σ))

    L = b .- a
    nL = hypot(L...)
    if nL == 0.0
        return
    end

    LnL = L ./ nL^2
    k = nL/(2σ)
    f = -1/(4σ^2)
    p = 1/(2*√(π)*σ*nL)

    for i=left:right
        for j=bottom:top
            x = (i,j)
            xa = x .- a
            M = sum(LnL .* xa)
            pre = p*exp(f*sum(xa .* (xa .- M .* L)))
            nkM = -k*M
            #img[i,j] += pre*erf(nkM, k + nkM)
            img[i,j] += pre*(erf(k + nkM) - erf(nkM))
        end
    end
end
function addpheromone!(p::Pheromone, oldpos::NTuple{2,Float64}, pos::NTuple{2,Float64})
    xn, yn = pos
    n = (trunc(Int64, xn / p.Δx) + 2, trunc(Int64, yn / p.Δx) + 2)
    xo, yo = oldpos
    o = (trunc(Int64, xo / p.Δx) + 2, trunc(Int64, yo / p.Δx) + 2)
    drawgaussianline!(p.amount[1], o, n, 2.0)
end

function steppheromone!(p::Pheromone, time::Float64)
    while p.time < time - p.Δt
        steppheromone!(p)
    end
end
function steppheromone!(p::Pheromone)
    A = p.amount[1]
    B = p.amount[2]
    for j=2:size(A,2)-1
        for i=2:size(A,1)-1
            @inbounds B[i,j] = 0.1 * (A[i-1,j]+A[i+1,j] + A[i,j-1]+A[i,j+1]) + 0.6*A[i,j]
        end
    end
    p.amount[1] = B
    p.amount[2] = A
    p.time += p.Δt
end

function localgradient(p::Pheromone, pos::NTuple{2,Float64}, radius::Float64)::NTuple{2,Float64}
    x, y = pos
    N = 0
    s = (0.0, 0.0)
    for xs=x-radius:p.Δx:x+radius
        for ys=y-radius:p.Δx:y+radius
            s = s .+ localgradient(p, (xs, ys))
            N += 1
        end
    end
    return s ./ N
end
function localgradient(p::Pheromone, pos::NTuple{2,Float64})::NTuple{2,Float64}
    x, y = pos
    il = floor(Int64, x / p.Δx) + 2
    jl = floor(Int64, y / p.Δx) + 2
    iu = ceil(Int64, x / p.Δx) + 2
    ju = ceil(Int64, y / p.Δx) + 2

    xr = x/p.Δx - il + 2
    yr = y/p.Δx - jl + 2

    # Bilinear interpolation
    a00 = localgradient(p, il, jl)
    a01 = localgradient(p, il, jl+1)
    a11 = localgradient(p, il+1, jl+1)
    a10 = localgradient(p, il+1, jl)

    dx = (a00[1]*(1-yr) + a01[1]*yr)*(1-xr) + (a10[1]*(1-yr) + a11[1]*yr)*xr
    dy = (a00[2]*(1-yr) + a01[2]*yr)*(1-xr) + (a10[2]*(1-yr) + a11[2]*yr)*xr

    return (dx, dy)
end
function localgradient(p::Pheromone, i::Int64, j::Int64)::NTuple{2,Float64}
    c = p.amount[1]

    if (i < 2) | (j < 2) | (i > size(c,1)-1) | (j > size(c,2)-1)
        return (0.0, 0.0)
    end

    dx = ((c[i+1, j+1] + 4*c[i+1, j] + c[i+1, j-1]) -
          (c[i-1, j+1] + 4*c[i-1, j] + c[i-1, j-1]))/(p.Δx*12.0)
    dy = ((c[i+1, j+1] + 4*c[i, j+1] + c[i-1, j+1]) -
          (c[i+1, j-1] + 4*c[i, j-1] + c[i-1, j-1]))/(p.Δx*12.0)

    return (dx, dy)
end
function Base.getindex(p::Pheromone, pos::NTuple{2,Float64})::Float64
    c = p.amount[1]
    x, y = pos
    i = trunc(Int64, x / p.Δx) + 2
    j = trunc(Int64, y / p.Δx) + 2
    if (i < 2) | (j < 2) | (i > size(c,1)-1) | (j > size(c,2)-1)
        return 0.0
    end

    return c[i,j]
end

"""
    pheromone_interaction(ant, model)

Calculates the gradient force and the positional attraction due to pheromone trails.
The type of interaction is determined by `T` in `Ant{T}` and can be `:perna`, `:gradient`,
`:gradient_nonlocal` or `:gradient_attraction`.
"""
# Pheromone interaction model according to Perna et al
function pheromone_model3(state::Vector{Float64}, ph::Pheromone)::Tuple{Float64,NTuple{2,Float64}}
    pos = (state[1], state[2])
    θ = state[3]
    dir = (cos(θ), sin(θ))
    perp = (-dir[2], dir[1])
    λ = ph.interaction_parameters[1]
    ρ = ph.interaction_parameters[2]

    L = ph[pos .+ (λ .* dir) .+ (ρ .* perp)]
    R = ph[pos .+ (λ .* dir) .- (ρ .* perp)]
    if L+R > 0.0
        gradient_force = (L-R)/(L+R)
    else
        gradient_force = 0.0
    end
    return (gradient_force, (0.0, 0.0))
end
# Pheromone interaction model - takes the gradient at the ant position only
function pheromone_model1(state::Vector{Float64}, ph::Pheromone)::Tuple{Float64,NTuple{2,Float64}}
    pos = (state[1], state[2])
    θ = state[3]
    localgrad = localgradient(ph, pos)
    n = (-sin(θ), cos(θ))
    gradient_force = sum(n .* localgrad)
    return (gradient_force, (0.0, 0.0))
end
# Pheromone interaction model - takes the gradient half an ant length in front of the ant
function pheromone_model4(state::Vector{Float64}, ph::Pheromone)::Tuple{Float64,NTuple{2,Float64}}
    pos = (state[1], state[2])
    θ = state[3]
    dir = (cos(θ), sin(θ))
    perp = (-dir[2], dir[1])
    λ = ph.interaction_parameters[1]

    epos = pos .+ λ .* (cos(θ), sin(θ))
    conc = ph[epos]
    if conc > 1e-8
        localgrad = localgradient(ph, epos)
        gradient_force = sum(perp .* localgrad) / conc
    else
        gradient_force = 0.0
    end
    return (gradient_force, (0.0, 0.0))
end
# Pheromone interaction model - takes the gradient and adds an positional attraction force
function pheromone_model2(state::Vector{Float64}, ph::Pheromone)::Tuple{Float64,NTuple{2,Float64}}
    pos = (state[1], state[2])
    θ = state[3]
    dir = (cos(θ), sin(θ))

    localgrad = localgradient(ph, pos)
    gradient_force = -sum(dir .* localgrad)
    return (gradient_force, localgrad)
end

function Pheromone_model1(width::Float64, height::Float64, D::Float64, Δx::Float64)
    Pheromone(width, height, D, Δx, pheromone_model1, [])
end

function Pheromone_model2(width::Float64, height::Float64, D::Float64, Δx::Float64)
    Pheromone(width, height, D, Δx, pheromone_model2, [])
end

function Pheromone_model3(width::Float64, height::Float64, D::Float64, Δx::Float64, λ::Float64, ρ::Float64)
    Pheromone(width, height, D, Δx, pheromone_model3, [λ, ρ])
end

function Pheromone_model4(width::Float64, height::Float64, D::Float64, Δx::Float64, λ::Float64)
    Pheromone(width, height, D, Δx, pheromone_model4, [λ])
end

function pheromone_interaction(state::Vector{Float64}, ph::Pheromone)::Tuple{Float64,NTuple{2,Float64}}
    ph.interaction_model(state, ph)
end
