using SpecialFunctions

mutable struct Pheromone{ModelType}
    amount::Vector{Array{Float64,2}}
    time::Float64
    D::Float64
    Δx::Float64
    Δt::Float64
    η::Float64
    interaction_parameters::Vector{Float64}
    function Pheromone{ModelType}(
            width::Float64,
            height::Float64,
            D::Float64,
            Δx::Float64,
            η::Float64,
            interaction_parameters::Vector{Float64}) where {ModelType}
        Nw = trunc(Int64, width/Δx) + 4
        Nh = trunc(Int64, height/Δx) + 4
        Δt = 0.1Δx^2/D # One fifth of stability condition
        new([zeros(Float64,Nw,Nh), zeros(Float64,Nw,Nh)], 0.0, D, Δx, Δt, η, interaction_parameters)
    end
end

####################
# For testing only!!
function Pheromone{:testing}(width::Float64, height::Float64, D::Float64, Δx::Float64, η::Float64, params::Vector{Float64})
    ph = Pheromone{:testing}(width, height, 0.0, Δx, η, Float64[])
    Nw = size(ph.amount[1],1)
    Nh = size(ph.amount[1],2)
    for s=16:15:Nw
        drawgaussianline!(ph.amount[1], (s-15, (Nh-3)÷2-2), (s, (Nh-3)÷2-2), η)
        drawgaussianline!(ph.amount[2], (s-15, (Nh-3)÷2-2), (s, (Nh-3)÷2-2), η)
    end
    ph
end
function pheromone_interaction(pos::NTuple{2,Float64}, θ::Float64, ph::Pheromone{:testing})::Tuple{Float64,NTuple{2,Float64}}
    return (0.0, (0.0, 0.0))
end
####################


function drawgaussianline!(img::Array{Float64,2}, a::NTuple{2,Int64}, b::NTuple{2,Int64}, σ::Float64)
    Δ::Int64 = ceil(Int64, 6σ)
    left::Int64 = max(2, min(a[1]-Δ, b[1]-Δ))
    right::Int64 = min(size(img,1)-1, max(a[1]+Δ, b[1]+Δ))
    bottom::Int64 = max(2, min(a[2]-Δ, b[2]-Δ))
    top::Int64 = min(size(img,2)-1, max(a[2]+Δ, b[2]+Δ))

    L = b .- a
    nL = hypot(L...)
    if nL == 0.0
        return
    end

    LnL = L ./ nL^2
    k = nL/(2σ)
    f = -1/(2σ^2)
    #p = 1/(2*√(π)*σ*nL)
    p = 1/(2*√(π)*σ)

    for j=bottom:top
        for i=left:right
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
function addpheromone!(p::Pheromone{<:Any}, oldpos::SVector{2,Float64}, pos::SVector{2,Float64})
    xn, yn = pos
    n = (trunc(Int64, xn / p.Δx) + 2, trunc(Int64, yn / p.Δx) + 2)
    xo, yo = oldpos
    o = (trunc(Int64, xo / p.Δx) + 2, trunc(Int64, yo / p.Δx) + 2)
    drawgaussianline!(p.amount[1], o, n, p.η / p.Δx)
end

function steppheromone!(p::Pheromone{<:Any}, time::Float64)
    while p.time < time - p.Δt
        steppheromone!(p)
    end
end

function steppheromone!(p::Pheromone{:nopheromone})
    # Noop
end
function steppheromone!(p::Pheromone{<:Any})
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

function localgradient(p::Pheromone{<:Any}, pos::SVector{2,Float64}, radius::Float64)::SVector{2,Float64}
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
function bilinear_interp(::Type{T}, f::Function, pos::SVector{2,Float64}, Δx::Float64)::T where {T}
    x, y = pos
    il = floor(Int64, x / Δx) + 2
    jl = floor(Int64, y / Δx) + 2

    xr = x/Δx - il + 2
    yr = y/Δx - jl + 2

    # Bilinear interpolation
    a00::T = f(il, jl)
    a01::T = f(il, jl+1)
    a11::T = f(il+1, jl+1)
    a10::T = f(il+1, jl)

    v::T = @. ((a00 * (1-yr) + a01*yr)*(1-xr) + (a10*(1-yr) + a11*yr)*xr)

    return v
end
function localgradient(p::Pheromone{<:Any}, pos::SVector{2,Float64})::SVector{2,Float64}
    return bilinear_interp(SVector{2,Float64}, (i::Int64, j::Int64)->localgradient(p, i, j), pos, p.Δx)
    # x, y = pos
    # i = trunc(Int64, x / p.Δx) + 2
    # j = trunc(Int64, y / p.Δx) + 2

    # return localgradient(p, i, j)
end
# function localgradient(p::Pheromone{<:Any}, pos::NTuple{2,Float64})::NTuple{2,Float64}
#     x, y = pos
#     il = floor(Int64, x / p.Δx) + 2
#     jl = floor(Int64, y / p.Δx) + 2
#     iu = ceil(Int64, x / p.Δx) + 2
#     ju = ceil(Int64, y / p.Δx) + 2

#     xr = x/p.Δx - il + 2
#     yr = y/p.Δx - jl + 2

#     # Bilinear interpolation
#     a00 = localgradient(p, il, jl)
#     a01 = localgradient(p, il, jl+1)
#     a11 = localgradient(p, il+1, jl+1)
#     a10 = localgradient(p, il+1, jl)

#     dx = (a00[1]*(1-yr) + a01[1]*yr)*(1-xr) + (a10[1]*(1-yr) + a11[1]*yr)*xr
#     dy = (a00[2]*(1-yr) + a01[2]*yr)*(1-xr) + (a10[2]*(1-yr) + a11[2]*yr)*xr

#     return (dx, dy)
# end
function localgradient(p::Pheromone{<:Any}, i::Int64, j::Int64)::SVector{2,Float64}
    c = p.amount[1]

    if (i < 2) | (j < 2) | (i > size(c,1)-1) | (j > size(c,2)-1)
        return (0.0, 0.0)
    end

    dx = ((c[i+1, j+1] + 4*c[i+1, j] + c[i+1, j-1]) -
          (c[i-1, j+1] + 4*c[i-1, j] + c[i-1, j-1]))/(p.Δx*12.0)
    dy = ((c[i+1, j+1] + 4*c[i, j+1] + c[i-1, j+1]) -
          (c[i+1, j-1] + 4*c[i, j-1] + c[i-1, j-1]))/(p.Δx*12.0)

    return SVector(dx, dy)
end
function Base.getindex(p::Pheromone{<:Any}, pos::SVector{2,Float64})::Float64
    c = p.amount[1]
    function access(i::Int64, j::Int64)::Float64
        if (i < 2) | (j < 2) | (i > size(c,1)-1) | (j > size(c,2)-1)
            return 0.0
        end
        return c[i,j]
    end
    return bilinear_interp(Float64, access, pos, p.Δx)
    # x, y = pos
    # i = trunc(Int64, x / p.Δx) + 2
    # j = trunc(Int64, y / p.Δx) + 2
    # if (i < 2) | (j < 2) | (i > size(c,1)-1) | (j > size(c,2)-1)
    #     return 0.0
    # end

    # return c[i,j]
end

"""
    pheromone_interaction(ant, model)

Calculates the gradient force and the positional attraction due to pheromone trails.
The type of interaction is determined by `T` in `Ant{T}` and can be `:perna`, `:gradient`,
`:gradient_nonlocal` or `:gradient_attraction`.
"""
# Pheromone interaction model according to Perna et al
function pheromone_interaction(pos::SVector{2,Float64}, θ::Float64, ph::Pheromone{:perna})::Tuple{Float64,SVector{2,Float64}}
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
    return (gradient_force, SVector(0.0, 0.0))
end
# Pheromone interaction model - takes the gradient at the ant position only
function pheromone_interaction(pos::SVector{2,Float64}, θ::Float64, ph::Pheromone{:gradient})::Tuple{Float64,SVector{2,Float64}}
    localgrad = localgradient(ph, pos)
    n = (-sin(θ), cos(θ))
    gradient_force = sum(n .* localgrad)
    return (gradient_force, SVector(0.0, 0.0))
end
# Pheromone interaction model - takes the gradient half an ant length in front of the ant
function pheromone_interaction(pos::SVector{2,Float64}, θ::Float64, ph::Pheromone{:gradient_nonlocal})::Tuple{Float64,SVector{2,Float64}}
    dir = (cos(θ), sin(θ))
    perp = (-dir[2], dir[1])
    λ = ph.interaction_parameters[1]

    epos = pos .+ λ .* (cos(θ), sin(θ))
    conc = ph[epos]
    if conc > eps(1.0)
        localgrad = localgradient(ph, epos)
        gradient_force = sum(perp .* localgrad) / conc
    else
        gradient_force = 0.0
    end
    return (gradient_force, SVector(0.0, 0.0))
end
# Pheromone interaction model - takes the gradient and adds an positional attraction force
function pheromone_interaction(pos::SVector{2,Float64}, θ::Float64, ph::Pheromone{:gradient_attraction})::Tuple{Float64,SVector{2,Float64}}
    dir = (cos(θ), sin(θ))

    localgrad = localgradient(ph, pos)
    gradient_force = -sum(dir .* localgrad)
    return (gradient_force, localgrad)
end
# No pheromone interaction model
function pheromone_interaction(pos::SVector{2,Float64}, θ::Float64, ph::Pheromone{:nopheromone})::Tuple{Float64,SVector{2,Float64}}
    return (0.0, SVector(0.0, 0.0))
end
