using Crayons

export
    AbstractState,
    StateVector,
    StateMatrix,

    𝛒,
    purity

abstract type AbstractState end

################
# state vector #
################

mutable struct StateVector{T <: Number} <: AbstractState
    v::Vector{T}
    dim::Int64
end

function Base.show(io::IO, state::StateVector{T}) where {T}
    print(io, "StateVector{$T}( ")
    v = abs2.(state.v)
    v /= maximum(v)
    for p in v
        c = convert(RGB, HSL(0, p, 0.7))
        print(io, "$(Crayon(foreground=(
            round(Int, c.r * 255), round(Int, c.g * 255), round(Int, c.b * 255)
        )))\u2587")
    end
    print(io, "$(Crayon(reset=true)) )")
end

Base.vec(state::StateVector{<:Number}) = state.v

𝛒(state::StateVector{<:Number}) = state.v * state.v'

function purity(state::StateVector{<:Number})
    𝛒 = state.v * state.v'
    𝛒 /= tr(𝛒)

    return real(tr(𝛒^2))
end

function Base.copy(state::StateVector{T}) where {T<:Number}
    return StateVector{T}(copy(state.v), state.dim)
end

################
# state matrix #
################

mutable struct StateMatrix{T <: Number} <: AbstractState
    𝛒::Matrix{T}
    dim::Int64
end

function Base.show(io::IO, state::StateMatrix{T}) where {T}
    function show_𝛒(𝛒::Matrix{<:Real})
        for (i, p) in enumerate(𝛒)
            c = (p>0) ? convert(RGB, HSL(0, p, 0.7)) : convert(RGB, HSL(240, abs(p), 0.7))
            print(io, "$(Crayon(foreground=(
                round(Int, c.r * 255), round(Int, c.g * 255), round(Int, c.b * 255)
            )))\u2587")
            (i%state.dim == 0) && println(io)
        end
    end

    println(io, "StateMatrix{$T}(")
    𝛒_r = real(state.𝛒)
    𝛒_r /= maximum(abs.(𝛒_r))
    show_𝛒(𝛒_r)
    print(io, "$(Crayon(reset=true)))")
end

function StateMatrix(state::StateVector{T}) where {T <: Number}
    𝛒 = state.v * state.v'
    return StateMatrix{T}(𝛒, state.dim)
end

𝛒(state::StateMatrix{<:Number}) = state.𝛒

function purity(state::StateMatrix{<:Number})
    𝛒 = state.𝛒
    𝛒 /= tr(𝛒)

    return real(tr(𝛒^2))
end

function Base.copy(state::StateMatrix{T}) where {T<:Number}
    return StateMatrix{T}(copy(state.𝛒), state.dim)
end
