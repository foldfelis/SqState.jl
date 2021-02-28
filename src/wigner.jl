export
    wigner,
    WignerFunction

#=
    Wigner function by Laguerre Polynominal
=#

function wigner(m::Integer, n::Integer, x::Real, p::Real)
    w = gaussian_function(x, p)
    w *= coefficient_of_wave_function(m, n)
    w *= z_to_power(m, n, x, p)
    w *= laguerre(m, n, x, p)

    return w
end

wigner(m::Integer, n::Integer) = (x, p)->wigner(m, n, x, p)

function create_wigner(m_dim::Integer, n_dim::Integer, xs, ps)
    W = Array{ComplexF64,4}(undef, m_dim, n_dim, length(xs), length(ps))
    Threads.@threads for m = 1:m_dim
        Threads.@threads for n = 1:n_dim
            Threads.@threads for (x_i, x) = collect(enumerate(xs))
                Threads.@threads for (p_j, p) = collect(enumerate(ps))
                    W[m, n, x_i, p_j] = wigner(m ,n, x, p)
                end
            end
        end
    end
    return W
end

mutable struct WignerFunction{T<:Integer}
    m_dim::T
    n_dim::T
    xs
    ps
    W::Array{ComplexF64,4}

    function WignerFunction(m_dim::T, n_dim::T, xs, ps) where {T<:Integer}
        if check_zero(m_dim, n_dim) && check_empty(xs, ps)
            W = create_wigner(m_dim, n_dim, xs, ps)
        else
            W = Array{ComplexF64,4}(undef, 0, 0, 0, 0)
        end
        new{T}(m_dim, n_dim, xs, ps, W)
    end
end

function WignerFunction(m_dim::T, n_dim::T) where {T<:Integer}
    return WignerFunction(m_dim, n_dim, [], [])
end

function WignerFunction(xs::Vector, ps::Vector)
    return WignerFunction(0, 0, xs, ps)
end

function WignerFunction(xs::StepRangeLen, ps::StepRangeLen; ρ_size=35)
    return WignerFunction(ρ_size, ρ_size, xs, ps)
end

function (wf::WignerFunction)(ρ::AbstractMatrix)
    reshape(real(sum(ρ .* wf.W, dims=(1, 2))), length(wf.xs), length(wf.ps))
end

function Base.setproperty!(wf::WignerFunction, name::Symbol, x)
    setfield!(wf, name, x)
    m_dim = getproperty(wf, :m_dim)
    n_dim = getproperty(wf, :n_dim)
    xs = getproperty(wf, :xs)
    ps = getproperty(wf, :ps)
    if check_zero(m_dim, n_dim) && check_empty(xs, ps)
        W = create_wigner(m_dim, n_dim, xs, ps)
        setfield!(wf, :W, W)
    end
end

check_zero(m_dim, n_dim) = !iszero(m_dim) && !iszero(n_dim)

check_empty(xs, ps) = !isempty(xs) && !isempty(ps)
