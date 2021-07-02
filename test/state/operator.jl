@testset "a† and a" begin
    dim = 70

    @test create!(VacuumState(dim=dim)) ≈ SinglePhotonState(dim=dim)
    @test annihilate!(SinglePhotonState(dim=dim)) ≈ VacuumState(dim=dim)
    @test create!(VacuumState(dim=dim, rep=StateMatrix)) ≈ SinglePhotonState(dim=dim, rep=StateMatrix)
    @test annihilate!(SinglePhotonState(dim=dim, rep=StateMatrix)) ≈ VacuumState(dim=dim, rep=StateMatrix)

    @test create(VacuumState(dim=dim)) ≈ SinglePhotonState(dim=dim)
    @test annihilate(SinglePhotonState(dim=dim)) ≈ VacuumState(dim=dim)
    @test create(VacuumState(dim=dim, rep=StateMatrix)) ≈ SinglePhotonState(dim=dim, rep=StateMatrix)
    @test annihilate(SinglePhotonState(dim=dim, rep=StateMatrix)) ≈ VacuumState(dim=dim, rep=StateMatrix)
end

@testset "α and ξ" begin
    @test repr(Arg(2., π/4)) == "Arg{Float64}(2.0exp($(π/4)im))"
    @test SqState.z(α(2., π/4)) ≈ 2 * exp(im * π/4)
    @test SqState.z(ξ(2., π/4)) ≈ 2 * exp(im * π/4)
end

@testset "Displacement" begin
    dim = 70
    r = 2.
    θ = π/4

    @test displace!(VacuumState(dim=dim), α(r, θ)).v ≈ exp(
        SqState.z(α(r, θ)) * Creation(dim=dim) -
        SqState.z(α(r, θ))' * Annihilation(dim=dim)
    ) * VacuumState().v
    @test displace!(VacuumState(dim=dim, rep=StateMatrix), α(r, θ)).𝛒 ≈ exp(
        SqState.z(α(r, θ)) * Creation(dim=dim) -
        SqState.z(α(r, θ))' * Annihilation(dim=dim)
    ) * VacuumState(rep=StateMatrix).𝛒 * exp(
        SqState.z(α(r, θ)) * Creation(dim=dim) -
        SqState.z(α(r, θ))' * Annihilation(dim=dim)
    )'
end

@testset "squeezing" begin
    dim = 70
    r = 2.
    θ = π/4

    @test squeeze!(VacuumState(dim=dim), α(r, θ)).v ≈ exp(
        0.5 * SqState.z(ξ(r, θ))' * Annihilation(dim=dim)^2 -
        0.5 * SqState.z(ξ(r, θ)) * Creation(dim=dim)^2
    ) * VacuumState().v
    @test squeeze!(VacuumState(dim=dim, rep=StateMatrix), α(r, θ)).𝛒 ≈ exp(
        0.5 * SqState.z(ξ(r, θ))' * Annihilation(dim=dim)^2 -
        0.5 * SqState.z(ξ(r, θ)) * Creation(dim=dim)^2
    ) * VacuumState(rep=StateMatrix).𝛒 * exp(
        0.5 * SqState.z(ξ(r, θ))' * Annihilation(dim=dim)^2 -
        0.5 * SqState.z(ξ(r, θ)) * Creation(dim=dim)^2
    )'
end
