using KernelDensity

@testset "pdf and Gaussian state data generator" begin
    state = SqueezedThermalState(ξ(1., π/4), 0.5)
    θs = LinRange(0, 2π, 10)
    xs = LinRange(-10, 10, 10)

    ground_truth_pdf = SqState.pdf(state, θs, xs)

    single_point_pdf = (θ, x) -> SqState.pdf(state, θ, x)
    @test single_point_pdf.(θs, xs') ≈ ground_truth_pdf

    n = 100000
    data = gen_gaussian_training_data(state, n)
    sampled_pdf = KernelDensity.pdf(
        kde((LinRange(0, 2π, n), data)),
        θs, xs
    )

    @test sum(abs.(sampled_pdf .- ground_truth_pdf)) / n < 1e-4
end

@testset "nongaussian util" begin
    range = (0., 2π)
    n = 1000000
    @test isapprox(sum(SqState.ranged_rand(n, range))/n, π, atol=1e-2)
end

@testset "pdf and non-Gaussian state data generator" begin
    state = displace!(
        squeeze!(
            SinglePhotonState(rep=StateMatrix, dim=100),
            ξ(0.5, π/2)
        ),
        α(3., π/2)
    )
    θs = LinRange(0, 2π, 10)
    xs = LinRange(-10, 10, 10)

    ground_truth_pdf = SqState.pdf(state, θs, xs)

    single_point_pdf = (θ, x) -> SqState.pdf(state, θ, x)
    @test single_point_pdf.(θs, xs') ≈ ground_truth_pdf

    n = 4096
    @info "gen non-gaussian data"
    @time data = gen_nongaussian_training_data(state; n=n, batch_size=64, show_log=false)
    sampled_pdf = KernelDensity.pdf(kde((data[1, :], data[2, :])), θs, xs)

    @test sum(abs.(sampled_pdf .- ground_truth_pdf)) / n  < 5e-2
end
