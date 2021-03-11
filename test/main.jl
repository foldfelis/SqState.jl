using SqState

function main()
    ########################
    # init wigner function #
    ########################
    x_range = -10:0.1:10
    p_range = -10:0.1:10
    @info "Initialising"
    start_time = time()
    wf = WignerFunction(x_range, p_range)
    end_time = time()
    @info "Done, took $(end_time - start_time)(s)"

    ##########
    # render #
    ##########
    data_path = joinpath(SqState.PROJECT_PATH, "../data", "dm.hdf5")
    data_name = "SQ4"
    ρ = read_ρ(data_path, data_name)
    w = wf(ρ)

    ########
    # plot #
    ########
    file_path = joinpath(SqState.PROJECT_PATH, "../data/render", "density_matrix_total.png")
    p = plot_ρ(ρ, file_path=file_path)
    file_path = joinpath(SqState.PROJECT_PATH, "../data/render", "density_matrix.png")
    p = plot_ρ(ρ, state_n=5, file_path=file_path)

    file_path = joinpath(SqState.PROJECT_PATH, "../data/render", "wigner_contour.png")
    p = plot_wigner(wf, w, Contour, file_path=file_path)
    file_path = joinpath(SqState.PROJECT_PATH, "../data/render", "wigner_heatmap.png")
    p = plot_wigner(wf, w, Heatmap, file_path=file_path)
    file_path = joinpath(SqState.PROJECT_PATH, "../data/render", "wigner_surface.png")
    p = plot_wigner(wf, w, Surface, file_path=file_path)
    file_path = joinpath(SqState.PROJECT_PATH, "../data/render", "wigner_surface_banner.png")
    p = plot_wigner(wf, w, Surface, size=(1280, 640), file_path=file_path)

    return p
end

main()
