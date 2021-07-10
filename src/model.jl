export training_process

function conv_layers(ch::NTuple{4, <:Integer}, kernel_size::NTuple{3, <:Integer}, pad::NTuple{3, <:Any})
    return Chain(
        Conv((kernel_size[1], ), ch[1]=>ch[2], pad=pad[1]),
        BatchNorm(ch[2], relu),
        Conv((kernel_size[2], ), ch[2]=>ch[3], pad=pad[2]),
        BatchNorm(ch[3], relu),
        Conv((kernel_size[3], ), ch[3]=>ch[4], pad=pad[3]),
        BatchNorm(ch[4]),
    )
end

function short_cut(ch::NTuple{2, <:Integer}, kernel_size::Integer, pad::Any)
    return Chain(
        Conv((kernel_size, ), ch[1]=>ch[2], pad=pad),
        BatchNorm(ch[2])
    )
end

function model(; dim=70)
    return Chain(
        # stage 0
        BatchNorm(1),
        Conv((31, ), 1=>64, pad=15),
        BatchNorm(64, relu),
        # res 1
        Parallel(+,
            conv_layers((64, 32, 32, 96), (1, 15, 7), (0, 7, 3)),
            short_cut((64, 96), 1, 0),
        ),
        x -> relu.(x),
        MeanPool((2, )),
        BatchNorm(96, relu),
        # res 2
        SkipConnection(conv_layers((96, 32, 32, 96), (1, 7, 1), (0, 3, 0)), +),
        x -> relu.(x),
        # res 3
        SkipConnection(conv_layers((96, 32, 32, 96), (1, 7, 1), (0, 3, 0)), +),
        x -> relu.(x),
        MeanPool((4, )),
        BatchNorm(96, relu),
        # res 4
        Parallel(+,
            conv_layers((96, 64, 64, 128), (1, 3, 1), (0, 1, 0)),
            short_cut((96, 128), 1, 0),
        ),
        x -> relu.(x),
        # res 5
        SkipConnection(conv_layers((128, 64, 64, 128), (1, 3, 1), (0, 1, 0)), +),
        x -> relu.(x),
        # res 6
        SkipConnection(conv_layers((128, 64, 64, 128), (1, 3, 1), (0, 1, 0)), +),
        x -> relu.(x),
        MeanPool((8, )),
        BatchNorm(128, relu),
        # res 7
        Parallel(+,
            conv_layers((128, 96, 96, 196), (1, 3, 1), (0, 1, 0)),
            short_cut((128, 196), 1, 0),
        ),
        x -> relu.(x),
        # res 8
        SkipConnection(conv_layers((196, 96, 96, 196), (1, 3, 1), (0, 1, 0)), +),
        x -> relu.(x),
        # res 9
        SkipConnection(conv_layers((196, 96, 96, 196), (1, 3, 1), (0, 1, 0)), +),
        x -> relu.(x),
        # res 10
        SkipConnection(conv_layers((196, 96, 96, 196), (1, 3, 1), (0, 1, 0)), +),
        x -> relu.(x),
        # res 11
        SkipConnection(conv_layers((196, 96, 96, 196), (1, 3, 1), (0, 1, 0)), +),
        x -> relu.(x),
        MeanPool((2, )),
        BatchNorm(196, relu),
        # stage 1
        flatten,
        Dense(32*196, 5586, relu),
        Dense(5586, dim*dim),
        x -> x ./ sum(x.^2) # l-2 normalize
    )
end

function training_process(;
    file_names=readdir(SqState.training_data_path()),
    batch_size=100, epochs=10,
    is_gpu=true
)
    if CUDA.has_cuda() && is_gpu
        @info "CUDA is on"
        CUDA.allowscalar(false)
    end

    # prepare model
    m = is_gpu ? model() |> gpu : model()
    loss(x, y) = Flux.Losses.crossentropy(m(x).^2, y.^2)
    loss_mse(x, y) = Flux.mse(m(x), y)
    ps = Flux.params(m)
    opt = ADAM(1e-2, (0.7, 0.9))

    # jit model
    @time begin
        @info "jit..."
        x, y = first(preprocess(file_names[2], batch_size=1))
        x = is_gpu ? x |> gpu : x
        y = is_gpu ? y |> gpu : y
        gs = Flux.gradient(() -> loss(x, y), ps)
        Flux.update!(opt, ps, gs)
    end

    # prepare data
    test_data_loader = preprocess(file_names[1], batch_size=batch_size)
    @info "numbers of data fragments: $(length(file_names)-1)"
    data_loaders = Channel(5, spawn=true) do ch
        for e in 1:epochs
            for (i, file_name) in enumerate(file_names[2:end])
                put!(ch, preprocess(file_name, batch_size=batch_size))
                @info "Load epoch $(e), $(i)th files into buffer"
            end
        end
    end

    # training
    in_losses = Float32[]
    in_losses_mse = Float32[]
    out_losses = Float32[]
    out_losses_mse = Float32[]
    for (t, loader) in enumerate(data_loaders)
        @time for (b, (x, y)) in enumerate(loader)
            x = is_gpu ? x |> gpu : x
            y = is_gpu ? y |> gpu : y

            opt.eta = 1e-2 / 2^((length(loader)*(t-1)+b)/(2*length(loader)))
            gs = Flux.gradient(() -> loss(x, y), ps)
            Flux.update!(opt, ps, gs)

            push!(in_losses, loss(x, y))
            push!(out_losses, validation(test_data_loader, loss, is_gpu))
            push!(in_losses_mse, loss_mse(x, y))
            push!(out_losses_mse, validation(test_data_loader, loss_mse, is_gpu))
        end

        in_loss = sum(
            x->x/length(loader),
            in_losses[(end-length(loader)+1):end]
        )
        out_loss = sum(
            x->x/length(test_data_loader),
            out_losses[(end-length(loader)+1):end]
        )
        out_loss_mse = sum(
            x->x/length(test_data_loader),
            out_losses_mse[(end-length(loader)+1):end]
        )
        @info "$t\n# learning rate: $(opt.eta)\n# in data loss:  $in_loss\n# out data loss: $out_loss\n# mse out loss:  $(out_loss_mse)"
    end

    return model, in_losses, out_losses, in_losses_mse, out_losses_mse
end

function validation(test_data_loader::DataLoader, loss_func, is_gpu)
    x, y = first(test_data_loader)
    x = is_gpu ? x |> gpu : x
    y = is_gpu ? y |> gpu : y

    return loss_func(x, y)
end
