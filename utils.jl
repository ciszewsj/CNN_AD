function kaiming_normal_weights(n_input::Int, n_output::Int)
    stddev = sqrt(1 / n_input)
    weight = stddev .- rand(n_output, n_input) * 2 * stddev
    return permutedims(weight, (2, 1))
end

function create_kernel(n_input::Int64, n_output::Int64; kernel_size = 3)
    stddev = sqrt(1 / (n_input * 9))
    return stddev .- rand(kernel_size, kernel_size, n_input, n_output) * stddev * 2
end

function initialize_uniform_bias(in_features::Int64, out_features::Int64)
    k = sqrt(1 / in_features)
    return k .- 2 * rand(out_features) * k
end
