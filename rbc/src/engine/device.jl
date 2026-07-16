# Compute-device selection (CPU / NVIDIA CUDA / Apple Metal).
#
# The whole NN stack — network weights, training batches, loss — is uniformly
# Float32 on every device: Apple GPUs have no Float64 at all, NVIDIA consumer
# cards run Float64 at 1/64 rate, and Float32 precision is far below typical
# converged residual losses (~1e-4), so nothing scientific is lost. Keep the
# precision-sensitive numerics (grid benchmarks, simulated state paths) in
# Float64 on the model side, casting to Float32 only at the network-input
# boundary.

"""
    select_device(preference=:auto)

Pick the compute device for NN training. `:auto` returns the first functional
GPU backend whose trigger package is loaded (CUDA on NVIDIA, Metal on Apple —
see the conditional loading in `FullRBC.jl`), falling back to the CPU;
`:cpu` forces the CPU; `:gpu` errors if no functional GPU is found.
"""
function select_device(preference::Symbol=:auto)
    preference === :cpu && return cpu_device()
    dev = gpu_device()
    if preference === :gpu && dev isa typeof(cpu_device())
        error("device=:gpu requested but no functional GPU backend is available. " *
              "Install/load CUDA.jl (NVIDIA) or Metal.jl (Apple) and check `gpu_device()`.")
    end
    return dev
end

"True if `device` is the CPU device (as returned by `cpu_device()` / fallback)."
is_cpu_device(device) = device isa typeof(cpu_device())

"""
    default_batch_size(device)

2048 on the CPU; 32768 on a GPU. The policy networks are tiny, so a GPU is
kernel-launch-bound at small batches — it needs wide batches to have enough
work per launch. The training loss is an expectation over uniform draws, so a
larger batch is statistically free (it only lowers gradient noise); budget
accordingly: at 16x the batch, each epoch sees 16x the samples, so fewer
epochs are needed.
"""
default_batch_size(device) = is_cpu_device(device) ? 2048 : 32_768
