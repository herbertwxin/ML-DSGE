# Gauss-Hermite quadrature for expectations over normal innovations, shared
# by the NN training loss and any grid benchmark.

"""
    Quadrature(n)

Gauss-Hermite nodes/weights rescaled for expectations over a standard-normal
innovation: `E[f(eps)] ≈ sum(w .* f.(nodes))`.
"""
struct Quadrature
    nodes::Vector{Float64}
    weights::Vector{Float64}
end

function Quadrature(n::Int)
    x, w = gausshermite(n)
    return Quadrature(x .* sqrt(2), w ./ sqrt(pi))
end
