
using DocStringExtensions
using Core: @doc
using LinearAlgebra, SparseArrays
using TensorOperations

export ⊗
⊗ = kron
######################################################
# Matrix Definitions for Fixed Single System Operators
######################################################

raise!(m, v, i) = setindex!(m, v*√i, i+1, i)
lower!(m, v, i) = setindex!(m, v*√(i-1), i-1, i)
setdiag!(m, v, i) = setindex!(m, v, i, i)

"""
Matrix Representation for the Bosonic Raising/Creation Operator `exp(i2πϕ) * a†`.

    $(TYPEDSIGNATURES)
"""
function raising(dim::Integer, ϕ=0.0, scale::T=1.0) where T
    m = spzeros(float(complex(T)), dim, dim)
    v = scale * exp(2π * im * ϕ)
    foreach(i -> raise!(m, v, i), 1:dim-1)
    m
end


"""
Matrix Representation for the Bosonic Lowering/Annhilation Operator. `exp(-i2πϕ) * a†`.

    $(TYPEDSIGNATURES)
"""
function lowering(dim::Integer, ϕ=0.0, scale::T=1.0) where T
    m = spzeros(float(complex(T)), dim, dim)
    v = scale * exp(-2π * im * ϕ)
    foreach(i -> lower!(m, v, i), 2:dim)
    m
end

"""
Matrix Representation for the Bosonic Number Operator `a†a`.

    $(TYPEDSIGNATURES)

"""
function number(dim::Integer, scale::T=1.0, offset=0.0) where T
    m = spzeros(float(complex(T)), dim, dim)
    foreach(i -> setdiag!(m, scale*(i-1-offset), i), 1:dim)
    m
end

"""
Matrix Representation for the Canonical Position Operator `X = a† + a`.

    $(TYPEDSIGNATURES)
"""
function X(dim::Integer, ϕ=0.0, scale=1.0)
    m = raising(dim, ϕ, scale)
    foreach(i->setindex!(m, m[i, i-1]', i-1, i), 2:dim)
    m
end

"""
Matrix Representation for the Canonical Conjugate Momentum Operator `Y = im(a† - a)`.

    $(TYPEDSIGNATURES)

"""
function Y(dim::Integer, ϕ=0.0, scale=1.0)
    m = raising(dim, ϕ, im * scale)
    foreach(i->setindex!(m, m[i, i-1]', i-1, i), 2:dim)
    m
end


######################################################
## Two Body Operators X_Y, XY, flip_flop
######################################################

"""
⊗X + ⊗Y operator.

    $(TYPEDSIGNATURES)

Useful for implementing couplings of type X1X2 + Y1Y2
"""
X_Y(dvec::AbstractVector{<:Number}, ϕvec::AbstractVector{<:Number}=[0,0]) = X(dvec, ϕvec) + Y(dvec, ϕvec)

"""
Bilinear photon exchange Hamiltonian on `a` and `b`: `ab† + a†b` with an additional relative phase.

    $(TYPEDSIGNATURES)

Apply an additional relative phase ϕ: `exp(2πiϕ)ab† + exp(-2πiϕ)a†b`.
"""
function flip_flop(dim1::Integer, dim2::Integer, ϕ=0.0, scale::T=1.0) where T
    d = dim1*dim2
    m = spzeros(float(complex(T)), d, d)
    @inbounds kron!(m, raising(dim1), lowering(dim2, ϕ, scale))
    m += adjoint(m) 
end


"""
$(TYPEDSIGNATURES)
Apply an additional phase rotation to the XY Hamiltonian 2*(exp(2πiϕ)ab† + exp(-2πiϕ)a†b)
"""
XY(dim1::Integer, dim2::Integer, ϕ::Real=0) = flip_flop(dim1, dim2, ϕ, 2.0)

######################################################
## Many Body Operators
######################################################


for op in [:raising, :lowering, :X, :Y]
    quote
        """
        Multi-Qsystem Bosonic Operator.
            $(TYPEDSIGNATURES)
        """
        function $(op)(dims::AbstractVector{<:Number}, ϕs::AbstractVector{<:Number})
            mapreduce(x -> $(op)(x...), ⊗, zip(dims, ϕs))
        end
    end |> eval
end