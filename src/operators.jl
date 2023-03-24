include("bosonic_operator_matrix_defs.jl")

export raising, lowering, number, X, Y, X_Y, XY, flip_flop,
    decay, dephasing, dipole_drive, parametric_drive, rwa_dipole

######################################################
# Operators Definitions for QSystem
######################################################

const _DOCSTRINGS = Dict(
    :raising => """
Bosonic Raising/Creation/Ladder-a† operator.

    $(TYPEDSIGNATURES)

Given an eignenstate of the number operator `|n⟩` on a
`QSystem q`, `raising(q)|n⟩ = √(n+1)|n+1⟩`. 

Optionally an additional phase exp(2πiϕ) is applied.

## Examples
```jldoctest
julia> q = DuffingTransmon("q", 3, DuffingSpec(5, -0.2))
DuffingTransmon("q", 3, DuffingSpec(5, -0.2))

julia> raising(q)
3×3 Array{Float64,2}:
    0.0  0.0      0.0
    1.0  0.0      0.0
    0.0  1.41421  0.0
```
""",
    :lowering => """
 Bosonic Lowering/Destruction/a ladder operator. 

     $(TYPEDSIGNATURES)

 Given an eignenstate of the number operator `|n⟩` on a
 `QSystem q`, `lowering(q)|n⟩ = √(n)|n-1⟩`. Optionally apply an additional phase ϕ and scaling `factor`.

 ## Examples
 ```jldoctest
 julia> q = DuffingTransmon("q", 3, DuffingSpec(5, -0.2))
 DuffingTransmon("q", 3, DuffingSpec(5, -0.2))

 julia> lowering(q)
 3×3 Array{Float64,2}:
     0.0  1.0  0.0
     0.0  0.0  1.41421
     0.0  0.0  0.0
 ```
 """,
    :number => """
 Bosonic Number Operator `a†a` on a QSystem `q`, with an optional overall scaling through `factor`.

     $(TYPEDSIGNATURES)

 Setting `factor != 1` amounts to adding dephasing to the system.
 TODO: ADD more information.
     """,
    :X => """
 Canonical Position Operator `X = a† + a` on a QSystem `q`. 

     $(TYPEDSIGNATURES)

 Optionally you  may apply a phase of ϕ (units of τ) to the X operator, and provide an overall `scale` factor.
 """,
    :Y => """
Canonical Conjugate Momentum Operator `Y = im(a† - a)` on a QSystem `q`. 
    
    $(TYPEDSIGNATURES)
    
Optionally you  may apply a phase of ϕ (units of τ) to the Y operator, and provide an overall `scale` factor.
    """
)

# """
# Apply a phase ϕ (units of τ) to an X operator

# # Examples
# ```jldoctest
# julia> q = DuffingTransmon("q", 2, DuffingSpec(5, -0.2))
# DuffingTransmon("q", 2, DuffingSpec(5, -0.2))

# julia> X(q, 0.25)
# 2×2 Array{Complex{Float64},2}:
#          0.0+0.0im  6.12323e-17-1.0im
#  6.12323e-17+1.0im          0.0+0.0im

# julia> X(q, 0.25) ≈ Y(q)
# true
# ```
# """

"""
Internal Macro to generate the single and many body bosonic operators for
"""

for op in [:raising :lowering :X :Y]
    expr = quote
        @doc $(_DOCSTRINGS[op])
        function $(op)(q::QSystem, ϕ=0.0, scale=1)
            $(op)(dimension(q), ϕ, scale)
        end

        """
        Multi-Qsystem Bosonic Operator.
            $(TYPEDSIGNATURES)
        """
        function $(op)(qs::AbstractVector{<:QSystem}, ϕs::AbstractVector)
            $(op)(dimension.(qs), ϕs)
        end
    end
    @eval $expr
end


"""
    $(TYPEDSIGNATURES)
"""
function number(q::QSystem, scale=1, offset=0)
    number(dimension(q), scale, offset)
end

"""
    $(TYPEDSIGNATURES)
"""
function number(qs::AbstractVector{<:QSystem}, scale=1, offset=1)
    dims = dimension.(qs)
    kron(broadcast(dims, offset, scale)...)
end


######################################################
## Two Body Operators X_Y, XY, flip_flop
######################################################
"""
    $(TYPEDSIGNATURES)
"""
X_Y(qs::AbstractVector{<:QSystem}, ϕs::AbstractVector{<:Number}=[0,0]) = X_Y(dimension.(qs), ϕs)


"""
Bilinear photon exchange Hamiltonian on `a` and `b`: `ab† + a†b`.

    $(TYPEDSIGNATURES)

For qubits i.e. (d=2) systems, this corresponds to Pauli operator `σ⁺σ⁻ + σ⁻σ⁺`
Bilinear photon exchange Hamiltonian on `a` and `b`: `ab† + a†b` with an additional relative phase.
Apply an additional relative phase ϕ: `exp(2πiϕ)ab† + exp(-2πiϕ)a†b`.
"""
flip_flop(a::QSystem, b::QSystem, ϕ::Real=0) = flip_flop(dimension(a), dimension(b), ϕ)


"""
    XY(a,b) = XᵃXᵇ + YᵃYᵇ = 2*(ab† + a†b)

    $(TYPEDSIGNATURES)

Bilinear XY Hamiltonian on `a` and `b` which is (up to a scale) equivalent to a "flip-flop"
Hamiltonian.
"""
XY(a::QSystem, b::QSystem, ϕ::Real=0) = XY(dimension(a), dimension(b), ϕ)


######################################################
## Parametric Operators
######################################################

"""
Apply a time dependent diple drive to the system.

    $(TYPEDSIGNATURES)

Given some function of time, return a function applying a time dependent
dipole Hamiltonian. Note that this does not use the rotating wave approximation
and therefore requires a real valued drive. See also `rwa_dipole`

## args
* `qs`: a QSystem.
* `drive`: a function of time returning a real value.
* `rotation_rate`: the rotation rate of a rotating frame.

Returns a univariate function of time.
"""
function dipole_drive(qs::QSystem, drive::Function, rotation_rate::Real=0.0)
    function ham(t)
        pulse::Real = drive(t)
        return pulse * X(qs, rotation_rate * t)
    end
    return ham
end

"""
Apply a time dependent diple drive hamiltonian to the system under the rotating wave approximation.

    $(TYPEDSIGNATURES)

Given some function of time, return a function applying a time dependent
dipole Hamiltonian under the rotating wave approximation (RWA).

## args
* `qs`: a QSystem.
* `drive`: a function of time returning a real or complex value. The real
    part couples to X and the imaginary part couples to Y.

Returns a univariate function of time.
"""
function rwa_dipole(qs::QSystem, drive::Function)
    x_ham = X(qs)
    y_ham = Y(qs)
    function ham(t)
        pulse = drive(t)
        return real(pulse) * x_ham + imag(pulse) * y_ham
    end
    return ham
end

"""
Construct a time dependent hamiltonian for a given QSystem.

    $(TYPEDSIGNATURES)

## args
* `qs`: a QSystem with a method of `hamiltonian` accepting a function of time.
* `drive`: a function of time returning a real value.

This function requires that the `QSystem` in question has an implementation of
    hamiltonian(qs::QSystem, drive::Function) 

Currently we can generate only a single parameter hamiltonian. This is a limitation we hope to lift in the future.
"""
function parametric_drive(qs::QSystem, drive::Function)
    ham(t) = hamiltonian(qs, drive(t))
    return ham
end

######################################################
### Fixed in Time Lindblad Operators
######################################################

"""
Contrust lindblad operator matrix representation for simulating T1 decay on a bosonic `Qsystem`.

    $(TYPEDSIGNATURES)

    `γ` : non negative decay rate

"""
function decay(qs::QSystem, γ::Real)
    @assert γ >= 0 "Decay rate γ must be non-negative."
    return lowering(qs, 0, sqrt(γ))
end

"""
Contrust lindblad operator matrix representation for simulating T2 decay on a bosonic `Qsystem`.

    $(TYPEDSIGNATURES)

    `γ` : non-negative decay rate

"""
function dephasing(qs::QSystem, γ::Real)
    @assert γ >= 0 "Decay rate γ must be non-negative."
    return number(qs, sqrt(2γ))
end