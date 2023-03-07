# Perturbative expansion of the Mathieu functions for the energy levels of a tunable transmon. The
# expansion is performed in the dimensionless paramter ξ = √(2EC/EJ). For details see Didier, N.,
# Sete, E. A., da Silva, M. P., & Rigetti, C. (2017). Analytical modeling of
# parametrically-modulated transmon qubits. http://arxiv.org/abs/1706.06566.

export perturbative_transmon_freq, perturbative_transmon_anharm
export perturbative_transmon_λ, perturbative_transmon_Λ
export transmon_N, transmon_φ

include("perturbative_transmon_ueigen.jl") # kind of messy!

"""
    transmon_φ(dim::Integer, ξ, ϕ::Real=0.0)
Transmon phase operator, φ̂. This function returns a matrix in the basis defined
by the Fock states of the linearized transmon Hamiltonian. See eq. (5) in
arXiv:1706.06566.

Tip: type φ with `\\varphi`.
"""
function transmon_φ(dim::Integer, ξ, ϕ::Real=0.0)
    return √(ξ) * (raising(dim, ϕ) + lowering(dim, ϕ))
end

"""
    transmon_N(dim::Integer, ξ, ϕ::Real=0.0)
Transmon charge number operator, N̂. This function returns a matrix in the basis
defined by the Fock states of the linearized transmon Hamiltonian. See eq. (5)
in arXiv:1706.06566.
"""
function transmon_N(dim::Integer, ξ, ϕ::Real=0.0)
    return (im / (2 * √ξ)) * (raising(dim, ϕ) - lowering(dim, ϕ))
end

# The expansions are given as a sum ∑ₚ aₚ/2^bₚ ξ^p and so each 2-tuple element below gives an (a,b)
# pair.

# 0 ↔ 1 transisition
const PT_FREQ = [
    (-1.0, 0),
    (-1.0, 2),
    (-21.0, 7),
    (-19.0, 7),
    (-5319.0, 15),
    (-6649.0, 15),
    (-1180581.0, 22),
    (-446287.0, 20),
    (-1489138635.0, 31),
    (-648381403.0, 29),
    (-614557854099.0, 38),
    (-75265839129.0, 34),
    (-637411859250147.0, 46),
    (-86690561488017.0, 42),
    (-405768570324517701.0, 53),
    (-15191635582891041.0, 47),
    (-2497063196283456607731.0, 63),
    (-102281923716042917215.0, 57),
    (-2292687293949773041433127.0, 70),
    (-25544408245062216574759.0, 62),
    (-4971071120163260007203175705.0, 78),
    (-59956026877695226936825271.0, 70),
    (-6299936888270974385982624367587.0, 85),
    (-20465345194746565030172477629.0, 75),
    (-36984324599399309412347250837528543.0, 94),
    (-128862667153189778842334459173303.0, 84),
    (-62313306363032484263243187605857135455.0, 101),
    (-57979140436623262897403437875845329.0, 89),
    (-239123145585215826671902664445701932931163.0, 109),
    (-236766982175345008541229542667501202161.0, 97),
    (-518667194120793070334115565427490753019904133.0, 116)
]

# anharmonicity
const PT_ANH = [
    (-1.0, 0),
    (-9.0, 4),
    (-81.0, 7),
    (-3645.0, 12),
    (-46899.0, 15),
    (-1329129.0, 19),
    (-20321361.0, 22),
    (-2648273373.0, 28),
    (-45579861135.0, 31),
    (-1647988255539.0, 35),
    (-31160327412879.0, 38),
    (-2457206583272505.0, 43),
    (-50387904068904927.0, 46),
    (-2145673984043982897.0, 50),
    (-47368663010124907041.0, 53),
    (-17329540083222030375645.0, 60),
    (-410048712835835979799431.0, 63),
    (-20066784213453521778111375.0, 67),
    (-507447585299180759749453827.0, 70),
    (-53019019946496461235728807475.0, 75),
    (-1429754157181172012054040903645.0, 78),
    (-79571741391885949104006842758911.0, 82),
    (-2283773190022904454409743892590327.0, 85),
    (-540565733415401595950277192471356985.0, 91),
    (-16479511149218202447739080120870460083.0, 94),
    (-1034743270413623494225962156243473940687.0, 98),
    (-33436402163767100825528499521512789823595.0, 101),
    (-4445866752212713247387096882061024305480817.0, 106),
    (-151943275517394187333713519962683890787229463.0, 109),
    (-10671890872277478133986830879693284605566580129.0, 113),
    (-384886904723357410697832985058012690322303378753.0, 116)
]

# weight of the number operator on the g-e σ_y operator in the three-level transmon eigenbasis
const PT_λ = [
    (1.0, 0),
    (-1.0, 3),
    (-11.0, 8),
    (-65.0, 11),
    (-4203.0, 17),
    (-40721.0, 20),
    (-1784885.0, 25),
    (-21465147.0, 28),
    (-4455462653.0, 35),
    (-61698199851.0, 38),
    (-3623317643901.0, 43),
    (-56143119646191.0, 46),
    (-7321743985484303.0, 52),
    (-125280019793719221.0, 55),
    (-8984438512815167237.0, 60),
    (-168544684286400995331.0, 63),
    (-105741913308715347076701.0, 71),
    (-2164311753394257835891059.0, 74),
    (-184798694135089048676718297.0, 79),
    (-4109869091672376619457585371.0, 82),
    (-761062061371895548979377743237.0, 88),
    (-18317012159331390907042783219855.0, 91),
    (-1831630981593132690479908285273395.0, 96),
    (-47512263370928552970648689915451821.0, 99),
    (-20440707519371829420653298425077482201.0, 106),
    (-569157711742925565406447462105395143103.0, 109)
]

# weight of the number operator on the e-f s_y operator in the three-level transmon eigenbasis
const PT_Λ = [
    (1.0, 0),
    (-1.0, 2),
    (-73.0, 9),
    (-79.0, 9),
    (-113685.0, 19),
    (-747533.0, 21),
    (-175422349.0, 28),
    (-698471247.0, 29),
    (-1520876829389.0, 39),
    (-13668058962903.0, 41),
    (-4122722770459287.0, 48),
    (-2534488707574995.0, 46),
    (-26543348405245135937.0, 58),
    (-281548290669062665101.0, 60),
    (-98933257452818263360213.0, 67),
    (-561603848629069641896937.0, 68),
    (-3372037991404912212166296765.0, 79),
    (-40819563311626093062783992331.0, 81),
    (-16314102788878455728540034311379.0, 88),
    (-52535388424912627194648863334467.0, 88),
    (-178610931461508948221684711385383067.0, 98),
    (-2444937960639526361173164055382471707.0, 100),
    (-1103567409503040799217165335410059740779.0, 107),
    (-8017554417550804194373089101907638666069.0, 108),
    (-30711842188423912661533983529887505235301321.0, 118),
    (-473069922042437374183190305740304564254754227.0, 120)
]

_prepare(x) = [par[1] / (2.0^par[2]) for par in x]
const PT_FREQ_PRE = _prepare(PT_FREQ)
const PT_ANH_PRE = _prepare(PT_ANH)
const PT_λ_PRE = _prepare(PT_λ)
const PT_Λ_PRE = _prepare(PT_Λ)

const PERTURBATIVE_NUM_TERMS = 10 # default number of terms to use

"""
    xi_effective(EC::Real, EJ1::Real, EJ2::Real, ϕ::Real)

Calculate the dimensionless transmon parameter ξ = √(2EC/EJ) where EJ is
the effective Josephson energy at some flux.

## args
* `EC`: the charging energy.
* `EJ1`: the Josephson energy of one junction.
* `EJ2`: the Josephson energy of the other junction.
* `ϕ`: the external flux through the SQUID in units of the flux quantum.
"""
function xi_effective(EC::Real, EJ1::Real, EJ2::Real, ϕ::Real)
    return sqrt((2 * EC) / sqrt((EJ1^2 + EJ2^2) + (2 * EJ1 * EJ2) * cos(2π * ϕ)))
end


"""
    perturbative_transmon_freq(EC::Real, EJ1::Real, EJ2::Real, ϕ::Real;
        num_terms::Int=$(PERTURBATIVE_NUM_TERMS))

Calculate the 01 transition frequency of a tunable or fixed transmon.

## args
* `EC`: the charging energy.
* `EJ1`: the Josephson energy of one junction.
* `EJ2`: the Josephson energy of the other junction.
* `ϕ`: the external flux through the SQUID in units of the flux quantum.
* `num_terms`: the number of terms in the perturbative expansion to use.
  Must be at least 1 and at most $(length(PT_FREQ_PRE)). To obtain the frequency
  at order `n` in `ξ`, use `n+2` terms. For `num_terms == 1`, the plasma
  frequency `√(8*EJ*EC)` is returned.

## returns
The frequency of the transmon.
"""
function perturbative_transmon_freq(EC::Real, EJ1::Real, EJ2::Real, ϕ::Real;
    num_terms::Int=PERTURBATIVE_NUM_TERMS)
    max_num_terms = length(PT_FREQ_PRE) + 1
    num_terms < 1 && throw(ArgumentError("num_terms < 1"))
    (num_terms > max_num_terms) && @warn "only using $max_num_terms terms."
    t = min(num_terms, max_num_terms)
    ξ = xi_effective(EC, EJ1, EJ2, ϕ)
    plasma_freq_over_EC = 4 / ξ
    return EC * (plasma_freq_over_EC + evalpoly(ξ, PT_FREQ_PRE[1:(t-1)]))
end

"""
    perturbative_transmon_anharm(EC::Real, EJ1::Real, EJ2::Real, ϕ::Real;
        num_terms::Int=$(PERTURBATIVE_NUM_TERMS))

Calculate the anharmonicity a tunable or fixed transmon. The sign convention
is that the anharmonicity of a transmon is negative.

## args
* `EC`: the charging energy.
* `EJ1`: the Josephson energy of one junction.
* `EJ2`: the Josephson energy of the other junction.
* `ϕ`: the external flux through the SQUID in units of the flux quantum.
* `num_terms`: the number of terms in the perturbative expansion to use.
  Must be at least 1 and at most $(length(PT_ANH_PRE)). To obtain the
  anharmonicity at order `n` in `ξ`, use `n+2` terms. For `num_terms == 1`,
  the function returns zero.

## returns
The anharmonicity of the transmon.
"""
function perturbative_transmon_anharm(EC::Real, EJ1::Real, EJ2::Real, ϕ::Real;
    num_terms::Int=PERTURBATIVE_NUM_TERMS)
    max_num_terms = length(PT_ANH_PRE) + 1
    num_terms < 1 && throw(ArgumentError("num_terms < 1"))
    (num_terms > max_num_terms) && @warn "only using $max_num_terms terms."
    t = min(num_terms, max_num_terms)
    ξ = xi_effective(EC, EJ1, EJ2, ϕ)
    return EC * evalpoly(ξ, PT_ANH_PRE[1:(t-1)])
end

"""
    perturbative_transmon_λ(EC::Real, EJ1::Real, EJ2::Real, ϕ::Real;
        num_terms::Int=$(PERTURBATIVE_NUM_TERMS))

Calculate the weight of the number operator on the g-e σ_y operator in the
three-level transmon eigenbasis, i.e. `N̂ = λ σ_y / (2*√ξ) + ⋯`

## args
* `EC`: the charging energy.
* `EJ1`: the Josephson energy of one junction.
* `EJ2`: the Josephson energy of the other junction.
* `ϕ`: the external flux through the SQUID in units of the flux quantum.
* `num_terms`: the number of terms in the perturbative expansion to use.
    Must be at least 1 and at most $(length(PT_λ_PRE)). To obtain λ at order `n`
    in `ξ`, use `n+1` terms.

## returns
The value of λ for the transmon.
"""
function perturbative_transmon_λ end

"""
    perturbative_transmon_Λ(EC::Real, EJ1::Real, EJ2::Real, ϕ::Real;
        num_terms::Int=$(PERTURBATIVE_NUM_TERMS))

Calculate the weight of the number operator on the e-f s_y operator in the
three-level transmon eigenbasis, i.e. `N̂ = ⋯ + Λ s_y / √(2ξ) + ⋯`

## args
* `EC`: the charging energy.
* `EJ1`: the Josephson energy of one junction.
* `EJ2`: the Josephson energy of the other junction.
* `ϕ`: the external flux through the SQUID in units of the flux quantum.
* `num_terms`: the number of terms in the perturbative expansion to use.
    Must be at least 1 and at most $(length(PT_Λ_PRE)). To obtain Λ at order `n`
    in `ξ`, use `n+1` terms.

## returns
The value of Λ for the transmon.
"""
function perturbative_transmon_Λ end

for (name, array) in [(:perturbative_transmon_λ, PT_λ_PRE),
    (:perturbative_transmon_Λ, PT_Λ_PRE)]
    @eval function ($name)(EC::Real, EJ1::Real, EJ2::Real, ϕ::Real;
        num_terms::Int=PERTURBATIVE_NUM_TERMS)
        max_num_terms = length($array)
        num_terms < 1 && throw(ArgumentError("num_terms < 1"))
        (num_terms > max_num_terms) && @warn "only using $max_num_terms terms."
        t = min(num_terms, max_num_terms)
        ξ = xi_effective(EC, EJ1, EJ2, ϕ)
        return evalpoly(ξ, $array[1:t])
    end
end

######################################################
# Backwards compatibility
######################################################
export mathieu_f01, mathieu_η

function mathieu_f01(t_params::Tuple{Float64,Float64,Float64}, ϕ::Real)
    @warn "Deprecation warning: mathieu_f01."
    EC, EJ1, EJ2 = t_params
    return perturbative_transmon_freq(EC, EJ1, EJ2, ϕ, num_terms=PERTURBATIVE_NUM_TERMS)
end

function mathieu_η(t_params::Tuple{Float64,Float64,Float64}, ϕ)
    @warn "Deprecation warning: mathieu_η."
    EC, EJ1, EJ2 = t_params
    return perturbative_transmon_anharm(EC, EJ1, EJ2, ϕ, num_terms=PERTURBATIVE_NUM_TERMS)
end
