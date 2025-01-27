export perturbative_transmon_Ueigen

# The following matrices were taken from the source of arXiv:1706.06566.
# The regexes below were used to construct a matrix in Julia syntax
# from the LaTeX source syntax:

# \\ => newline
# \^\{([0-9]+)\}([^\}]) => ^$1*$2
# \\sqrt\{([0-9]+)\} => (√$1)
# \\frac\{([√0-9\^\*\{\}\(\)]+)\}\{([√0-9\^\*\{\}\(\)]+)\} => ($1)/($2)
# \^\{([0-9]+)\} => ^$1
# \)\( => )*(

# Additionally, Julia didn't recognize the minus signs (incorrect glyph?)
# I replaced those.

const U0 = collect(adjoint([
   1 0 0 0 0
   0 1 0 0 0
   0 0 1 0 0
   0 0 0 1 0
   0 0 0 0 1
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0]))

const U1 = collect(adjoint([
   0 0 -(1)/((√2)) 0 -(1)/((√2)^3*(√3))
   0 0 0 -(5)/((√2)*(√3)) 0
   (1)/((√2)) 0 0 0 -(7)/((√3))
   0 (5)/((√2)*(√3)) 0 0 0
   (1)/((√2)^3*(√3)) 0 (7)/((√3)) 0 0
   0 ((√5))/((√2)^3*(√3)) 0 (3(√5))/(1) 0
   0 0 ((√5))/((√2)^3) 0 (11(√5))/((√2)*(√3))
   0 0 0 ((√35))/((√2)^3*(√3)) 0
   0 0 0 0 ((√35))/(2(√3))
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0])) / 2^3

const U2 = collect(adjoint([
   -(13)/(2^8*3) 0 -(5)/((√2)^7*3) 0 ((√3))/((√2)^9)
   0 -(35)/(2^8) 0 -(13)/((√2)^7*(√3)) 0
   (13)/((√2)^11*3) 0 -(419)/(2^8*3) 0 -(67(√3))/(2^6)
   0 (37)/((√2)^11*(√3)) 0 -(405)/(2^8) 0
   (1)/((√2)^5*(√3)) 0 (145)/(2^6*(√3)) 0 -(961)/(2^8)
   0 (41)/((√2)^9*(√15)) 0 (79(√5))/(2^6) 0
   (23)/(2^6*3(√5)) 0 (103)/((√2)^7*3(√5)) 0 (29(√15))/((√2)^9)
   0 (11(√7))/(2^6*(√5)) 0 (103(√7))/((√2)^9*(√15)) 0
   ((√35))/((√2)^15*3) 0 (43(√7))/(2^5*3(√5)) 0 (3(√21))/((√5))
   0 ((√35))/((√2)^15) 0 (53(√7))/(2^5*(√15)) 0
   0 0 (5(√7))/((√2)^15) 0 (21(√21))/((√2)^11)
   0 0 0 (5(√77))/((√2)^15*(√3)) 0
   0 0 0 0 (5(√77))/((√2)^15)
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0])) / 2^2

const U3 = collect(adjoint([
   -(17)/(2^3*3) 0 -(725)/((√2)^11*3) 0 (1747)/((√2)^13*(√3))
   0 -(113)/(2^2*3) 0 -(869(√3))/((√2)^11) 0
   (347)/((√2)^11*3) 0 -(1241)/(2^3*3) 0 -(6217)/(2^5*(√3))
   0 (515(√3))/((√2)^11) 0 -(189)/(1) 0
   (605)/((√2)^13*(√3)) 0 (3793)/(2^5*(√3)) 0 -(2165)/(2^2)
   0 (2009(√5))/((√2)^13*(√3)) 0 (2225(√5))/(2^5) 0
   (409)/(2^4*3(√5)) 0 (68981)/((√2)^13*3(√5)) 0 (44689)/((√2)^11*(√15))
   0 (461(√5))/(2^4*(√7)) 0 (8663(√35))/((√2)^13*(√3)) 0
   (239)/((√2)^5*3(√35)) 0 (52681)/(2^5*3(√35)) 0 (36119(√7))/(2^5*(√15))
   0 (127(√5))/((√2)^3*3(√7)) 0 (52763(√5))/(2^5*3(√21)) 0
   (31(√7))/(2^5*3) 0 (1963)/((√2)^5*3(√7)) 0 (202283)/((√2)^9*3(√21))
   0 (41(√77))/(2^5*3) 0 (233(√11))/((√2)*(√21)) 0
   (5(√77))/(2^6*3(√3)) 0 (17(√231))/((√2)^9) 0 (3769(√11))/((√2)^5*3(√7))
   0 (5(√1001))/(2^6*3(√3)) 0 (61(√1001))/((√2)^9*3) 0
   0 0 (35(√143))/(2^6*3(√3)) 0 (497(√143))/(2^5*3)
   0 0 0 (35(√715))/(2^6*3(√3)) 0
   0 0 0 0 (35(√715))/(2^5*3(√3))
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0])) / 2^8

const U4 = collect(adjoint([
   -(62333)/((√2)^21*15) 0 -(9911)/(2^5*15) 0 (39007)/(2^6*3(√3))
   0 -(1313189)/((√2)^21*15) 0 -(137483)/(2^5*15(√3)) 0
   (16181)/(2^7*15) 0 -(651579)/((√2)^21) 0 -(3086339)/((√2)^15*15(√3))
   0 (110401)/(2^7*5(√3)) 0 -(44292461)/((√2)^21*15) 0
   (269(√3))/(2^5) 0 (227829(√3))/((√2)^15*5) 0 -(149454457)/((√2)^21*15)
   0 (8281(√5))/(2^6*(√3)) 0 (254291(√5))/((√2)^15*3) 0
   (7601)/((√2)^15*(√5)) 0 (5833)/(3(√5)) 0 (55231)/(2^2*3(√15))
   0 (85841(√7))/((√2)^15*3(√5)) 0 (5911741)/(2^6*3(√105)) 0
   (164443)/(2^9*3(√35)) 0 (1543667)/((√2)^13*3(√35)) 0 (11114269)/((√2)^11*3(√105))
   0 (4110871)/(2^9*9(√35)) 0 (1935007(√5))/((√2)^13*3(√21)) 0
   (135697)/((√2)^13*45(√7)) 0 (6278647(√7))/(2^9*45) 0 (1265399(√7))/(2^6*3(√3))
   0 (283201(√11))/((√2)^13*45(√7)) 0 (26982119(√11))/(2^9*15(√21)) 0
   (2759(√11))/((√2)^9*15(√21)) 0 (199855(√11))/(2^7*3(√21)) 0 (135190589(√11))/(2^9*45(√7))
   0 (8843(√143))/((√2)^11*15(√21)) 0 (1592063(√143))/(2^7*45(√7)) 0
   (91(√143))/(2^7*3(√3)) 0 (3217(√143))/((√2)^7*15(√3)) 0 (73963(√143))/((√2)^5*45)
   0 (343(√715))/(2^7*9) 0 (17593(√143))/((√2)^11*3(√15)) 0
   (35(√715))/(2^9*9) 0 (413(√715))/((√2)^11*9) 0 (11509(√143))/((√2)^7*3(√15))
   0 (35(√12155))/(2^9*9) 0 (161(√12155))/((√2)^11*3(√3)) 0
   0 0 (35(√12155))/(2^9*3) 0 (553(√12155))/(2^6*3(√3))
   0 0 0 (35(√230945))/(2^9*3(√3)) 0
   0 0 0 0 (175(√46189))/(2^9*3(√3))
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0
   0 0 0 0 0])) / (√2)^21

const U5 = collect(adjoint([
   -(153917)/((√2)^9*45) 0 -(10765201)/(2^8*45) 0 (202767029)/(2^9*15(√3))
   0 -(41527)/((√2)^3*3) 0 -(10916291)/(2^8*3(√3)) 0
   (4600889)/(2^8*45) 0 -(50029111)/((√2)^9*45) 0 -(51091643)/((√2)^15*5(√3))
   0 (9434411)/(2^8*3(√3)) 0 -(28185541)/((√2)^7*9) 0
   (9851179)/(2^9*15(√3)) 0 (188993981)/((√2)^15*15(√3)) 0 -(187170527)/((√2)^7*15)
   0 (25184897)/(2^9*(√15)) 0 (394451371)/((√2)^15*9(√5)) 0
   (994363)/((√2)^11*9(√5)) 0 (887375369)/(2^9*9(√5)) 0 (204498013)/(2^8*3(√15))
   0 (1873273(√7))/((√2)^11*3(√5)) 0 (5612971213)/(2^9*3(√105)) 0
   (558181)/(2^3*9(√35)) 0 (530587109)/((√2)^13*9(√35)) 0 (3942561341)/((√2)^17*(√105))
   0 (350365(√5))/(29(√7)) 0 (451113919)/((√2)^13*(√105)) 0
   (4552733)/((√2)^13*15(√7)) 0 (99431659)/(2^3*45(√7)) 0 (7205463341)/(2^6*15(√21))
   0 (94247731)/((√2)^13*9(√77)) 0 (7581187(√11))/(2^2*3(√21)) 0
   (117428699)/((√2)^15*45(√231)) 0 (3349510421)/(2^6*45(√231)) 0 (301458499(√11))/(2^4*15(√7))
   0 (6249499(√13))/((√2)^15*(√231)) 0 (449418449(√13))/(2^6*9(√77)) 0
   (108659(√143))/(2^5*45(√3)) 0 (555486781(√13))/((√2)^15*45(√33)) 0 (1352574373(√13))/((√2)^13*15(√11))
   0 (203167(√143))/(2^5*9(√5)) 0 (108310393(√13))/((√2)^15*(√165)) 0
   (2633(√143))/(2^3*9(√5)) 0 (59661(√715))/((√2)^13) 0 (1051619747(√13))/((√2)^15*3(√165))
   0 (491(√2431))/(9(√5)) 0 (454661(√2431))/((√2)^13*(√15)) 0
   (329(√12155))/((√2)^13*9) 0 (16369(√2431))/(2^3*9(√5)) 0 (2945699(√2431))/(2^6*3(√15))
   0 (133(√230945))/((√2)^13*3) 0 (401(√138567))/(2^2*(√5)) 0
   (35(√46189))/((√2)^15*9) 0 (2345(√46189))/(2^6*9) 0 (3071(√46189))/(2^3*(√3))
   0 (35(√323323))/((√2)^15*3(√3)) 0 (2695(√323323))/(2^6*9) 0
   0 0 (385(√29393))/((√2)^15*3(√3)) 0 (11165(√29393))/((√2)^13*3)
   0 0 0 (385(√676039))/((√2)^15*9) 0
   0 0 0 0 (385(√676039))/(2^7*3(√3))])) / (√2)^33

"""
   perturbative_transmon_Ueigen(ξ)
Returns a matrix representing a unitary transformation taking Fock states to
perturbative transmon eigenstates. The transformation is fifth-order in ξ and
uses the first 5 transmon eigenstates.
"""
perturbative_transmon_Ueigen(ξ) = @evalpoly(ξ, U0, U1, U2, U3, U4, U5)
