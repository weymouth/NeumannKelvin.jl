# Analytic prolate-spheroid axial-flow surface cp (Lamb/Munk potential theory).
# a = semi-major (flow-aligned) axis, b = semi-minor axis, η = x/a ∈ [-1,1] (η=1 is the nose).
function ellipsoid_cp_exact(a, b, η)
    e = sqrt(1-(b/a)^2)
    ξ0 = 1/e
    Q1(ξ) = (ξ/2)*log((ξ+1)/(ξ-1)) - 1
    Q1p(ξ) = (1/2)*log((ξ+1)/(ξ-1)) - ξ/(ξ^2-1)
    c = a*e
    B = -c/Q1p(ξ0)
    K = c*ξ0 + B*Q1(ξ0)
    uη = K*sqrt((1-η^2)/(ξ0^2-η^2))/c
    1 - uη^2   # U=1
end

# sanity check: as b→a (sphere limit) this must match cp = -1.25+2.25η²
# ellipsoid_cp_exact(1.0001,1.0,η) ≈ -1.25+2.25η²  (checked to ~1e-4)

# aspect ratio 5 prolate spheroid, sharp nose, flow aligned with the long (z) axis
a_minor, c_major = 0.4, 2.0
spheroid(θ₁,θ₂) = SA[a_minor*cos(θ₂)*sin(θ₁), a_minor*sin(θ₂)*sin(θ₁), c_major*cos(θ₁)]
Uax = SA[0.,0.,1.]
ell_cp_exact(x) = ellipsoid_cp_exact(c_major, a_minor, x[3]/c_major)

for hu in (0.4, 0.282, 0.2, 0.141, 0.1)
    pn = panelize(spheroid,0,π,0,2π,hᵤ=hu,N_max=3000)
    sys = BodyPanelSystem(pn,U=Uax)
    directsolve!(sys;verbose=false)
    err = cₚ(sys) .- ell_cp_exact.(pn.x)
    nosemask = abs.(getindex.(pn.x,3)) .> 0.9*c_major   # points near the sharp nose
    @show hu, length(pn), maximum(abs.(err)), maximum(abs.(err[nosemask])), sum(abs.(err))/length(err)


    pn = mapreduce(vcat,(-3,-2,-1,1,2,3)) do face
        panelize(-1.,1.,-1.,1.;hᵤ=hu,flip=face<0) do u,v
            SA[a_minor,a_minor,c_major] .* normalize(SVector{3}(circshift([sign(face),u,v],face)...))
        end
    end |> Table
    sys = BodyPanelSystem(pn,U=Uax)
    directsolve!(sys;verbose=false)
    err = cₚ(sys) .- ell_cp_exact.(pn.x)
    nosemask = abs.(getindex.(pn.x,3)) .> 0.9*c_major   # points near the sharp nose
    @show hu, length(pn), maximum(abs.(err)), maximum(abs.(err[nosemask])), sum(abs.(err))/length(err)
end