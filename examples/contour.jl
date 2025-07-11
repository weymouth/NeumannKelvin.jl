using NeumannKelvin,Plots,ColorSchemes
using NeumannKelvin:quadgl,nearfield,wavelike
# Define cylinder surface and Green's function on the surface contour
R = 10; S(θ,z) = SA[R*cos(θ),R*sin(θ),z]
kelvin(x,y,z;ltol) = nearfield(x,y,z)+wavelike(x,y,z,ltol)
kelvin(v::SVector{3};ltol) = kelvin(v...;ltol)
kelvin_contour(θᵢ,θⱼ;z=-R/100,ltol=-8log(10)) = kelvin(S(θⱼ,z)-S(θᵢ,0);ltol)

# Find hardest segment
using QuadGK,FastGaussQuadrature
x,w = gausslegendre(10)
N = 64; dθ = 2π/N; θₛ = 0.5dθ:dθ:2π
err,θᵢ,θⱼ = maximum(θₛ) do θᵢ
    maximum(θₛ[1:N÷2+1] .-0.5dθ) do θⱼ # need this shift to break symmetry
        I,_ = quadgk(θ->kelvin_contour(θ,θⱼ),θᵢ-0.5dθ,θᵢ+0.5dθ)
        I₀ = quadgl(θ->kelvin_contour(θ,θⱼ),θᵢ-0.5dθ,θᵢ+0.5dθ;x,w)
        abs(I-I₀),θᵢ,θⱼ
    end
end
kelvin_contour(θ;kwargs...) = kelvin_contour(θ,θⱼ;kwargs...)
plot(range(θᵢ-0.5dθ,θᵢ+0.5dθ,1000),kelvin_contour)
quadgk_count(kelvin_contour,θᵢ-0.5dθ,θᵢ+0.5dθ)

# Plot G
N = 2^12; θ = range(0,2π,N+1)[1:end-1]
begin; plt=plot(xlabel="θ",ylabel="G")
for (n,c) in zip((100,50,25),reverse(colorschemes[:Reds_4]))
    plot!(θ,kelvin_contour.(θ;z=-R/n),label="z=-R/$n";c)
end;end;plt
savefig("examples/contour_G.png")

# Plot Ĝ
using FFTW
begin;plt=plot(ylims=(1e-8,1),yscale=:log10,ylabel="|FFT(G)|",
               xlims=(1e-1,1e2),xscale=:log10,xlabel="-kz");
for (n,c) in zip((100,50,25),reverse(colorschemes[:Reds_4]))
    Ĝ = fft(kelvin_contour.(θ;z=-R/n))/N
    dk = 1/n
    plot!(0:dk:(N÷2-1)*dk,abs.(Ĝ[1:N÷2]),label="z=-R/$n";c)
end;end;plt
savefig("examples/contour_spectral.png")

# Cylinder test with z-based contour sampling
function ∫kelvin_cylinder(ξ,p;q=0.1,ℓ=1,ltol=-3log(10))
    # reflected panel location & size
    z,θᵢ = -p.x[3],atan(p.x[2]/q,p.x[1])
    dz = extent(components(p.xᵤᵥ,3))
    dθ = extent(atan.(components(p.xᵤᵥ,2)/q,components(p.xᵤᵥ,1)))
    # evaluate kelvin along the contour
    x,w = gausslegendre(ceil(Int,p.dA/dz/abs(ξ[3]-z)))
    θ = @. θᵢ+0.5dθ*x
    ϕₖ = 0.5p.dA*w'*kelvin.((Ref(ξ) .- S.(θ,z))/ℓ;ltol)/ℓ
    # Add WL contribution if needed
    onwaterline(p) && (ϕₖ *= 1-ℓ/dz*p.n[1]^2)
    # Add source & sink contribution
    ϕₖ+∫G(ξ,p)-∫G(ξ,reflect(p,3))
end
∫kelvin_cylinder₂ = reflect(∫kelvin_cylinder,2)

function WL(panels;kwargs...)
    x,y,_ = filter(onwaterline,panels) |> p -> components(p.x)
    q = influence(panels;kwargs...)\components(panels.n,1)
    z = map(xy->ζ(xy[1],xy[2],q,panels;kwargs...),zip(x,y))
    x,z,q
end

begin
    dθ = π/64; θₛ = 0.5dθ:dθ:π
    dz = R/50 .* (1.31 .^ (0:15))
    z = 0.5dz .- cumsum(dz)
    ℓ = 0.16*2R
    cylinder = measure_panel.(S,θₛ,z',dθ,dz') |> Table
    x,z,q = WL(cylinder;ϕ=reflect(∫kelvin,2),ℓ,contour=true,filter=false)
    plot(x,z,label="2x2: no filter")
    x,z,q = WL(cylinder;ϕ=reflect(∫kelvin,2),ℓ,contour=true)
    plot!(x,z,label="2x2: z_max filter")
    x,z,q = WL(cylinder;ϕ=∫kelvin_cylinder₂,ℓ)
    plot!(x,z,label="N-point Gauss")
end