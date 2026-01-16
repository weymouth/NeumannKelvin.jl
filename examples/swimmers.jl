using NeumannKelvin,GLMakie
function plane(L::T,ℓ;hᵤ=0.3ℓ,hᵥ=hᵤ, s=L/2+2π*ℓ, half=true) where T
    measure.((u,v)->SA[u,v,0],s:-hᵤ:-2s,((half ? hᵥ/2 : -s):hᵥ:s)',hᵤ,hᵥ,flip=true;T)
end
using GeometryBasics,FileIO
function affine(mesh, A, b)
    position = [Point3f(A * p + b) for p in mesh.position]
    GeometryBasics.Mesh(position, mesh.faces)
end

# Dolphin
mesh = affine(load("examples/LowPolyDolphin.stl"), SA[0 -1 0;1 0 0;0 0 1]/65,SA[0.043,0,-0.08])
dolphin = panelize(mesh); PanelTree(dolphin)
sys = gmressolve!(BodyPanelSystem(dolphin,wrap=PanelTree))
viz(sys)

ℓ = 0.2^2; freesurf = plane(1f0,ℓ,hᵤ=2π*ℓ/15,s=2/3,half=false); PanelTree(Table(freesurf))
sys = gmressolve!(FSPanelSystem(dolphin,freesurf;ℓ,θ²=16),itmax=150)
viz(sys)

# Shark
s,c = sincos(π/4); model = false ? "examples/Bruce.stl" : "examples/LowPolyBruce.stl"
mesh = load(model) |> m->affine(m, SA[c -s 0;s c 0;0 0 1]/275, SA[0,0,-0.28])
shark = panelize(mesh); shark = shark[shark.dA .> 0.1sum(shark.dA)/length(shark)]; viz(PanelTree(shark))
sys = gmressolve!(BodyPanelSystem(shark,wrap=PanelTree))
viz(sys)

ℓ = 0.3^2; freesurf = plane(1f0,ℓ,s=2/3,half=false); PanelTree(Table(freesurf))
sys = gmressolve!(FSPanelSystem(shark,freesurf;ℓ,θ²=16),itmax=150)
gmressolve!(sys,itmax=150)
viz(sys)