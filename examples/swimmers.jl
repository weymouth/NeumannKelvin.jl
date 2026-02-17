using NeumannKelvin#,GLMakie
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

plane(L,ℓ;hᵤ=ℓ/3,hᵥ=hᵤ, s=L/2f0+4ℓ) = measure.((x,y)->SA[x,y,0],s:-hᵤ:-2s,(-s:hᵥ:s)',hᵤ,hᵥ,flip=true)
ℓ = 4f-2; freesurf = plane(1,ℓ,hᵤ=ℓ/2); PanelTree(Table(freesurf))
sys = gmressolve!(FSPanelSystem(dolphin,freesurf;ℓ,θ²=16),itmax=150)
viz(sys)

# Shark
s,c = sincos(π/4f0); model = false ? "examples/Bruce.stl" : "examples/LowPolyBruce.stl"
mesh = load(model) |> m->affine(m, SA[c -s 0;s c 0;0 0 1]/275, SA[0,0,-0.28f0])
shark = panelize(mesh); shark = shark[shark.dA .> 0.1sum(shark.dA)/length(shark)]; viz(PanelTree(shark))
sys = gmressolve!(BodyPanelSystem(shark,wrap=PanelTree))
viz(sys)

ℓ = 9f-2; freesurf = plane(1,ℓ); PanelTree(Table(freesurf))
sys = gmressolve!(FSPanelSystem(shark,freesurf;ℓ,θ²=16),itmax=150)
gmressolve!(sys,itmax=150)
viz(sys)