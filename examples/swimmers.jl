using NeumannKelvin,GLMakie
using GeometryBasics,FileIO
function affine(mesh, A, b)
    position = [Point3f(A * p + b) for p in mesh.position]
    GeometryBasics.Mesh(position, mesh.faces)
end
plane(L,ℓ;hᵤ=ℓ/3,hᵥ=hᵤ, s=L/2f0+4ℓ) = measure.((x,y)->SA[x,y,0],s:-hᵤ:-2s,(-s:hᵥ:s)',hᵤ,hᵥ,flip=true)

# Dolphin
mesh = affine(load("examples/LowPolyDolphin.stl"), SA[0 -1 0;1 0 0;0 0 1]/65,SA[0.043,0,-0.08])
dolphin = panelize(mesh)
sys = gmressolve!(PanelSystem(PanelTree(dolphin)))
viz(sys)

ℓ = 4f-2; freesurf = plane(1,ℓ,hᵤ=ℓ/2)
sys = gmressolve!(PanelSystem((PanelTree(dolphin;θ²=16),FreeSurface(freesurf;ℓ,θ²=16))),itmax=150)
viz(sys)

# Shark
s,c = sincos(π/4f0)
mesh = affine(load("examples/LowPolyBruce.stl"), SA[c -s 0;s c 0;0 0 1]/275, SA[0,0,-0.28f0])
shark = panelize(mesh); shark = shark[shark.dA .> 0.1sum(shark.dA)/length(shark)] # drop sliver panels
sys = gmressolve!(PanelSystem(PanelTree(shark)))
viz(sys)

ℓ = 9f-2; freesurf = plane(1,ℓ)
sys = gmressolve!(PanelSystem((PanelTree(shark;θ²=16),FreeSurface(freesurf;ℓ,θ²=16))),itmax=150)
gmressolve!(sys,itmax=150) # restart from the current solution to finish converging
viz(sys)
