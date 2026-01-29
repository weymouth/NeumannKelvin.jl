using NeumannKelvin
h = 0.1 # spacing
freesurf = measure.((u,v)->SA[u,v,0],2:-h:-4,(2:-h:-2)',h,h)
S(θ₁,θ₂) = SA[0.5cos(θ₁),-0.1sin(θ₂)*sin(θ₁),0.1cos(θ₂)*sin(θ₁)-0.15]
body = panelize(S,0,π,0,2π,hᵤ=h)
sys = BodyPanelSystem(body) |> directsolve!

bigbody = panelize(S,0,π,0,2π,hᵤ=1/200,N_max=Inf); #20k panels
sys = BodyPanelSystem(bigbody,wrap=PanelTree) |> gmressolve!

halfbody = panelize(S,0,π,0,π,hᵤ=1/200,N_max=Inf); # half the θ₂ range
BodyPanelSystem(halfbody,wrap=PanelTree,sym_axes=2) |> gmressolve!

directsolve!(BodyPanelSystem(body,U=SA[0,-1,0]),verbose=false) |> addedmass

ℓ = 1/4; h = 0.3ℓ # Froude-length and spacing
# Make the free-surface grid using y-symmetry and resolving ℓ
freesurf = measure.((u,v)->SA[u,-v,0],2:-h:-4,(h/2:h:2)',h,h)
halfbody = panelize(S,0,π,0,π,hᵤ=h)

FSsys = FSPanelSystem(halfbody,freesurf;
                  ℓ,sym_axes=2,θ²=16) |> gmressolve!
steadyforce(FSsys)

# Read mesh, place it, and filter out panels intersecting z=0
using GeometryBasics,FileIO
function affine(mesh, A, b)  # rotate, scale, and shift the mesh
    position = [Point3f(A * p + b) for p in mesh.position]
    GeometryBasics.Mesh(position, mesh.faces)
end
dolphin = let 
    mesh = load("examples//LowPolyDolphin.stl")
    mesh = affine(mesh, SA[0 -1 0;1 0 0;0 0 1]/65,SA[0.043,0,-0.09])
    filter(p->abs(p.x[3])>0.01, panelize(mesh))
end

# The rest of the system matches the example above
h = 0.04; freesurf = measure.((u,v)->SA[u,-v,0],2/3:-h:-4/3,(-2/3:h:2/3)',h,h,T=Float32); # Float32 to match the Mesh
Meshsys = FSPanelSystem(dolphin,freesurf;ℓ=9f-2) |> gmressolve!

using GLMakie # can also use Plots, or WGLMakie (for browsers)
viz(Meshsys)

ℓ = 1/4; h = 0.3ℓ # Froude-length and spacing
halfbody = panelize(S,0,π,0,π,hᵤ=h)
NKsys = NKPanelSystem(halfbody;ℓ,sym_axes=2) |> directsolve!
steadyforce(NKsys)

DBsys = BodyPanelSystem(halfbody;sym_axes=(2,3)) |> directsolve! |> steadyforce

function wigley(hᵤ,hᵥ=hᵤ;B=1/8,D=1/6,kwargs...)
    S(u,v) = SA[u-0.5,-2B*u*(1-u)*(v)*(2-v),D*(v-1)]
    panelize(S;hᵤ,hᵥ,kwargs...)
end
NKsys = NKPanelSystem(wigley(0.025);
          ℓ=1/2π,sym_axes=2,contour=true) |> directsolve!
viz(NKsys)
