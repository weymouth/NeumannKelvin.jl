using NeumannKelvin,FileIO,GLMakie
svec = load("examples/optiwise_test_step.step") .|> s->NurbsSurface(s,1/300)
panels = mapreduce(s->panelize(s,hᵤ=0.02),vcat,svec[1:4])