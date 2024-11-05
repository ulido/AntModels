@testset "Basic running - arena" begin
    parameters = AntModelParameters(
        width = 100.0mm,
        height = 100.0mm,
        domain_type = :arena,
        end_time = 10s,
    )
    run_ant_model(parameters, showprogress = false)
end

@testset "Force test" begin
    parameters = AntModelParameters(
        width = 100.0mm,
        height = 100.0mm,
        speed = 10.0mm/s,
        angle_diffusion_coefficient = 0.0rad^2/s,
        gamma = 0.0rad*mm/s,
        kappa = 0.0mm^2/s,
        pheromone_dx = 0.1mm,
        interaction_strength = 129mm^2/s,
        interaction_scale = 1.98mm,
        spawn_rate = 0.0/s,
        time_step = 0.04s,
        pheromone_model = :nopheromone,
        domain_type = :arena,
    )
    model = AntModels.ant_model(parameters)
    a1 = AntModels.add_agent!((40.0, 50.0), model, 0.0)
    a2 = AntModels.add_agent!((60.0, 50.0), model, π)
    r1 = 0.0
    r2 = a2.pos[1] - a1.pos[1]

    while abs(r1 - r2) > eps(r1)
        AntModels.run!(model, AntModels.ant_step!, AntModels.model_step!, 1)
        r1 = r2
        r2 = a2.pos[1] - a1.pos[1]
    end

    # The steady-state distance needs to be d(t→∞) = -L ln(v0 L / S)
    L = model.interaction_scale
    S = model.interaction_strength
    v0 = AntModels.speed(model.speed, model.time)
    d∞ = -L * log(v0 * L / S)
    @test (a2.pos[1] - a1.pos[1]) ≈ d∞
end

@testset "Pillars" begin
    parameters = AntModelParameters(
        domain=Domain(170.0mm, 10mm, :bridge),
        v0=AntModels.ConstantSpeedProcess(15.5),
        Dθ=0.38rad^2/s,
        L=2*1.98mm,
        β=0/s,
        T=100s,
        Δt=0.01s
    )

    model = AntModels.ant_model(parameters)
    AntModels.add_pillar!(m, (100.0, 5.0), 1.0)
    
    a = AntModels.add_agent!((100.0, 3.9), model, 0.0)
    println(AntModels.pillarcheck!(a, (100.0, 4.1), m))
    println(a)
end

# @testset "Pheromone" begin
#     parameters = AntModelParameters(
#         width = 100.0mm,
#         height = 10.0mm,
#         speed = 10.0mm/s,
#         angle_diffusion_coefficient = 0.0rad^2/s,
#         gamma = 0.1rad*mm/s,
#         kappa = 0.0mm^2/s,
#         pheromone_dx = 0.1mm,
#         interaction_strength = 129mm^2/s,
#         interaction_scale = 1.98mm,
#         spawn_rate = 0.0/s,
#         time_step = 0.04s,
#         pheromone_model = :gradient_nonlocal,
#         pheromone_λ = 1.0mm,
#         domain_type = :arena,
#     )

#     model = AntModels.ant_model(parameters)
#     AntModels.addpheromone!(model.pheromone, (0.0, 5.0), (100.0, 5.0))
#     a = AntModels.add_agent!((50.0, 1.0), model, π/4)

#     position = [a.pos]
#     while model.time < 2.0
#         AntModels.run!(model, AntModels.ant_step!, AntModels.model_step!, 1)
#         push!(position, a.pos)
#     end
#     println(position)
# end