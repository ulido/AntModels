

@testset "Basic running - arena" begin
    parameters = AntModelParameters(
        width = 100.0mm,
        height = 100.0mm,
        domain_type = :arena,
        end_time = 10s,
    )
    run_ant_model(parameters)
end