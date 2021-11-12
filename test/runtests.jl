# using Distributed
const num_wavelens = 28
# TODO issue of matrix inverse for n≥4, see emails
const n_max = 1
const lr = 1e-9 # 0.5 too small, <<1% change per iter (more like 0.1%). 2 blows up: dloss goes from 18 to 1961 all of a sudden and r goes negative
const num_ctrl_pts = 20

# if length(workers()) == 1
# addprocs(num_wavelens)
# println("Workers: $(length(workers()))")
# end

# @everywhere
using Tmatrix
using Zygote
using Plots
using Distributions

function objective(r, θ, wavelens, target_emiss)
    T = map(
        w -> Tmatrix.T_matrix_SeparateRealImag_arbitrary_mesh(
            n_max,
            w,
            input_unit,
            ε₁.re,
            ε₁.im,
            μ₁.re,
            μ₁.im,
            ε₂.re,
            ε₂.im,
            μ₂.re,
            μ₂.im,
            r,
            θ,
            zeros(size(θ)),
            rotationally_symmetric,
            symmetric_about_plane_perpendicular_z,
            BigFloat_precision,
        ),
        # CachingPool(workers()),
        wavelens,
    )

    k₁ = map(
        w -> Tmatrix.get_WaveVector(w; input_unit = input_unit, Eps_r = ε₁, Mu_r = μ₁),
        # CachingPool(workers()),
        wavelens,
    )

    surface_area = Tmatrix.calculate_surface_area_of_axisymmetric_particle(r, θ)

    emiss = map(
        (t, k₁) ->
            Tmatrix.get_OrentationAv_emissivity_from_Tmatrix(t, k₁, surface_area),
        # CachingPool(workers()),
        T,
        k₁,
    )
    println("pred emiss: $emiss")

    loss = sqrt(sum(abs.(emiss - target_emiss) .^ 2) / length(emiss))
    return loss
end

function dobjective(r, θ, target_wavelens, target_emiss)
    loss, back = Zygote.pullback(r -> objective(r, θ, target_wavelens, target_emiss), r)
    grad = back(one(loss))
    grad = grad[1] #index to unwrap 1-tuple
    return loss, grad
end

# TODO replace with perturbation from orig data set
function rand_target_emiss(n::Integer = num_wavelens)
    wavelens = collect(LinRange(1e-6, 20e-6, n))
    emiss = rand(Uniform(0.0, 0.2), n)

    # picked from actual data
    wavelens = [
        1.8406e-6,
        1.8981e-6,
        1.9592e-6,
        2.0245e-6,
        2.0942e-6,
        2.1689e-6,
        2.2491e-6,
        2.3354e-6,
        2.4287e-6,
        2.5297e-6,
        2.6395e-6,
        2.7593e-6,
        2.8904e-6,
        3.0346e-6,
        3.1940e-6,
        3.3710e-6,
        3.5689e-6,
        3.7913e-6,
        4.0434e-6,
        4.3314e-6,
        4.6635e-6,
        5.0508e-6,
        5.5083e-6,
        6.0569e-6,
        6.7269e-6,
        7.5635e-6,
        8.6377e-6,
        10.0676e-6,
    ]

    emiss = [
        0.0032,
        0.0067,
        0.0081,
        0.0088,
        0.0090,
        0.0091,
        0.0092,
        0.0092,
        0.0092,
        0.0092,
        0.0092,
        0.0092,
        0.0092,
        0.0093,
        0.0093,
        0.0093,
        0.0093,
        0.0094,
        0.0094,
        0.0095,
        0.0095,
        0.0096,
        0.0097,
        0.0099,
        0.0100,
        0.0102,
        0.0104,
        0.0108,
    ]

    return (wavelens[1:num_wavelens], emiss[1:num_wavelens])
end

# Particle size should be comparable to wavelen.
const input_unit = "m"
# electric permittivity
# vacuum
const ε₁ = 1.0 + 0.0im

# TODO value for gold?
# const ε₂ = 1.5 + 0.01im
const ε₂ = .467 + 2.415im
# TODO should depend on wavelen
# const ε₂ = -0.882 + 0.0076im

# magnetic permeability
const μ₁ = 1.0 + 0.0im
const μ₂ = 1.0 + 0.0im

const rotationally_symmetric = true
const symmetric_about_plane_perpendicular_z = false
const BigFloat_precision = nothing

# Set small to avoid radii growing too big, past wavelength, and T matrix
# blowing up in Bessel function calculation.
# TODO replace
const target_wavelens, target_emiss = rand_target_emiss(num_wavelens)

const rx, rz = (4.4e-6, 4.6e-6)

# 1e-5 is to pick a reasonable starting size
# TODO will be replaced by network
# r = rand(Uniform(1e-6, 100e-6), size(θ))

function main()
    θ = collect(LinRange(1e-6, pi - 1e-6, num_ctrl_pts))
    # jitter to break gradient symmetry
    θ += [
        0.0,
        rand(
            Uniform(-pi / (2 * num_ctrl_pts), pi / (2 * num_ctrl_pts)),
            num_ctrl_pts - 2,
        )...,
        0.0,
    ]
    r, _ = Tmatrix.ellipsoid(rx, rz, θ)
    losses = []
    for n_iteration in 1:5000
        # TODO use zygote.pullback
        loss, dloss_r = dobjective(r, θ, target_wavelens, target_emiss)

        # if any(abs(i) > 1 for i in dloss_r)
        # global lr /= 10
        # end

        # dloss_r = [clamp.(i, -1, 1) for i in dloss_r]

        r = r .- lr .* dloss_r
        append!(losses, loss)

        println("""
        iteration = $n_iteration
        loss = $loss
        dloss_r = $dloss_r
        r = $r

        """)

        # TODO restore
        #= mkpath("cache/plus_up")
        xyz = vcat(Tmatrix.convert_coordinates_Sph2Cart.(r, θ, zeros(size(r)))...)
        p1 = plot!(
            xyz[:, 1],
            xyz[:, 3],
            aspect_ratio = :equal,
            label = "particle after $n_iteration iterations",
            title = "scattering cross section = $loss m",
        )
        p2 = plot(
            1:length(losses),
            losses,
            xlabel = "iteration #",
            ylabel = "scattering cross section (m)",
        )
        fig = plot(p1, p2, layout = (1, 2), size = (1200, 800))
        # TODO fix ASAP
        savefig(fig, "cache/plus_up/$(n_iteration).png") =#
    end
end
main()
