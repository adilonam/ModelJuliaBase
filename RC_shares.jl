using LinearAlgebra
include("./lib.jl")
using Statistics
function RC_shares(DeltaM, SchoolsM, DistanceM, CweightsMAll, CweightsMTypes, Estimation, Set, Params, nargout)

    # Apply Normalization by Market x Year
    # if (DeltaM[SchoolsM.NormId .== 1] != 0)
    #     DeltaM = DeltaM - DeltaM[SchoolsM.NormId .== 1]
    # end

    S = zeros(size(SchoolsM, 1))
    Shares = fill(NaN, size(SchoolsM, 1), 4)
    share_ijv = Matrix{Array{Float64, 3}}(undef, 1, 4)

    dSdDelta = nothing
    dSdTheta = nothing
    moM = nothing
    dMo_dTheta = nothing
    dMo_dSdDdDdT = nothing
    MM = nothing
    dDeltadThetanorm = nothing

    if nargout > 3
        dSdDelta = zeros(size(SchoolsM, 1), size(SchoolsM, 1))
        if nargout > 4
            dSdTheta = zeros(size(SchoolsM, 1), size(Estimation.ThetaMask, 1))
            if nargout > 5
                moM = []
                if nargout > 6
                    dMo_dTheta = zeros(5 * 4, size(Estimation.ThetaMask, 1))
                    dMo_dSdDdDdT = zeros(5 * 4, size(Estimation.ThetaMask, 1))
                end
            end
        end
    end

    for type in 1:4
        if Set[:model] == "model 1"
            UoType = hcat([SchoolsM.Mu[i] * Params[:betaK][type, 1] .+ SchoolsM.Price[i] * Params[:betaK][type, 2] + DistanceM[i,:] * Params[:betaK][type, 3] for i in 1:size(SchoolsM,1)]...)'
            Uv = vcat([SchoolsM.Mu[i] * Params[:betai][:, 1]' for i in 1:size(SchoolsM,1)]...)
            Uv = reshape(Uv, size(Uv, 1), 1, size(Uv, 2))
        elseif Set[:model] == "model 2"
            UoType = hcat([SchoolsM.Mu[i] * Params[:betaK][type, 1] .+ SchoolsM.Price[i] * Params[:betaK][type, 2] .+ DistanceM[i,:] * Params[:betaK][type, 3] .+ SchoolsM.Ze[i][1, type] * Params[:betaK][type, 4] .+ SchoolsM.Zy[i][1, type] * Params[:betaK][type, 5] for i in 1:size(SchoolsM,1)]...)'
            Uv = vcat([SchoolsM.Mu[i] * Params[:betai][:, 1]' .+ SchoolsM.Ze[i][1, type] * Params[:betai][:, 2]' .+ SchoolsM.Zy[i][1, type] * Params[:betai][:, 3]' for i in 1:size(SchoolsM,1)]...)
            Uv = reshape(Uv, size(Uv, 1), 1, size(Uv, 2))
        end

        maxuij = maximum(DeltaM) + maximum(UoType) + maximum(Uv)
        num = exp.(DeltaM .+ UoType .+ Uv .- maxuij)
        share_ijv[1, type] = num ./ sum(num, dims=1)
        share_ij = dropdims(sum(share_ijv[1, type] .* Params[:w], dims=3), dims=3)
        Shares[:, type] = sum(share_ij .* CweightsMTypes[:, type]', dims=2)
        S += sum(share_ij .* CweightsMAll[:, type]', dims=2)

        if nargout > 5
            if Set[:model] == "model 1"
                moMy = (hcat(share_ij' * hcat(SchoolsM.Mu, SchoolsM.Price), sum(share_ij .* DistanceM, dims=1)')') * CweightsMTypes[:, type]
                moM = vcat(moM, moMy)
            elseif Set[:model] == "model 2"
                moMy = hcat(share_ij' * [SchoolsM.Mu SchoolsM.Price] , Matrix(sum(share_ij .* DistanceM, dims=1))',share_ij' * [SchoolsM.Ze[i][1, type] for i in 1:size(SchoolsM,1)],share_ij' *[SchoolsM.Zy[i][1, type] for i in 1:size(SchoolsM,1)])'* CweightsMTypes[:, type]
                moM = vcat(moM, moMy)
            end
        end

        if nargout > 3
            if size(SchoolsM, 1) < 1400
                dSdDelta += sum(mmx_mult(mmx_mult(permutedims(-share_ijv[1, type], (1, 3, 2)) ,diagm(vec(Params[:w][:, 1, :]))) , permutedims(share_ijv[1, type], (3, 1, 2))) .* reshape(CweightsMAll[:, type], 1, 1, :), dims=3)
            else
                for n in 1:length(CweightsMTypes[:, type])
                    dSdDelta += -squeeze(share_ijv[1, type][:, n, :], (1, 2)) * 
                                diagm(vec(Params[:w][1, 1, :])) * 
                                squeeze(share_ijv[1, type][:, n, :], (1, 2))' * 
                                CweightsMAll[n, type]
                end
            end
            

            if nargout > 4
                if size(SchoolsM, 1) < 1400
                    for i in Estimation.TypesTheta[type, 1]
                        if Estimation.ThetaMask[i][2] == 1
                            if Estimation.ThetaMask[i][1] == 0
                                dSdTheta[:, i] += squeeze(mmx_mult((share_ijv[1, type] .* (SchoolsM.Mu .- mmx_mult(SchoolsM.Mu, share_ijv[1, type], "tn"))), CweightsMAll[:, type])) * squeeze(Params[:w][:, 1, :])
                                if nargout > 6
                                    dMo_dTheta[1 + (type - 1) * 5, i] = mmx_mult(
                                        squeeze(mmx_mult(
                                            (share_ijv[1, type] .* bsxfun(-, SchoolsM.Mu, mmx_mult(SchoolsM.Mu, share_ijv[1, type], "tn"))),
                                            SchoolsM.Mu, "tn")
                                        ),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :])

                                    dMo_dTheta[2 + (type - 1) * 5, i] = mmx_mult(
                                        squeeze(mmx_mult(
                                            (share_ijv[1, type] .* bsxfun(-, SchoolsM.Mu, mmx_mult(SchoolsM.Mu, share_ijv[1, type], "tn"))),
                                            SchoolsM.Price, "tn")
                                        ),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :])

                                    dMo_dTheta[3 + (type - 1) * 5, i] = mmx_mult(
                                        squeeze(sum(
                                            (share_ijv[1, type] .* bsxfun(-, SchoolsM.Mu, mmx_mult(SchoolsM.Mu, share_ijv[1, type], "tn"))) .* DistanceM,
                                            dims=1
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :])
                                    
                                    println("type of squeeze is", typeof(squeeze(Params[:w][1, 1, :])))
                                    println("size of squeeze is", size(squeeze(Params[:w][1, 1, :])))
                                    dMo_dTheta[4 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            (share_ijv[1, type] .* bsxfun(-, SchoolsM.Mu, mmx_mult(SchoolsM.Mu, share_ijv[1, type], "tn"))),
                                            SchoolsM.Ze[:, type], "tn")
                                        ),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[5 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            (share_ijv[1, type] .* bsxfun(-, SchoolsM.Mu, mmx_mult(SchoolsM.Mu, share_ijv[1, type], "tn"))),
                                            SchoolsM.Zy[:, type], "tn")
                                        ),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                end
                            elseif Estimation.ThetaMask[i][1] == 1
                                dSdTheta[:, i] += squeeze(mmx_mult(
                                    bsxfun(*, share_ijv[1, type], reshape(Params[:dbetai][:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* 
                                    bsxfun(-, SchoolsM.Mu, mmx_mult(SchoolsM.Mu, share_ijv[1, type], "tn")),
                                    CweightsMAll[:, type]
                                )) * squeeze(Params[:w][1, 1, :])
                                if nargout > 6
                                    dMo_dTheta[1 + (type - 1) * 5, i] = squeeze(mmx_mult(
                                        squeeze(mmx_mult(
                                            bsxfun(*, share_ijv[1, type], reshape(Params[:dbetai][:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* 
                                            bsxfun(-, SchoolsM.Mu, mmx_mult(SchoolsM.Mu, share_ijv[1, type], "tn")),
                                            SchoolsM.Mu, "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    ))' * squeeze(Params[:w][1, 1, :])

                                    dMo_dTheta[2 + (type - 1) * 5, i] = squeeze(mmx_mult(
                                        squeeze(mmx_mult(
                                            bsxfun(*, share_ijv[1, type], reshape(Params[:dbetai][:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* 
                                            bsxfun(-, SchoolsM.Mu, mmx_mult(SchoolsM.Mu, share_ijv[1, type], "tn")),
                                            SchoolsM.Price, "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    ))' * squeeze(Params[:w][1, 1, :])

                                    dMo_dTheta[3 + (type - 1) * 5, i] = squeeze(mmx_mult(
                                        sum(
                                            bsxfun(*, share_ijv[1, type], reshape(Params[:dbetai][:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* 
                                            bsxfun(-, SchoolsM.Mu, mmx_mult(SchoolsM.Mu, share_ijv[1, type], "tn")) .* DistanceM,
                                            dims = 1
                                        ),
                                        CweightsMTypes[:, type]
                                    ))' * squeeze(Params[:w][1, 1, :])

                                    dMo_dTheta[4 + (type - 1) * 5, i] = squeeze(mmx_mult(
                                        squeeze(mmx_mult(
                                            bsxfun(*, share_ijv[1, type], reshape(Params[:dbetai][:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* 
                                            bsxfun(-, SchoolsM.Mu, mmx_mult(SchoolsM.Mu, share_ijv[1, type], "tn")),
                                            SchoolsM.Ze[:, type], "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    ))' * squeeze(Params[:w][1, 1, :])

                                    dMo_dTheta[5 + (type - 1) * 5, i] = squeeze(mmx_mult(
                                        squeeze(mmx_mult(
                                            bsxfun(*, share_ijv[1, type], reshape(Params[:dbetai][:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* 
                                            bsxfun(-, SchoolsM.Mu, mmx_mult(SchoolsM.Mu, share_ijv[1, type], "tn")),
                                            SchoolsM.Zy[:, type], "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    ))' * squeeze(Params[:w][1, 1, :])                  
                                end
                            end
                        elseif Estimation.ThetaMask[i][2] == 2
                            if Estimation.ThetaMask[i][1] == 0
                                dSdTheta[:, i] += squeeze(mmx_mult(
                                    share_ijv[1, type] .* bsxfun(-, SchoolsM.Price, mmx_mult(SchoolsM.Price, share_ijv[1, type], "tn")),
                                    CweightsMAll[:, type]
                                )) * squeeze(Params[:w][1, 1, :])
                                if nargout > 6
                                    dMo_dTheta[1 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            share_ijv[1, type] .* bsxfun(-, SchoolsM.Price, mmx_mult(SchoolsM.Price, share_ijv[1, type], "tn")),
                                            SchoolsM.Mu, "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[2 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            share_ijv[1, type] .* bsxfun(-, SchoolsM.Price, mmx_mult(SchoolsM.Price, share_ijv[1, type], "tn")),
                                            SchoolsM.Price, "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[3 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(sum(
                                            (share_ijv[1, type] .* bsxfun(-, SchoolsM.Price, mmx_mult(SchoolsM.Price, share_ijv[1, type], "tn"))) .* DistanceM,
                                            dims=1
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[4 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            share_ijv[1, type] .* bsxfun(-, SchoolsM.Price, mmx_mult(SchoolsM.Price, share_ijv[1, type], "tn")),
                                            SchoolsM.Ze[:, type], "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[5 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            share_ijv[1, type] .* bsxfun(-, SchoolsM.Price, mmx_mult(SchoolsM.Price, share_ijv[1, type], "tn")),
                                            SchoolsM.Zy[:, type], "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))
                                end
                            end
                        elseif Estimation.ThetaMask[i][2] == 3
                            if Estimation.ThetaMask[i][1] == 0
                                dSdTheta[:, i] += squeeze(mmx_mult(
                                    share_ijv[1, type] .* bsxfun(-, DistanceM, sum(bsxfun(*, DistanceM, share_ijv[1, type]), dims=1)),
                                    CweightsMAll[:, type]
                                )) * squeeze(Params[:w][1, 1, :])
                                if nargout > 6
                                    dMo_dTheta[1 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            share_ijv[1, type] .* bsxfun(-, DistanceM, sum(bsxfun(*, DistanceM, share_ijv[1, type]), dims=1)),
                                            SchoolsM.Mu, "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[2 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            share_ijv[1, type] .* bsxfun(-, DistanceM, sum(bsxfun(*, DistanceM, share_ijv[1, type]), dims=1)),
                                            SchoolsM.Price, "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[3 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(sum(
                                            (share_ijv[1, type] .* bsxfun(-, DistanceM, sum(bsxfun(*, DistanceM, share_ijv[1, type]), dims=1))) .* DistanceM,
                                            dims=1
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[4 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            share_ijv[1, type] .* bsxfun(-, DistanceM, sum(bsxfun(*, DistanceM, share_ijv[1, type]), dims=1)),
                                            SchoolsM.Ze[:, type], "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[5 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            share_ijv[1, type] .* bsxfun(-, DistanceM, sum(bsxfun(*, DistanceM, share_ijv[1, type]), dims=1)),
                                            SchoolsM.Zy[:, type], "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))
                                end
                            end
                        elseif Estimation.ThetaMask[i][2] == 4
                            if Estimation.ThetaMask[i][1] == 0
                                dSdTheta[:, i] += squeeze(mmx_mult(
                                    share_ijv[1, type] .*  bsxfun(.-, SchoolsM.Ze[:, type], mmx_mult(SchoolsM.Ze[:, type], share_ijv[1, type], "tn")),
                                    CweightsMAll[:, type]
                                )) * squeeze(Params[:w][1, 1, :])
                                if nargout > 6
                                    dMo_dTheta[1 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            share_ijv[1, type] .* bsxfun(-, SchoolsM.Ze[:, type], mmx_mult(SchoolsM.Ze[:, type], share_ijv[1, type], "tn")),
                                            SchoolsM.Mu, "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[2 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            share_ijv[1, type] .* bsxfun(-, SchoolsM.Ze[:, type], mmx_mult(SchoolsM.Ze[:, type], share_ijv[1, type], "tn")),
                                            SchoolsM.Price, "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[3 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(sum(
                                            (share_ijv[1, type] .* bsxfun(-, SchoolsM.Ze[:, type], mmx_mult(SchoolsM.Ze[:, type], share_ijv[1, type], "tn"))) .* DistanceM,
                                            dims=1
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[4 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            share_ijv[1, type] .* bsxfun(-, SchoolsM.Ze[:, type], mmx_mult(SchoolsM.Ze[:, type], share_ijv[1, type], "tn")),
                                            SchoolsM.Ze[:, type], "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[5 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            share_ijv[1, type] .* bsxfun(-, SchoolsM.Ze[:, type], mmx_mult(SchoolsM.Ze[:, type], share_ijv[1, type], "tn")),
                                            SchoolsM.Zy[:, type], "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))
                                end
                            elseif Estimation.ThetaMask[i][1] == 1
                                dSdTheta[:, i] += squeeze(mmx_mult(
                                    bsxfun(*, share_ijv[1, type], reshape(Params[:dbetai][:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* bsxfun(-, SchoolsM.Ze[:, type], mmx_mult(SchoolsM.Ze[:, type], share_ijv[1, type], "tn")),
                                    CweightsMAll[:, type]
                                )) * squeeze(Params[:w][1, 1, :])
                                if nargout > 6
                                    dMo_dTheta[1 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            bsxfun(*, share_ijv[1, type], reshape(Params[:dbetai][:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* bsxfun(-, SchoolsM.Ze[:, type], mmx_mult(SchoolsM.Ze[:, type], share_ijv[1, type], "tn")),
                                            SchoolsM.Mu, "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[2 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            bsxfun(*, share_ijv[1, type], reshape(Params[:dbetai][:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* bsxfun(-, SchoolsM.Ze[:, type], mmx_mult(SchoolsM.Ze[:, type], share_ijv[1, type], "tn")),
                                            SchoolsM.Price, "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[3 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(sum(
                                            (bsxfun(*, share_ijv[1, type], reshape(Params[:dbetai][:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* bsxfun(-, SchoolsM.Ze[:, type], mmx_mult(SchoolsM.Ze[:, type], share_ijv[1, type], "tn"))) .* DistanceM,
                                            dims=1
                                        )),
                                        CweightsMTypes[:, type]
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[4 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            bsxfun(*, share_ijv[1, type], reshape(Params[:dbetai][:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* bsxfun(-, SchoolsM.Ze[:, type], mmx_mult(SchoolsM.Ze[:, type], share_ijv[1, type], "tn")),
                                            SchoolsM.Ze[:, type], "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[5 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            bsxfun(*, share_ijv[1, type], reshape(Params[:dbetai][:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* bsxfun(-, SchoolsM.Ze[:, type], mmx_mult(SchoolsM.Ze[:, type], share_ijv[1, type], "tn")),
                                            SchoolsM.Zy[:, type], "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))
                                end
                            end
                        elseif Estimation.ThetaMask[i][2] == 5
                            if Estimation.ThetaMask[i][1] == 0
                                dSdTheta[:, i] += squeeze(mmx_mult(
                                    share_ijv[1, type] .* bsxfun(-, SchoolsM.Zy[:, type], mmx_mult(SchoolsM.Zy[:, type], share_ijv[1, type], "tn")),
                                    CweightsMAll[:, type]
                                )) * squeeze(Params[:w][1, 1, :])
                                if nargout > 6
                                    dMo_dTheta[1 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            share_ijv[1, type] .* bsxfun(-, SchoolsM.Zy[:, type], mmx_mult(SchoolsM.Zy[:, type], share_ijv[1, type], "tn")),
                                            SchoolsM.Mu, "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[2 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            share_ijv[1, type] .* bsxfun(-, SchoolsM.Zy[:, type], mmx_mult(SchoolsM.Zy[:, type], share_ijv[1, type], "tn")),
                                            SchoolsM.Price, "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[3 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(sum(
                                            (share_ijv[1, type] .* bsxfun(-, SchoolsM.Zy[:, type], mmx_mult(SchoolsM.Zy[:, type], share_ijv[1, type], "tn"))) .* DistanceM,
                                            dims=1
                                        )),
                                        CweightsMTypes[:, type]
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[4 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            share_ijv[1, type] .* bsxfun(-, SchoolsM.Zy[:, type], mmx_mult(SchoolsM.Zy[:, type], share_ijv[1, type], "tn")),
                                            SchoolsM.Ze[:, type], "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[5 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            share_ijv[1, type] .* bsxfun(-, SchoolsM.Zy[:, type], mmx_mult(SchoolsM.Zy[:, type], share_ijv[1, type], "tn")),
                                            SchoolsM.Zy[:, type], "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))
                                end
                            elseif Estimation.ThetaMask[i][1] == 1
                                dSdTheta[:, i] += squeeze(mmx_mult(
                                    bsxfun(*, share_ijv[1, type], reshape(Params[:dbetai][:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* bsxfun(-, SchoolsM.Zy[:, type], mmx_mult(SchoolsM.Zy[:, type], share_ijv[1, type], "tn")),
                                    CweightsMAll[:, type]
                                )) * squeeze(Params[:w][1, 1, :])
                                if nargout > 6
                                    dMo_dTheta[1 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            bsxfun(*, share_ijv[1, type], reshape(Params[:dbetai][:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* bsxfun(-, SchoolsM.Zy[:, type], mmx_mult(SchoolsM.Zy[:, type], share_ijv[1, type], "tn")),
                                            SchoolsM.Mu, "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[2 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            bsxfun(*, share_ijv[1, type], reshape(Params[:dbetai][:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* bsxfun(-, SchoolsM.Zy[:, type], mmx_mult(SchoolsM.Zy[:, type], share_ijv[1, type], "tn")),
                                            SchoolsM.Price, "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[3 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(sum(
                                            (bsxfun(*, share_ijv[1, type], reshape(Params[:dbetai][:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* bsxfun(-, SchoolsM.Zy[:, type], mmx_mult(SchoolsM.Zy[:, type], share_ijv[1, type], "tn"))) .* DistanceM,
                                            dims=1
                                        )),
                                        CweightsMTypes[:, type]
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[4 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            bsxfun(*, share_ijv[1, type], reshape(Params[:dbetai][:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* bsxfun(-, SchoolsM.Zy[:, type], mmx_mult(SchoolsM.Zy[:, type], share_ijv[1, type], "tn")),
                                            SchoolsM.Ze[:, type], "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))

                                    dMo_dTheta[5 + (type - 1) * 5, i] = mean(mmx_mult(
                                        squeeze(mmx_mult(
                                            bsxfun(*, share_ijv[1, type], reshape(Params[:dbetai][:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* bsxfun(-, SchoolsM.Zy[:, type], mmx_mult(SchoolsM.Zy[:, type], share_ijv[1, type], "tn")),
                                            SchoolsM.Zy[:, type], "tn"
                                        )),
                                        CweightsMTypes[:, type], "tn"
                                    )' * squeeze(Params[:w][1, 1, :]))
                                end
                            end
                        end
                    end
                end
            end
        end

    end

        
        if nargout > 5
            MM = moM
            # MM = [moM, moV]
        
            if nargout > 6
                dDeltadThetanorm = zeros(size(S, 1), size(Estimation.ThetaMask, 1))
        
                dSdDelta += diagm(S)
                TempDdDt = -inv(dSdDelta[SchoolsM.NormId .== 0, SchoolsM.NormId .== 0]) * dSdTheta[SchoolsM.NormId .== 0, :]
                dDeltadThetanorm[SchoolsM.NormId .== 0, :] = TempDdDt
        
                for type in 1:4
                    share_ij = sum(share_ijv[1, type] .* Params[:w], dims=3)
                    Diag = diagm(share_ij[:, 1])
                    
                    for n in 2:size(share_ij, 2)
                        Diag = cat(3, Diag, diagm(share_ij[:, n]))
                    end
        
                    dMo_dSdDdDdT[1 + (type - 1) * 5, :] = sum(mmult((mmult((mmult(mmult(permutedims(-share_ijv[1, type], (1, 3, 2)), diagm(Params[:w][:, 1, :])), permutedims(share_ijv[1, type], (3, 1, 2)))) .+ Diag, dDeltadThetanorm)), SchoolsM.Mu', "T") .* reshape(CweightsMTypes[:, type], 1, 1, :), dims=3)'
                    dMo_dSdDdDdT[2 + (type - 1) * 5, :] = sum(mmult((mmult((mmult(mmult(permutedims(-share_ijv[1, type], (1, 3, 2)), diagm(Params[:w][:, 1, :])), permutedims(share_ijv[1, type], (3, 1, 2)))) .+ Diag, dDeltadThetanorm)), SchoolsM.Price', "T") .* reshape(CweightsMTypes[:, type], 1, 1, :), dims=3)'
                    dMo_dSdDdDdT[3 + (type - 1) * 5, :] = sum(mmult(sum((mmult((mmult(mmult(permutedims(-share_ijv[1, type], (1, 3, 2)), diagm(Params[:w][:, 1, :])), permutedims(share_ijv[1, type], (3, 1, 2)))) .+ Diag, dDeltadThetanorm)) .* reshape(DistanceM, size(DistanceM, 1), 1, size(DistanceM, 2)), dims=1), SchoolsM.Ze[:, type]', "T") .* reshape(CweightsMTypes[:, type], 1, 1, :), dims=3)'
                    dMo_dSdDdDdT[4 + (type - 1) * 5, :] = sum(mmult((mmult((mmult(mmult(permutedims(-share_ijv[1, type], (1, 3, 2)), diagm(Params[:w][:, 1, :])), permutedims(share_ijv[1, type], (3, 1, 2)))) .+ Diag, dDeltadThetanorm)), SchoolsM.Ze[:, type]', "T") .* reshape(CweightsMTypes[:, type], 1, 1, :), dims=3)'
                    dMo_dSdDdDdT[5 + (type - 1) * 5, :] = sum(mmult((mmult((mmult(mmult(permutedims(-share_ijv[1, type], (1, 3, 2)), diagm(Params[:w][:, 1, :])), permutedims(share_ijv[1, type], (3, 1, 2)))) .+ Diag, dDeltadThetanorm)), SchoolsM.Zy[:, type]', "T") .* reshape(CweightsMTypes[:, type], 1, 1, :), dims=3)'
        
                end
        
                DMM_dTheta = dMo_dSdDdDdT + dMo_dTheta
        
            end
        end
        
        return S, Shares, share_ijv, dSdDelta, dSdTheta, MM, DMM_dTheta, dDeltadThetanorm
        
        end
        