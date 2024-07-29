function RC_shares(DeltaM::AbstractVector{T}, SchoolsM, DistanceM, CweightsMAll, CweightsMTypes, Estimation, Set, Theta2::AbstractVector{T}) where T

    # Apply Normalization by Market x Year
    # if (DeltaM[SchoolsM.NormId .== 1] != 0)
    #     DeltaM = DeltaM - DeltaM[SchoolsM.NormId .== 1]
    # end
    # Debug prints

    println("Types: DeltaM = $(typeof(DeltaM)), theta2 = $(typeof(theta2))")
     Params = GetParams(Estimation, Set, Theta2)


    nargout =7

    S = zeros(eltype(DeltaM), size(SchoolsM, 1))
    Shares = fill(T(NaN), size(SchoolsM, 1), 4)
    share_ijv = Array{Array{T, 3}}(undef, 4)
  
    moM = []

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
        println("num type: $(typeof(num))")

        denom = reshape(sum(num, dims=1), (1, size(num, 2), size(num, 3)))
        share_ijv[type] = num ./ denom
        println("share_ijv[$type] type: $(typeof(share_ijv[type]))")
        
        share_ij = dropdims(sum(share_ijv[type] .* Params[:w], dims=3), dims=3)
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

    end

    return S, Shares, share_ijv, moM

end

 
nargout = 7

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

dSdDelta = nothing
dSdTheta = nothing
moM = nothing
dMo_dTheta = nothing
dMo_dSdDdDdT = nothing
MM = nothing
dDeltadThetanorm = nothing

# RC_shares(DeltaM, SchoolsM, DistanceM, CweightsMAll, CweightsMTypes, Estimation, Set, Theta2)



#### dSdDelta

f = x -> RC_shares(x, SchoolsM, DistanceM, CweightsMAll, CweightsMTypes, Estimation, Set, Theta2)[1][1]
ff = x -> RC_shares(x, SchoolsM, DistanceM, CweightsMAll, CweightsMTypes, Estimation, Set, Theta2)[1]

# Calculating gradient using ForwardDiff
gradient = ForwardDiff.gradient(f, DeltaM)
jacobian = ForwardDiff.jacobian(ff, DeltaM)
### dSdTheta



# Define the function to extract the entire vector S
function f2(theta2)
    S, _, _, _ = RC_shares(DeltaM, SchoolsM, DistanceM, CweightsMAll, CweightsMTypes, Estimation, Set, theta2)

    return S
end


# Calculating Jacobian using ForwardDiff
jacobian2 = ForwardDiff.gradient(f2, Theta2)

println(jacobian2)




# Define the function to extract the entire vector S
f = x -> RC_shares(x, SchoolsM, DistanceM, CweightsMAll, CweightsMTypes, Estimation, Set, Params)[1]

# Calculating Jacobian using ForwardDiff
jacobian = ForwardDiff.jacobian(f, DeltaM)

println(jacobian)

dSdDelta = ForwardDiff.gradient(RC_shares(DeltaM, SchoolsM, DistanceM, CweightsMAll, CweightsMTypes, Estimation, Set, Params), DeltaM)

   # Commenting out the rest of the code
#=


        if nargout > 3
            if size(SchoolsM, 1) < 1400
                dSdDelta += sum(((permutedims(-share_ijv[1, type], (1, 3, 2)) .* diagm(Params.w[:, 1, :])) .* permutedims(share_ijv[1, type], (3, 1, 2))) .* reshape(CweightsMAll[:, type], 1, 1, :), dims=3)
            else
                for n in 1:length(CweightsMTypes[:, type])
                    dSdDelta += (-share_ijv[1, type][:, n, :] * diagm(Params.w[:, 1, :]) * share_ijv[1, type][:, n, :]' * CweightsMAll[n, type])
                end
            end

            if nargout > 4
                if size(SchoolsM, 1) < 1400
                    for i in Estimation.TypesTheta[type, 1]
                        if Estimation.ThetaMask[i, 2] == 1
                            if Estimation.ThetaMask[i, 1] == 0
                                dSdTheta[:, i] += squeeze(mmult((share_ijv[1, type] .* (SchoolsM.Mu .- mmult(SchoolsM.Mu, share_ijv[1, type], "T"))), CweightsMAll[:, type])) * Params.w[:, 1, :]
                                if nargout > 6
                                    dMo_dTheta[1 + (type - 1) * 5, i] = sum(mmult((share_ijv[1, type] .* (SchoolsM.Mu .- mmult(SchoolsM.Mu, share_ijv[1, type], "T"))), SchoolsM.Mu', "T") .* CweightsMTypes[:, type])' * Params.w[:, 1, :]
                                    dMo_dTheta[2 + (type - 1) * 5, i] = sum(mmult((share_ijv[1, type] .* (SchoolsM.Mu .- mmult(SchoolsM.Mu, share_ijv[1, type], "T"))), SchoolsM.Price', "T") .* CweightsMTypes[:, type])' * Params.w[:, 1, :]
                                    dMo_dTheta[3 + (type - 1) * 5, i] = sum(mmult(sum((share_ijv[1, type] .* (SchoolsM.Mu .- mmult(SchoolsM.Mu, share_ijv[1, type], "T"))) .* DistanceM, dims=1), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[4 + (type - 1) * 5, i] = sum(mmult((share_ijv[1, type] .* (SchoolsM.Mu .- mmult(SchoolsM.Mu, share_ijv[1, type], "T"))), SchoolsM.Ze[:, type]', "T") .* CweightsMTypes[:, type])' * Params.w[:, 1, :]
                                    dMo_dTheta[5 + (type - 1) * 5, i] = sum(mmult((share_ijv[1, type] .* (SchoolsM.Mu .- mmult(SchoolsM.Mu, share_ijv[1, type], "T"))), SchoolsM.Zy[:, type]', "T") .* CweightsMTypes[:, type])' * Params.w[:, 1, :]
                                end
                            elseif Estimation.ThetaMask[i, 1] == 1
                                dSdTheta[:, i] += squeeze(mmult(bsxfun(*, share_ijv[1, type], reshape(Params.dbetai[:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* (SchoolsM.Mu .- mmult(SchoolsM.Mu, share_ijv[1, type], "T")), CweightsMAll[:, type])) * Params.w[:, 1, :]
                                if nargout > 6
                                    dMo_dTheta[1 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult(bsxfun(*, share_ijv[1, type], reshape(Params.dbetai[:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* (SchoolsM.Mu .- mmult(SchoolsM.Mu, share_ijv[1, type], "T")), SchoolsM.Mu', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[2 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult(bsxfun(*, share_ijv[1, type], reshape(Params.dbetai[:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* (SchoolsM.Mu .- mmult(SchoolsM.Mu, share_ijv[1, type], "T")), SchoolsM.Price', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[3 + (type - 1) * 5, i] = sum(mmult(sum((bsxfun(*, share_ijv[1, type], reshape(Params.dbetai[:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* (SchoolsM.Mu .- mmult(SchoolsM.Mu, share_ijv[1, type], "T"))) .* DistanceM, dims=1), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[4 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult(bsxfun(*, share_ijv[1, type], reshape(Params.dbetai[:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* (SchoolsM.Mu .- mmult(SchoolsM.Mu, share_ijv[1, type], "T")), SchoolsM.Ze[:, type]', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[5 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult(bsxfun(*, share_ijv[1, type], reshape(Params.dbetai[:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* (SchoolsM.Mu .- mmult(SchoolsM.Mu, share_ijv[1, type], "T")), SchoolsM.Zy[:, type]', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                end
                            end
                        elseif Estimation.ThetaMask[i, 2] == 2
                            if Estimation.ThetaMask[i, 1] == 0
                                dSdTheta[:, i] += squeeze(mmult((share_ijv[1, type] .* (SchoolsM.Price .- mmult(SchoolsM.Price, share_ijv[1, type], "T"))), CweightsMAll[:, type])) * Params.w[:, 1, :]
                                if nargout > 6
                                    dMo_dTheta[1 + (type - 1) * 5, i] = sum(mmult((share_ijv[1, type] .* (SchoolsM.Price .- mmult(SchoolsM.Price, share_ijv[1, type], "T"))), SchoolsM.Mu', "T") .* CweightsMTypes[:, type])' * Params.w[:, 1, :]
                                    dMo_dTheta[2 + (type - 1) * 5, i] = sum(mmult((share_ijv[1, type] .* (SchoolsM.Price .- mmult(SchoolsM.Price, share_ijv[1, type], "T"))), SchoolsM.Price', "T") .* CweightsMTypes[:, type])' * Params.w[:, 1, :]
                                    dMo_dTheta[3 + (type - 1) * 5, i] = sum(mmult(sum((share_ijv[1, type] .* (SchoolsM.Price .- mmult(SchoolsM.Price, share_ijv[1, type], "T"))) .* DistanceM, dims=1), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[4 + (type - 1) * 5, i] = sum(mmult((share_ijv[1, type] .* (SchoolsM.Price .- mmult(SchoolsM.Price, share_ijv[1, type], "T"))), SchoolsM.Ze[:, type]', "T") .* CweightsMTypes[:, type])' * Params.w[:, 1, :]
                                    dMo_dTheta[5 + (type - 1) * 5, i] = sum(mmult((share_ijv[1, type] .* (SchoolsM.Price .- mmult(SchoolsM.Price, share_ijv[1, type], "T"))), SchoolsM.Zy[:, type]', "T") .* CweightsMTypes[:, type])' * Params.w[:, 1, :]
                                end
                            end
                        elseif Estimation.ThetaMask[i, 2] == 3
                            if Estimation.ThetaMask[i, 1] == 0
                                dSdTheta[:, i] += squeeze(mmult((share_ijv[1, type] .* (DistanceM .- sum((share_ijv[1, type] .* DistanceM), dims=1))), CweightsMAll[:, type])) * Params.w[:, 1, :]
                                if nargout > 6
                                    dMo_dTheta[1 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult((share_ijv[1, type] .* (DistanceM .- sum((share_ijv[1, type] .* DistanceM), dims=1))), SchoolsM.Mu', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[2 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult((share_ijv[1, type] .* (DistanceM .- sum((share_ijv[1, type] .* DistanceM), dims=1))), SchoolsM.Price', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[3 + (type - 1) * 5, i] = sum(mmult(squeeze(sum((share_ijv[1, type] .* (DistanceM .- sum((share_ijv[1, type] .* DistanceM), dims=1))) .* DistanceM, dims=1)), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[4 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult((share_ijv[1, type] .* (DistanceM .- sum((share_ijv[1, type] .* DistanceM), dims=1))), SchoolsM.Ze[:, type]', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[5 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult((share_ijv[1, type] .* (DistanceM .- sum((share_ijv[1, type] .* DistanceM), dims=1))), SchoolsM.Zy[:, type]', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                end
                            end
                        elseif Estimation.ThetaMask[i, 2] == 4
                            if Estimation.ThetaMask[i, 1] == 0
                                dSdTheta[:, i] += squeeze(mmult((share_ijv[1, type] .* (SchoolsM.Ze[:, type] .- mmult(SchoolsM.Ze[:, type], share_ijv[1, type], "T"))), CweightsMAll[:, type])) * Params.w[:, 1, :]
                                if nargout > 6
                                    dMo_dTheta[1 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult((share_ijv[1, type] .* (SchoolsM.Ze[:, type] .- mmult(SchoolsM.Ze[:, type], share_ijv[1, type], "T"))), SchoolsM.Mu', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[2 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult((share_ijv[1, type] .* (SchoolsM.Ze[:, type] .- mmult(SchoolsM.Ze[:, type], share_ijv[1, type], "T"))), SchoolsM.Price', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[3 + (type - 1) * 5, i] = sum(mmult(squeeze(sum((share_ijv[1, type] .* (SchoolsM.Ze[:, type] .- mmult(SchoolsM.Ze[:, type], share_ijv[1, type], "T"))) .* DistanceM, dims=1)), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[4 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult((share_ijv[1, type] .* (SchoolsM.Ze[:, type] .- mmult(SchoolsM.Ze[:, type], share_ijv[1, type], "T"))), SchoolsM.Ze[:, type]', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[5 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult((share_ijv[1, type] .* (SchoolsM.Ze[:, type] .- mmult(SchoolsM.Ze[:, type], share_ijv[1, type], "T"))), SchoolsM.Zy[:, type]', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                end
                            elseif Estimation.ThetaMask[i, 1] == 1
                                dSdTheta[:, i] += squeeze(mmult(bsxfun(*, share_ijv[1, type], reshape(Params.dbetai[:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* (SchoolsM.Ze[:, type] .- mmult(SchoolsM.Ze[:, type], share_ijv[1, type], "T")), CweightsMAll[:, type])) * Params.w[:, 1, :]
                                if nargout > 6
                                    dMo_dTheta[1 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult(bsxfun(*, share_ijv[1, type], reshape(Params.dbetai[:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* (SchoolsM.Ze[:, type] .- mmult(SchoolsM.Ze[:, type], share_ijv[1, type], "T")), SchoolsM.Mu', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[2 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult(bsxfun(*, share_ijv[1, type], reshape(Params.dbetai[:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* (SchoolsM.Ze[:, type] .- mmult(SchoolsM.Ze[:, type], share_ijv[1, type], "T")), SchoolsM.Price', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[3 + (type - 1) * 5, i] = sum(mmult(squeeze(sum((bsxfun(*, share_ijv[1, type], reshape(Params.dbetai[:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* (SchoolsM.Ze[:, type] .- mmult(SchoolsM.Ze[:, type], share_ijv[1, type], "T"))) .* DistanceM, dims=1)), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[4 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult(bsxfun(*, share_ijv[1, type], reshape(Params.dbetai[:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* (SchoolsM.Ze[:, type] .- mmult(SchoolsM.Ze[:, type], share_ijv[1, type], "T")), SchoolsM.Ze[:, type]', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[5 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult(bsxfun(*, share_ijv[1, type], reshape(Params.dbetai[:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* (SchoolsM.Ze[:, type] .- mmult(SchoolsM.Ze[:, type], share_ijv[1, type], "T")), SchoolsM.Zy[:, type]', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                end
                            end
                        elseif Estimation.ThetaMask[i, 2] == 5
                            if Estimation.ThetaMask[i, 1] == 0
                                dSdTheta[:, i] += squeeze(mmult((share_ijv[1, type] .* (SchoolsM.Zy[:, type] .- mmult(SchoolsM.Zy[:, type], share_ijv[1, type], "T"))), CweightsMAll[:, type])) * Params.w[:, 1, :]
                                if nargout > 6
                                    dMo_dTheta[1 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult((share_ijv[1, type] .* (SchoolsM.Zy[:, type] .- mmult(SchoolsM.Zy[:, type], share_ijv[1, type], "T"))), SchoolsM.Mu', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[2 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult((share_ijv[1, type] .* (SchoolsM.Zy[:, type] .- mmult(SchoolsM.Zy[:, type], share_ijv[1, type], "T"))), SchoolsM.Price', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[3 + (type - 1) * 5, i] = sum(mmult(squeeze(sum((share_ijv[1, type] .* (SchoolsM.Zy[:, type] .- mmult(SchoolsM.Zy[:, type], share_ijv[1, type], "T"))) .* DistanceM, dims=1)), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[4 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult((share_ijv[1, type] .* (SchoolsM.Zy[:, type] .- mmult(SchoolsM.Zy[:, type], share_ijv[1, type], "T"))), SchoolsM.Ze[:, type]', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[5 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult((share_ijv[1, type] .* (SchoolsM.Zy[:, type] .- mmult(SchoolsM.Zy[:, type], share_ijv[1, type], "T"))), SchoolsM.Zy[:, type]', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                end
                            elseif Estimation.ThetaMask[i, 1] == 1
                                dSdTheta[:, i] += squeeze(mmult(bsxfun(*, share_ijv[1, type], reshape(Params.dbetai[:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* (SchoolsM.Zy[:, type] .- mmult(SchoolsM.Zy[:, type], share_ijv[1, type], "T")), CweightsMAll[:, type])) * Params.w[:, 1, :]
                                if nargout > 6
                                    dMo_dTheta[1 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult(bsxfun(*, share_ijv[1, type], reshape(Params.dbetai[:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* (SchoolsM.Zy[:, type] .- mmult(SchoolsM.Zy[:, type], share_ijv[1, type], "T")), SchoolsM.Mu', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[2 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult(bsxfun(*, share_ijv[1, type], reshape(Params.dbetai[:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* (SchoolsM.Zy[:, type] .- mmult(SchoolsM.Zy[:, type], share_ijv[1, type], "T")), SchoolsM.Price', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[3 + (type - 1) * 5, i] = sum(mmult(squeeze(sum((bsxfun(*, share_ijv[1, type], reshape(Params.dbetai[:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* (SchoolsM.Zy[:, type] .- mmult(SchoolsM.Zy[:, type], share_ijv[1, type], "T"))) .* DistanceM, dims=1)), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[4 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult(bsxfun(*, share_ijv[1, type], reshape(Params.dbetai[:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* (SchoolsM.Zy[:, type] .- mmult(SchoolsM.Zy[:, type], share_ijv[1, type], "T")), SchoolsM.Ze[:, type]', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
                                    dMo_dTheta[5 + (type - 1) * 5, i] = sum(mmult(squeeze(mmult(bsxfun(*, share_ijv[1, type], reshape(Params.dbetai[:, Estimation.ThetaMask[i, 3]], 1, 1, :)) .* (SchoolsM.Zy[:, type] .- mmult(SchoolsM.Zy[:, type], share_ijv[1, type], "T")), SchoolsM.Zy[:, type]', "T")), CweightsMTypes[:, type]))' * Params.w[:, 1, :]
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
                    share_ij = sum(share_ijv[1, type] .* Params.w, dims=3)
                    Diag = diagm(share_ij[:, 1])
                    
                    for n in 2:size(share_ij, 2)
                        Diag = cat(3, Diag, diagm(share_ij[:, n]))
                    end
        
                    dMo_dSdDdDdT[1 + (type - 1) * 5, :] = sum(mmult((mmult((mmult(mmult(permutedims(-share_ijv[1, type], (1, 3, 2)), diagm(Params.w[:, 1, :])), permutedims(share_ijv[1, type], (3, 1, 2)))) .+ Diag, dDeltadThetanorm)), SchoolsM.Mu', "T") .* reshape(CweightsMTypes[:, type], 1, 1, :), dims=3)'
                    dMo_dSdDdDdT[2 + (type - 1) * 5, :] = sum(mmult((mmult((mmult(mmult(permutedims(-share_ijv[1, type], (1, 3, 2)), diagm(Params.w[:, 1, :])), permutedims(share_ijv[1, type], (3, 1, 2)))) .+ Diag, dDeltadThetanorm)), SchoolsM.Price', "T") .* reshape(CweightsMTypes[:, type], 1, 1, :), dims=3)'
                    dMo_dSdDdDdT[3 + (type - 1) * 5, :] = sum(mmult(sum((mmult((mmult(mmult(permutedims(-share_ijv[1, type], (1, 3, 2)), diagm(Params.w[:, 1, :])), permutedims(share_ijv[1, type], (3, 1, 2)))) .+ Diag, dDeltadThetanorm)) .* reshape(DistanceM, size(DistanceM, 1), 1, size(DistanceM, 2)), dims=1), SchoolsM.Ze[:, type]', "T") .* reshape(CweightsMTypes[:, type], 1, 1, :), dims=3)'
                    dMo_dSdDdDdT[4 + (type - 1) * 5, :] = sum(mmult((mmult((mmult(mmult(permutedims(-share_ijv[1, type], (1, 3, 2)), diagm(Params.w[:, 1, :])), permutedims(share_ijv[1, type], (3, 1, 2)))) .+ Diag, dDeltadThetanorm)), SchoolsM.Ze[:, type]', "T") .* reshape(CweightsMTypes[:, type], 1, 1, :), dims=3)'
                    dMo_dSdDdDdT[5 + (type - 1) * 5, :] = sum(mmult((mmult((mmult(mmult(permutedims(-share_ijv[1, type], (1, 3, 2)), diagm(Params.w[:, 1, :])), permutedims(share_ijv[1, type], (3, 1, 2)))) .+ Diag, dDeltadThetanorm)), SchoolsM.Zy[:, type]', "T") .* reshape(CweightsMTypes[:, type], 1, 1, :), dims=3)'
        
                end
        
                DMM_dTheta = dMo_dSdDdDdT + dMo_dTheta
        
            end
        end
        
        return S, Shares, share_ijv, dSdDelta, dSdTheta, MM, DMM_dTheta, dDeltadThetanorm
        
        end
        =#