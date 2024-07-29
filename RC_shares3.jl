function RC_shares(DeltaM, SchoolsM, DistanceM, CweightsMAll, CweightsMTypes, Estimation, Set, Params, nargout)
    # Initialize outputs
    S = zeros(size(SchoolsM, 1))
    Shares = fill(NaN, size(SchoolsM, 1), 4)
    share_ijv = Vector{Any}(undef, 1, 4)
    
    # Conditional allocation for optional outputs
    dSdDelta = nothing
    dSdTheta = nothing
    moM = nothing
    dMo_dTheta = nothing
    
    if nargout > 3
        dSdDelta = zeros(size(SchoolsM, 1), size(SchoolsM, 1))
        
        if nargout > 4
            dSdTheta = zeros(size(SchoolsM, 1), size(Estimation.ThetaMask, 1))
            
            if nargout > 5
                moM = []
                
                if nargout > 6
                    dMo_dTheta = zeros(5*4, size(Estimation.ThetaMask, 1))
                end
            end
        end
    end
    
    for type in 1:4
        if Set.model == "model 1"
            UoType = SchoolsM.Mu .* Params.betaK[type, 1] + SchoolsM.Price .* Params.betaK[type, 2] + DistanceM .* Params.betaK[type, 3]
            Uv = SchoolsM.Mu .* Params.betai[:, 1]'
        elseif Set.model == "model 2"
            UoType = SchoolsM.Mu .* Params.betaK[type, 1] + SchoolsM.Price .* Params.betaK[type, 2] + DistanceM .* Params.betaK[type, 3] +
                     SchoolsM.Ze[:, type] .* Params.betaK[type, 4] + SchoolsM.Zy[:, type] .* Params.betaK[type, 5]
            Uv = SchoolsM.Mu .* Params.betai[:, 1]' + SchoolsM.Ze[:, type] .* Params.betai[:, 2]' + SchoolsM.Zy[:, type] .* Params.betai[:, 3]'
        end
        
        maxuij = maximum(DeltaM) + maximum(UoType) + maximum(Uv)
        num = exp.(DeltaM .+ UoType .+ Uv .- maxuij)
        share_ijv[1, type] = num ./ sum(num, dims=1)
        share_ij = sum(share_ijv[1, type] .* Params.w, dims=3)
        Shares[:, type] = sum(share_ij .* CweightsMTypes[:, type]', dims=2)
        S += sum(share_ij .* CweightsMAll[:, type]', dims=2)
        
        if nargout > 5
            if Set.model == "model 1"
                moMy = (hcat(share_ij' * hcat(SchoolsM.Mu, SchoolsM.Price), sum(share_ij .* DistanceM, dims=1)')') * CweightsMTypes[:, type]
                moM = vcat(moM, moMy)
            elseif Set.model == "model 2"
                moMy = hcat(share_ij' * [SchoolsM.Mu SchoolsM.Price] , Matrix(sum(share_ij .* DistanceM, dims=1))',share_ij' * [SchoolsM.Ze[i][1, type] for i in 1:size(SchoolsM,1)],share_ij' *[SchoolsM.Zy[i][1, type] for i in 1:size(SchoolsM,1)])'* CweightsMTypes[:, type]
                moM = vcat(moM, moMy)
            end
        end

        
        if nargout > 3
            for n in 1:length(CweightsMTypes[:, type])
                dSdDelta += -share_ijv[1, type][:, n, :] * diagm(Params.w[1, 1, :]) * share_ijv[1, type][:, n, :]' * CweightsMAll[n, type]
            end
        end
        
        if nargout > 4
            for i in Estimation.TypesTheta[type, 1]
                if Estimation.ThetaMask[i, 2] == 1
                    if Estimation.ThetaMask[i, 1] == 0
                        dSdTheta[:, i] += (share_ijv[1, type] .* (SchoolsM.Mu .- SchoolsM.Mu' * share_ijv[1, type])' * CweightsMAll[:, type])' * Params.w[1, 1, :]
                        
                        if nargout > 6
                            dMo_dTheta[1 + (type - 1) * 5, i] = (share_ijv[1, type] .* (SchoolsM.Mu .- SchoolsM.Mu' * share_ijv[1, type]) * SchoolsM.Mu' * CweightsMTypes[:, type])' * Params.w[1, 1, :]
                            dMo_dTheta[2 + (type - 1) * 5, i] = (share_ijv[1, type] .* (SchoolsM.Mu .- SchoolsM.Mu' * share_ijv[1, type]) * SchoolsM.Price' * CweightsMTypes[:, type])' * Params.w[1, 1, :]
                            dMo_dTheta[3 + (type - 1) * 5, i] = (sum((share_ijv[1, type] .* (SchoolsM.Mu .- SchoolsM.Mu' * share_ijv[1, type])) .* DistanceM, dims=1) * CweightsMTypes[:, type])' * Params.w[1, 1, :]
                            dMo_dTheta[4 + (type - 1) * 5, i] = (share_ijv[1, type] .* (SchoolsM.Mu .- SchoolsM.Mu' * share_ijv[1, type]) * SchoolsM.Ze[:, type]' * CweightsMTypes[:, type])' * Params.w[1, 1, :]
                            dMo_dTheta[5 + (type - 1) * 5, i] = (share_ijv[1, type] .* (SchoolsM.Mu .- SchoolsM.Mu' * share_ijv[1, type]) * SchoolsM.Zy[:, type]' * CweightsMTypes[:, type])' * Params.w[1, 1, :]
                        end
                    elseif Estimation.ThetaMask[i, 1] == 1
                        dSdTheta[:, i] += (share_ijv[1, type] .* Params.dbetai[:, Estimation.ThetaMask[i, 3]]' .* (SchoolsM.Mu .- SchoolsM.Mu' * share_ijv[1, type])' * CweightsMAll[:, type])' * Params.w[1, 1, :]
                        
                        if nargout > 6
                            dMo_dTheta[1 + (type - 1) * 5, i] = (share_ijv[1, type] .* Params.dbetai[:, Estimation.ThetaMask[i, 3]]' .* (SchoolsM.Mu .- SchoolsM.Mu' * share_ijv[1, type]) * SchoolsM.Mu' * CweightsMTypes[:, type])' * Params.w[1, 1, :]
                            dMo_dTheta[2 + (type - 1) * 5, i] = (share_ijv[1, type] .* Params.dbetai[:, Estimation.ThetaMask[i, 3]]' .* (SchoolsM.Mu .- SchoolsM.Mu' * share_ijv[1, type]) * SchoolsM.Price' * CweightsMTypes[:, type])' * Params.w[1, 1, :]
                            dMo_dTheta[3 + (type - 1) * 5, i] = (sum((share_ijv[1, type] .* Params.dbetai[:, Estimation.ThetaMask[i, 3]]' .* (SchoolsM.Mu .- SchoolsM.Mu' * share_ijv[1, type])) .* DistanceM, dims=1) * CweightsMTypes[:, type])' * Params.w[1, 1, :]
                            dMo_dTheta[4 + (type - 1) * 5, i] = (share_ijv[1, type] .* Params.dbetai[:, Estimation.ThetaMask[i, 3]]' .* (SchoolsM.Mu .- SchoolsM.Mu' * share_ijv[1, type]) * SchoolsM.Ze[:, type]' * CweightsMTypes[:, type])' * Params.w[1, 1, :]
                            dMo_dTheta[5 + (type - 1) * 5, i] = (share_ijv[1, type] .* Params.dbetai[:, Estimation.ThetaMask[i, 3]]' .* (SchoolsM.Mu .- SchoolsM.Mu' * share_ijv[1, type]) * SchoolsM.Zy[:, type]' * CweightsMTypes[:, type])' * Params.w[1, 1, :]
                        end
                    end
                elseif Estimation.ThetaMask[i, 2] == 2
                    if Estimation.ThetaMask[i, 1] == 0
                        dSdTheta[:, i] += (share_ijv[1, type] .* (SchoolsM.Price .- SchoolsM.Price' * share_ijv[1, type])' * CweightsMAll[:, type])' * Params.w[1, 1, :]
                        
                        if nargout > 6
                            dMo_dTheta[2 + (type - 1) * 5, i] = (share_ijv[1, type] .* (SchoolsM.Price .- SchoolsM.Price' * share_ijv[1, type]) * SchoolsM.Price' * CweightsMTypes[:, type])' * Params.w[1, 1, :]
                        end
                    elseif Estimation.ThetaMask[i, 1] == 1
                        dSdTheta[:, i] += (share_ijv[1, type] .* Params.dbetai[:, Estimation.ThetaMask[i, 3]]' .* (SchoolsM.Price .- SchoolsM.Price' * share_ijv[1, type])' * CweightsMAll[:, type])' * Params.w[1, 1, :]
                        
                        if nargout > 6
                            dMo_dTheta[2 + (type - 1) * 5, i] = (share_ijv[1, type] .* Params.dbetai[:, Estimation.ThetaMask[i, 3]]' .* (SchoolsM.Price .- SchoolsM.Price' * share_ijv[1, type]) * SchoolsM.Price' * CweightsMTypes[:, type])' * Params.w[1, 1, :]
                        end
                    end
                end
            end
        end
    end
    
    return S, Shares, share_ijv, dSdDelta, dSdTheta, moM, dMo_dTheta
end
