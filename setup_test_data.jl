
#====
1. load raw data and do any needed cleaning
at the end of this file we need the key tables in memory with no duplicates or missings,
in shape to run regressions / estimate models.
=#####


    # Initialize variables to be returned
    Schools = nothing
    Consumers = nothing
    Cweights = nothing
    Distance = nothing
    Moments = nothing
    Estimation = nothing
    markets = nothing

    # Description:
    # This code sets up the data for estimation

    # Input:
    #   - markets:
    #       - [xx xx]:  Any selection (markets go from 1 to 102)
    #       - [991]:    Small markets (under 100)
    #       - [992]:    Medium markets (under 300)
    #       - [993]:    All markets except Lima (101)
    #       - []:       All markets including Lima (102)
    #   - years:
    #       - 2014-2018
    #   - model:
    #       - 'model 1': {p,q,d}
    #       - 'model 2': {p,q,d,z}
    #   - moments:
    #       - 'moments 1': Micro,IV
    #       - 'moments 2': Micro,IV,RD
    #       - 'moments 3': Micro,IV,RD,MLE
    #   - Types:
    #       Type 1: Poor + Low Educ
    #       Type 2: Low Educ
    #       Type 3: Poor + High Educ
    #       Type 4: High Educ

    # Output: 'Data': Structure with the following fields:
    #     - schools:
    #         * sIndex [MarketId,Year,SchoolId,SchoolDistr,ChainId,Lat,Lon]
    #         * share [Share,ShareType,NumType]
    #         * charact [Price,Mu,Ze,Zy,Delta,Private,Charter,ForProfit,Religious,Emblematic]
    #         * Instruments [Instruments]
    #         * NormId [NormId]
    #     - consumers:
    #         * cIndex [marketId,Node,Lat,Lon]
    #     - draws:
    #         * nodes
    #         * weights
    #     - distance:
    #         * rows: schools, columns: consumer nodes
    #     - Cweights:
    #         * [rows: consumers, columns: years, third dim: types]
    #         * Cweights.All
    #         * Cweights.Types
    #     - Moments:
    #         * Moments.MM: Micromoments (MarketId,Year,Type,Charact,Moment,Variance,Nobs)
    #         * Moments.WMM: MM weighting matrix
    #         * Moments.RDM: RD Moments (MarketId,Year,Type,Node,ScoreAc,ScoreLd,Treat)

    # Fill the Estimation struct as per the MATLAB code


    school_data_path = joinpath(stemGit, "Data", "SchoolData_" * String(datadate) * ".csv")
    consumer_data_path = joinpath(stemGit, "Data", "ConsumerData_" * String(datadate) * ".csv")
    distance_data_path = joinpath(stemGit, "Data", "Distance.csv")
    mm_data_path = joinpath(stemGit, "Data", "MM.csv")
    wmm_data_path = joinpath(stemGit, "Data", "WMM.csv")
    rddata_data_path = joinpath(stemGit, "Data", "RDdata.csv")
    mat_data_path = joinpath(stemGit, "Data", "Cweights_" * String(datadate) * ".mat")

    Schools = CSV.read(school_data_path,DataFrame,ntasks=1,normalizenames=true)
    Consumers = CSV.read(consumer_data_path,DataFrame,ntasks=1,normalizenames=true)
    Distance = Matrix(CSV.read(distance_data_path,DataFrame; header=false))
    MM = CSV.read(mm_data_path,DataFrame,ntasks=1,normalizenames=true)
    WMM = Matrix(CSV.read(wmm_data_path,DataFrame; header=false))
    RDdata = CSV.read(rddata_data_path,DataFrame,ntasks=1,normalizenames=true)


    cwmat = matread(mat_data_path)
    CweightsAll = cwmat["Cweights"]["All"]
    CweightsTypes = cwmat["Cweights"]["Types"]

    function tabulate(arr)
        df = DataFrame(arr = arr)
        grouped = combine(groupby(df, :arr), nrow => :Count)
        return grouped
    end
    
    markets = []
    
    ### Choose Markets
    
    if set_markets == "small"
        D = tabulate(Schools[!,"MarketId"][:,1])
        pick = (D[:, :Count] ./ length(years) .< 100) .& (D[:, :Count] ./ length(years) .> 20)
        markets = D[pick, :arr]
    elseif set_markets == "medium"
        D = tabulate(Schools[!,"MarketId"][:,1])
        pick = (D[:, :Count] ./ length(years) .< 300) .& (D[:, :Count] ./ length(years) .>= 20)
        markets = D[pick, :arr]
    elseif set_markets == "large"
        D = tabulate(Schools[!,"MarketId"][:,1])
        pick = (D[:, :arr] .!= 312) .& (D[:, :Count] ./ length(years) .> 20)
        markets = D[pick, :arr]
    elseif set_markets == "all"
        D = tabulate(Schools[!,"MarketId"][:,1])
        pick = (D[:, :Count] ./ length(years) .> 20)
        markets = D[pick, :arr]
    end
    
    ###  Get feasible markets 
        # 1) have firm data, have shares and have moments
    
        # Get Row Index for moments
      
        global temp_mIndex = Int[]

        # Main code to avoid scoping issues

            # Iterate over markets and years
            for m in markets
                for y in years
                    indices = findall(x -> MM[!,"MarketId"][x] == m && MM[!,"Year"][x] == y && !isnan(MM[!,"Moment"][x]), 1:length(MM[!,"MarketId"]))
                    global temp_mIndex = vcat(temp_mIndex, indices)
                end
            end
    
    
        markets = unique(MM[!,"MarketId"][temp_mIndex])  # update markets to include only markets with moments
    
    # Get Row Index for schools
         
    global temp_sIndex = Int[]
         for m in markets
             for y in years
                global temp_sIndex = vcat(temp_sIndex, findall(x -> Schools[!,"MarketId"][x] == m && Schools[!,"Year"][x] == y && !isnan(Schools[!,"Share"][x]), 1:length(Schools[!,"MarketId"])))
             end
         end
     
         markets = unique(Schools[!,"MarketId"][temp_sIndex])  # update markets to include schools restrictions
     
    # Get Row Index for Consumers
    global  temp_cIndex = []
         for m in markets
            global temp_cIndex = vcat(temp_cIndex, findall(x -> Consumers[!,"MarketId"][x] == m, 1:length(Consumers[!,"MarketId"])))
         end
     
     # Get Row Index for RD sample
     global temp_RDIndex = []
         for m in markets
            global  temp_RDIndex = vcat(temp_RDIndex, findall(x -> RDdata[!,"MarketId"][x] == m, 1:length(RDdata[!,"MarketId"])))
         end
     
         # Adjust Peers model
    
         # Combine the Ze columns into matrices
         Schools.ZeB = [hcat(Schools.ZeB_1[i], Schools.ZeB_2[i], Schools.ZeB_3[i], Schools.ZeB_4[i]) for i in 1:nrow(Schools)]
         Schools.ZyB = [hcat(Schools.ZyB_1[i], Schools.ZyB_2[i], Schools.ZyB_3[i], Schools.ZyB_4[i]) for i in 1:nrow(Schools)]
         Schools.ZeLOO = [hcat(Schools.ZeLOO_1[i], Schools.ZeLOO_2[i], Schools.ZeLOO_3[i], Schools.ZeLOO_4[i]) for i in 1:nrow(Schools)]
         Schools.ZyLOO = [hcat(Schools.ZyLOO_1[i], Schools.ZyLOO_2[i], Schools.ZyLOO_3[i], Schools.ZyLOO_4[i]) for i in 1:nrow(Schools)]
         Schools.ShareType = [hcat(Schools.ShareType_1[i], Schools.ShareType_2[i], Schools.ShareType_3[i], Schools.ShareType_4[i]) for i in 1:nrow(Schools)]
         Schools.Instruments = [hcat(Schools.Instruments_1[i], Schools.Instruments_2[i], Schools.Instruments_3[i], Schools.Instruments_4[i],Schools.Instruments_5[i], Schools.Instruments_6[i], Schools.Instruments_7[i], Schools.Instruments_8[i], Schools.Instruments_9[i], Schools.Instruments_10[i], Schools.Instruments_11[i], Schools.Instruments_12[i], Schools.Instruments_13[i], Schools.Instruments_14[i], Schools.Instruments_15[i], Schools.Instruments_16[i], Schools.Instruments_17[i]) for i in 1:nrow(Schools)]
    
         select!(Schools, Not([:ZeB_1, :ZeB_2, :ZeB_3, :ZeB_4]))
         select!(Schools, Not([:ZyB_1, :ZyB_2, :ZyB_3, :ZyB_4]))
         select!(Schools, Not([:ZeLOO_1, :ZeLOO_2, :ZeLOO_3, :ZeLOO_4]))
         select!(Schools, Not([:ZyLOO_1, :ZyLOO_1, :ZyLOO_1, :ZyLOO_1]))
         select!(Schools, Not([:ShareType_1, :ShareType_2, :ShareType_3, :ShareType_4]))
         select!(Schools, Not([:Instruments_1, :Instruments_2, :Instruments_3, :Instruments_4,:Instruments_5, :Instruments_6, :Instruments_7, :Instruments_8,:Instruments_9, :Instruments_10, :Instruments_11, :Instruments_12,:Instruments_13, :Instruments_14, :Instruments_15, :Instruments_16, :Instruments_17]))
    
    # Remove the original Ze1, Ze2, Ze3, Ze4 columns
    
    if peersmodel == "basic"
        select!(Schools, Not(["ZeLOO", "ZyLOO"]))
        rename!(Schools, "ZeB" => "Ze", "ZyB" => "Zy")
    elseif peersmodel == "leave one out"
        select!(Schools, Not(["ZeB", "ZyB"]))
        rename!(Schools, "ZeLOO" => "Ze", "ZyLOO" => "Zy")
    end
    
    # Put the exogenous regressors together
    Schools.XX_exogenous = [hcat(Schools.Private[i], Schools.Charter[i], Schools.ForProfit[i], Schools.Religious[i], Schools.Emblematic[i]) for i in 1:nrow(Schools)]
    select!(Schools, Not(["Private", "Charter", "ForProfit", "Religious", "Emblematic"]))
    
    # Now get final group
    Schools = Schools[temp_sIndex, :]
    Consumers = Consumers[temp_cIndex, :]
    Types = CweightsTypes[temp_cIndex, :, :]
    All = CweightsAll[temp_cIndex, :, :]
    Distance = Distance[temp_sIndex, temp_cIndex]
    MM = MM[temp_mIndex, :]
    WMM = WMM[temp_mIndex, temp_mIndex]
    RDdata = RDdata[temp_RDIndex, :]
    
    # Standardize Mu, Price, Distance, Ze, and Zy by Market
    for m in markets
        for y in years
            mu_idx = findall(x -> Schools[!,:MarketId][x] == m && Schools[!,:Year][x] == y, 1:length(Schools[!,:MarketId]))
            norm_mu_idx = findall(x -> Schools[!,:MarketId][x] == m && Schools[!,:Year][x] == y && Schools[!,:NormId][x] == 1, 1:length(Schools[!,:MarketId]))
    
            Schools[!,:Mu][mu_idx] .= (Schools[!,:Mu][mu_idx] .- (Schools[!,:Mu][norm_mu_idx])) ./ std(Schools[!,:Mu][mu_idx])
            Schools[!,:Price][mu_idx] .= (Schools[!,:Price][mu_idx] .- (Schools[!,:Price][norm_mu_idx])) ./ std(Schools[!,:Price][mu_idx])
            
            TempZe =zeros(size(mu_idx,1), 4)
            TempZy =zeros(size(mu_idx,1), 4)

            for k in 1:4
               TempZe[:,k] .=[v[1] for v in (col .- (Schools[!,:Ze][norm_mu_idx[1]][1,k]) for col in [matrix[:, k] for matrix in Schools[!, :Ze][mu_idx]]) ./ std([matrix[:, k] for matrix in Schools[!, :Ze][mu_idx]])]
               TempZy[:,k] .= [v[1] for v in (col .- (Schools[!,:Zy][norm_mu_idx[1]][1,k]) for col in [matrix[:, k] for matrix in Schools[!, :Zy][mu_idx]]) ./ std([matrix[:, k] for matrix in Schools[!, :Zy][mu_idx]])]
            end


            for i in 1:length(mu_idx)
                index = mu_idx[i]
                Schools[!, :Ze][index] = transpose(TempZe[i, :])  # Assign the i-th row of TempZe to the first column of the matrix at index
                Schools[!, :Zy][index] = transpose(TempZy[i, :])
            end

        
     
    
            distance_idx = findall(x -> Schools[!,:MarketId][x] == m && Schools[!,:Year][x] == y, 1:length(Schools[!,:MarketId]))
            norm_distance_idx = findall(x -> Schools[!,:MarketId][x] == m && Schools[!,:Year][x] == y && Schools[!,:NormId][x] == 1, 1:length(Schools[!,:MarketId]))
    
            Distance[distance_idx, findall(x -> Consumers[!,:MarketId][x] == m, 1:length(Consumers[!,:MarketId]))] .= 
                (Distance[distance_idx, findall(x -> Consumers[!,:MarketId][x] == m, 1:length(Consumers[!,:MarketId]))] .- 
                minimum(abs.(Distance[norm_distance_idx, findall(x -> Consumers[!,:MarketId][x] == m, 1:length(Consumers[!,:MarketId]))]))) ./ 
                std(Distance[distance_idx, findall(x -> Consumers[!,:MarketId][x] == m, 1:length(Consumers[!,:MarketId]))], dims=1)
        end
    end



# Define your data


# Step 1: Replace 0 with NaN in ChainId
Schools.ChainId = float(Schools.ChainId)
Schools.ChainId[Schools.ChainId .== 0] .= NaN

# Step 2: Create dummy variables for ChainId
unique_chain_ids = unique(skipmissing(Schools.ChainId))
dummy_df = DataFrame()

for id in unique_chain_ids
    dummy_column = (Schools.ChainId .== id) .|> Int
    insertcols!(dummy_df, Symbol("ChainId_$id") => dummy_column)
end

# Combine dummy_df with the Schools DataFrame
Estimation = hcat(Schools, dummy_df)

# Step 3: Initialize Rfe and Cfe
Rfe = Int[]
Cfe = Int[]

MarketID = unique(Schools.MarketId)
TimeID = unique(Schools.Year)

global counter = 1

# Step 4: Iterate over MarketID and TimeID
for idm in MarketID
    for idt in TimeID
        rj = findall(x -> Schools[!,:MarketId][x] == idm && Schools[!,:Year][x] == idt, 1:length(Schools[!,:MarketId]))
        append!(Rfe, rj)
        append!(Cfe, fill(counter, length(rj)))
        global counter += 1
    end
end

# Step 5: Create sparse matrix for MarketYearFE

MarketYearFE = sparse(Rfe, Cfe, ones(Int, length(Rfe)))

# Fix weights for micro moments

MM.SanityCheck = .!((abs.(MM.Moment) .== 0) .| (MM.MobsN .< 30) .| ((MM.Charact .== 3) .& (MM.Year .== 2014)))

temp = MM.MVar[MM.SanityCheck] .^ (-1)

if any(isinf.(temp))
    error("Weights for micromoments include Inf")
end
if any(temp .== 0)
    error("Weights for micromoments include zeros")
end

for ii in 1:5
    tempweight = MM.Charact[MM.SanityCheck] .== ii
    temp[tempweight] .= temp[tempweight] ./ sum(temp[tempweight])
end

weightsM = temp
WMM = sparse(1:length(MM.MVar[MM.SanityCheck]), 1:length(MM.MVar[MM.SanityCheck]), weightsM)

# Prepare RD Data
DataY = fill(NaN, nrow(RDdata), 6)

global locb1 =  indexin( RDdata.School, Schools.SchoolId)

DataY[:, 1] .= Schools[!, :Mu][locb1]
DataY[:, 2] .= Schools.Price[locb1]
DataY[:, 4] .= Schools.Price[locb1]
for l in 1:length(locb1) 
    DataY[l, 5] = Schools[!,:Ze][locb1[l]][1,1]
    DataY[l, 6] = Schools[!,:Zy][locb1[l]][1,1]
end

for m in unique(RDdata.MarketId)
    for y in unique(RDdata.Year)
        
        # By Market, by year
        SchoolsM = Schools[(Schools.MarketId .== m) .& (Schools.Year .== y), :]
        DistanceM = Distance[(Schools.MarketId .== m) .& (Schools.Year .== y), (Consumers.MarketId .== m)]

        
        # Only for Model 2
        rdIndex = findall(x -> RDdata[!,:MarketId][x] == m && RDdata[!,:Year][x] == y, 1:length(RDdata[!,:MarketId]))
        nodes = RDdata.Node[rdIndex]

        global locb2 = indexin( RDdata.School,SchoolsM.SchoolId)
        global locb2 = filter(!isnothing, locb2)
        
        global temp = Float64[]
        
        for i in 1:length(nodes)
            push!(temp, DistanceM[locb2[i], nodes[i]])
        end
        
        DataY[rdIndex, 3] .= temp
    end
end


# RD regression
RDdataZ = hcat(RDdata.ThresCross, ones(nrow(RDdata), 1), RDdata.ScoreAc, RDdata.ScoreLd, RDdata.MiScoreLd, RDdata.ScoreAc .* RDdata.ScoreLd, RDdata.ScoreAc .* RDdata.MiScoreLd)
RDdataX = hcat(RDdata.Treat, ones(nrow(RDdata), 1), RDdata.ScoreAc, RDdata.ScoreLd, RDdata.MiScoreLd, RDdata.ScoreAc .* RDdata.ScoreLd, RDdata.ScoreAc .* RDdata.MiScoreLd)

RDBeta = zeros(6, size(RDdataZ,2))

for i in 1:6
    RDBeta[i, :] = inv(RDdataZ' * RDdataX) * RDdataZ' * DataY[:, i]
end

WRDM = (1/6) * I(6)


mutable struct M
    MM::DataFrame
    WMM::SparseMatrixCSC{Float64, Int64}
    RDdata::DataFrame
    RDdataZ::Matrix{Float64}
    RDdataX::Matrix{Float64}
    RDBeta::Matrix{Float64}
end

Moments = M(MM,WMM,RDdata,RDdataZ,RDdataX,RDBeta)

mutable struct CW
    All::Array{Float64, 3}
    Types::Array{Float64, 3}
  end

Cweights = CW(All,Types)


# Define structures
mutable struct Settings
    model::String
end

mutable struct EstimationStruct
    W_IV::Float64
    W_MM::Float64
    W_RDM::Float64
     drawsN::Matrix{Float64}
     drawsW::Vector{Float64}
     BetaEdu::Vector{Bool}
     BetaPoor::Vector{Bool}
     AlphaEdu::Vector{Bool}
     AlphaPoor::Vector{Bool}
     LambdaEdu::Vector{Bool}
     LambdaPoor::Vector{Bool}
     BetaRC::Vector{Bool}
     ThetaMask::Vector{Vector{Int}}
     TypesTheta::Vector{Vector{Int}}
     BetaQEdu::Vector{Bool}
     BetaQPoor::Vector{Bool}
     BetaZeEdu::Vector{Bool}
     BetaZePoor::Vector{Bool}
     BetaZpEdu::Vector{Bool}
     BetaZpPoor::Vector{Bool}
     BetaQRC::Vector{Bool}
     BetaZeRC::Vector{Bool}
     BetaZpRC::Vector{Bool}
     BetaZcorRC::Vector{Bool}
     Theta1Names::Vector{String}
     AdjustmentObjMM::Float64
     AdjustmentObjIV::Float64
end


# Initialize Estimation
Estimation = EstimationStruct(
    1/3, 1/3, 1/3, 
    Matrix{Float64}(undef, 0, 0)
    , Float64[], 
    Bool[], Bool[], Bool[], Bool[], Bool[], Bool[], Bool[], 
    Vector{Vector{Int}}(), Vector{Vector{Int}}(), 
    Bool[], Bool[], Bool[], Bool[], Bool[], Bool[], Bool[], Bool[], Bool[], Bool[], String[], 0.0, 0.0)

    if model == :model_1

 # Placeholder for nwspgr function
 n_nodes = 7
 Estimation.drawsN, Estimation.drawsW = gausshermite(n_nodes)

 # Model 1 (8 parameters)
 Estimation.BetaEdu = [true, false, false, false, false, false, false, false]
 Estimation.BetaPoor = [false, true, false, false, false, false, false, false]
 Estimation.AlphaEdu = [false, false, true, false, false, false, false, false]
 Estimation.AlphaPoor = [false, false, false, true, false, false, false, false]
 Estimation.LambdaEdu = [false, false, false, false, true, true, false, false]
 Estimation.LambdaPoor = [false, false, false, false, false, false, true, false]
 Estimation.BetaRC = [false, false, false, false, false, false, false, true]

 Estimation.ThetaMask = [[0, 1], [0, 1], [0, 2], [0, 2], [0, 3], [0, 3], [0, 3], [1, 1]]
 Estimation.TypesTheta = [[2, 4, 5, 7, 8], [5, 8], [1, 2, 3, 4, 6, 8], [1, 3, 6, 8]]

elseif model == :model_2
 # Placeholder for nwspgr function
 n_nodes = 5
 n_rc = 3

 function multidimensional_gausshermite(d, n)
    x, w = gausshermite(n)
    x=x*sqrt(2)
    w=w/sqrt(pi)
    grid_points = collect(product(ntuple(_ -> x, d)...))
    grid_weights = collect(product(ntuple(_ -> w, d)...))

    # Convert grid points to array of arrays and calculate the product of weights
    nodes = hcat(map(collect, grid_points)...)'
    temp = hcat(map(collect, grid_weights)...)'
    weights = temp[:,1].*temp[:,2]

    return nodes, weights
end

if Set[:model] =="model 2" && Set[:quadsource] == "Julia"
    Estimation.drawsN, Estimation.drawsW = multidimensional_gausshermite(n_rc,n_nodes)
elseif Set[:model] =="model 2" && Set[:quadsource] == "Matlab" &&  n_nodes == 5
    include("nwspgr.jl")
    Estimation.drawsN, Estimation.drawsW = nwspgr()
end


 # Model 2 (17 parameters)
 Estimation.BetaQEdu = [true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]
 Estimation.BetaQPoor = [false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]
 Estimation.AlphaEdu = [false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false]
 Estimation.AlphaPoor = [false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false]
 Estimation.BetaZeEdu = [false, false, false, false, true, true, false, false, false, false, false, false, false, false, false, false, false]
 Estimation.BetaZePoor = [false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false]
 Estimation.BetaZpEdu = [false, false, false, false, false, false, false, true, true, false, false, false, false, false, false, false, false]
 Estimation.BetaZpPoor = [false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false]
 Estimation.LambdaEdu = [false, false, false, false, false, false, false, false, false, false, true, true, false, false, false, false, false]
 Estimation.LambdaPoor = [false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false]
 Estimation.BetaQRC = [false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false]
 Estimation.BetaZeRC = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false]
 Estimation.BetaZpRC = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false]
 Estimation.BetaZcorRC = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true]

 Estimation.Theta1Names = ["BetaQEdu", "BetaQPoor", "AlphaEdu", "AlphaPoor", "BetaZeEdu", "BetaZeEdu", "BetaZePoor", "BetaZpEdu", "BetaZpEdu", "BetaZpPoor", "LambdaEdu", "LambdaEdu", "LambdaPoor", "BetaQRC", "BetaZeRC", "BetaZpRC", "BetaZcorRC"]

 Estimation.ThetaMask = [
        [0, 1], [0, 1], [0, 2], [0, 2], [0, 4], [0, 4], [0, 4], [0, 5], [0, 5], [0, 5], [0, 3], [0, 3], [0, 3], [1, 1], [1, 4], [1, 5], [1, 5],
        [1, 2], [1, 2], [2, 4], [2, 4], [3, 4], [3, 4], [3, 4], [4, 5], [4, 5], [4, 5], [5, 3], [5, 3], [5, 3], [1, 1], [4, 5], [5, 5], [5, 5],
        [1, 2], [2, 3], [3, 2]
    ]

    Estimation.TypesTheta = [[2,4,5,7,8,10,11,13,14,15,16,17],[5,8,11,14,15,16,17],[1,2,3,4,6,7,9,10,12,13,14,15,16,17],[1,3,6,9,12,14,15,16,17]]

    # Estimation.TypesTheta[1] = [2,4,5,7,8,10,11,13,14,15,16,17]
    # Estimation.TypesTheta[2]  = [5,8,11,14,15,16,17]
    # Estimation.TypesTheta[3]  = [1,2,3,4,6,7,9,10,12,13,14,15,16,17]
    # Estimation.TypesTheta[4]  = [1,3,6,9,12,14,15,16,17]

end

Estimation.AdjustmentObjMM = 0.0
Estimation.AdjustmentObjIV = 0.0


