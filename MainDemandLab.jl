using Random, Distributions, LinearAlgebra, StatsBase, DataFrames, CSV, Optim, SparseArrays, FileIO, JLD2, MAT
using StaticArrays, FastGaussQuadrature, MLJ, StatsFuns, ForwardDiff, GLM
using IterTools: product
#using ComponentArrays, Lux

const set_markets = "small"
years = [2014, 2018]
const model       =   :model_2;
const method      =   :squarem;
const test        =   true
const datadate    =   :Fake_Test
const peersmodel = "leave one out"
const quadmat = 1

# Setting parameters
Set                 = Dict()
Set[:mW]            = 0.5
Set[:tol_inner]     = 10^-12
Set[:tol_outer]     = 10^-12
Set[:model]         ="model 2"
Set[:quad]          = 5
Set[:quadsource]    = "Matlab"

include("setup_machines.jl")
include("estimation_functions.jl")

if test 
include("setup_test_data.jl") 
const datadate    =   :Fake_Test
else
include("setup_real_data.jl") 
end


if Set[:model] == "model 1"
    Set[:Th2_0] = [0.5, -0.2, -0.3, 0.2, -0.2, 0.3, 0.5, 0.3]
elseif Set[:model] == "model 2"
    Set[:Th2_0] = [0.5, -0.2, -0.3, 0.2, 0.8, -0.3, -0.6, 0.3, -0.2, 0.3, 0.5, 0.4, 0.6, 0.3, 0.1, 0.4, 0.2]
end

Theta2 = Set[:Th2_0]

Delta0 = ones(size(Schools, 1))
Schools.Share .= zeros(size(Schools.Share))

Params = GetParams(Estimation, Set, Theta2)


m=1
y=2014

DeltaM = filter(row -> row.MarketId == m && row.Year == y, Schools)[:, :Delta]
SchoolsM = filter(row -> row.MarketId == m && row.Year == y, Schools)
DistanceM = Distance[(Schools.MarketId .== m) .& (Schools.Year .== y), Consumers.MarketId .== m]
CweightsMAll = Cweights.All[Consumers.MarketId .== m, 2, :]
CweightsMTypes = Cweights.Types[Consumers.MarketId .== m, 2, :]


for m in markets
    for y in years
        DeltaM = filter(row -> row.MarketId == m && row.Year == y, Schools)[:, :Delta]
        SchoolsM = filter(row -> row.MarketId == m && row.Year == y, Schools)
        DistanceM = Distance[(Schools.MarketId .== m) .& (Schools.Year .== y), Consumers.MarketId .== m]
        CweightsMAll = Cweights.All[Consumers.MarketId .== m, 2, :]
        CweightsMTypes = Cweights.Types[Consumers.MarketId .== m, 2, :]
        #Schools[Schools.MarketId .== m .& Schools.Year .== y, :Share] .= RC_shares(DeltaM, SchoolsM, DistanceM, CweightsMAll, CweightsMTypes, Estimation, Set, Params)
    end
end

# Clearing temporary variables
nothing # This is the equivalent of `clear` in MATLAB, but Julia manages memory automatically

Schools[:Delta] .= SolveShares(Delta0, Schools, Consumers, Distance, Cweights, Estimation, Set, Theta2)




# const descriptivedatapath = "/Users/akapor/Dropbox/CHI_School_Search/1_Data/16_DescriptiveAnalysis/csv" 
# const datapath = "/Users/akapor/Dropbox/CHI_School_Search/1_Data/18_Model/csv"
# const workpath = "/Users/akapor/Dropbox/CHI_School_Search/1_Data/18_Model/working"
# const outputpath = "/Users/akapor/Dropbox/CHI_School_Search/1_Data/18_Model/output"
# const tablefigurepath = "/Users/akapor/Documents/GitHub/BiasedBeliefsSearchSchoolChoice/2_Writeup/Presentation slides/tables_figures"

#######################################
# these need to be command-line args
const keep_mothereduc = :low
const boot = 102 #bootstrap rep, 0 if point estimates
########################################

const run_step1 = true
const run_step2 = true
const run_step3 = true 
const run_step4 = true
const run_describesearchcosts = true

@assert isa(keep_mothereduc,Symbol)
@assert isa(boot,Int)
@assert keep_mothereduc in [:low,:high,:all]

if boot > 0
    b_tag = "_bootstrap$boot"
else
    b_tag = ""
end
suffix_short = (keep_mothereduc == :all ? "" : keep_mothereduc==:high ? "_mothereduc_h" : "_mothereduc_l")
suffix = suffix_short*b_tag

include("estimationfunctions.jl") #mcmc general functions here
include("step1ParameterEstimationFunctions.jl") #specific Gibbs steps for our model here
include("step1RegressionParameterUpdates.jl") #specific Gibbs steps for our model here
include("step1VarianceUpdates.jl") #specific Gibbs steps for our model here
include("counterfactualFunctions.jl")
include("step2functions.jl")
include("misspecifiedModelFunctions.jl")
include("SearchFunctions.jl") 


include("load_data.jl")
include("setup_estimationdatastructures.jl")
include("descriptives.jl")

if run_step1
    include("run_step1.jl")
else
    savedresults = JLD2.load("$workpath/current_state$suffix.jld2") #history, CFs, CC, LV
    LVs = savedresults["history"]
    lvmean = savedresults["lvmean"]
    lvvar = savedresults["lvvar"]
    DFcfs = savedresults["CFs"]
    CC = savedresults["CC"]
    LV = savedresults["LV"]
    dfJ = savedresults["dfJ"]
    ts = savedresults["ts"]
    inds_keep = savedresults["inds_keep"]
    savedresults = nothing
end