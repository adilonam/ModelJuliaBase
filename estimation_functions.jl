


function GetParams(Estimation, Set, Theta2)
    Params = Dict{Symbol, Any}()

    if Set[:model] == "model 1"
        # Model 1
        BetaQEdu = Theta2[findall(Estimation.BetaEdu)]
        BetaQPoor = Theta2[findall(Estimation.BetaPoor)]
        AlphaEdu = Theta2[findall(Estimation.AlphaEdu)]
        AlphaPoor = Theta2[findall(Estimation.AlphaPoor)]
        LambdaEdu = Theta2[findall(Estimation.LambdaEdu)]
        LambdaPoor = Theta2[findall(Estimation.LambdaPoor)]

        Params[:betaK] = [BetaQPoor AlphaPoor LambdaEdu[1] + LambdaPoor;
                          0 0 LambdaEdu[1];
                          BetaQEdu + BetaQPoor AlphaEdu + AlphaPoor LambdaEdu[2] + LambdaPoor;
                          BetaQEdu AlphaEdu LambdaEdu[2]]
        
        BetaRC = Theta2[Estimation.BetaRC]
        
        Params[:betai] = hcat(BetaRC * Estimation.drawsN[:, 1], zeros(size(Estimation.drawsN[:, 1])), zeros(size(Estimation.drawsN[:, 1])))
        Params[:dbetai] = Estimation.drawsN[:, 1]
        Params[:w] = reshape(Estimation.drawsW, 1, 1, :)
        Params[:dbetamask] = [1 1]
        Params[:lb] = 0
        Params[:ub] = Inf
    elseif Set[:model] == "model 2"
        # Model 2
        BetaQEdu = Theta2[findall(Estimation.BetaQEdu)]
        BetaQPoor = Theta2[findall(Estimation.BetaQPoor)]
        AlphaEdu = Theta2[findall(Estimation.AlphaEdu)]
        AlphaPoor = Theta2[findall(Estimation.AlphaPoor)]
        BetaZeEdu = Theta2[findall(Estimation.BetaZeEdu)]
        BetaZePoor = Theta2[findall(Estimation.BetaZePoor)]
        BetaZpEdu = Theta2[findall(Estimation.BetaZpEdu)]
        BetaZpPoor = Theta2[findall(Estimation.BetaZpPoor)]
        LambdaEdu = Theta2[findall(Estimation.LambdaEdu)]
        LambdaPoor = Theta2[findall(Estimation.LambdaPoor)]

        Params[:betaK] = [BetaQPoor[1] AlphaPoor[1] (LambdaEdu[1] + LambdaPoor[1]) (BetaZeEdu[1] + BetaZePoor[1]) (BetaZpEdu[1] + BetaZpPoor[1]);
                          0 0 LambdaEdu[1] BetaZeEdu[1] BetaZpEdu[1];
                          (BetaQEdu[1] + BetaQPoor[1]) (AlphaEdu[1] + AlphaPoor[1]) (LambdaEdu[2] + LambdaPoor[1]) (BetaZeEdu[2] + BetaZePoor[1]) (BetaZpEdu[2] + BetaZpPoor[1]);
                          BetaQEdu[1] AlphaEdu[1] LambdaEdu[2] BetaZeEdu[2] BetaZpEdu[2]]
        
        BetaQRC = Theta2[findall(Estimation.BetaQRC)] 
        BetaZeRC = Theta2[findall(Estimation.BetaZeRC)] 
        BetaZpRC = Theta2[findall(Estimation.BetaZpRC)] 
        BetaZcorRC = Theta2[findall(Estimation.BetaZcorRC)]

        
        Params[:betai] = hcat(BetaQRC[1] * Estimation.drawsN[:, 1], BetaZeRC[1] * Estimation.drawsN[:, 2], BetaZcorRC[1] * Estimation.drawsN[:, 2] + BetaZpRC[1] * Estimation.drawsN[:, 3])
        Params[:dbetai] = Estimation.drawsN[:, 1:3]
        Params[:w] = reshape(Estimation.drawsW, 1, 1, :)
        Params[:dbetamask] = [1 1; 4 2; 5 2; 4 3]
        Params[:lb] = [0, 0, -Inf, 0]
        Params[:ub] = [Inf, Inf, Inf, Inf]
    end

    return Params
end
