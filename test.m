dSdDelta= dSdDelta+sum((    
    mmx('mult',
mmx('mult',permute(-share_ijv{1,type},[1 3 2]),diag(squeeze(Params.w(1,1,:))))
,permute(share_ijv{1,type},[3 1 2]))

)

.*reshape(CweightsMAll(:,type),1,1,[]),3);



