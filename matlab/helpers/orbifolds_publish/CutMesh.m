classdef CutMesh < handle
    
    
    properties
        V;
        T;
        pathPairs;
        cutIndsToUncutInds;
        uncutIndsToCutInds;
    end
    
    methods
        function obj=CutMesh(V,T,pathPairs,cutIndsToUncutInds,uncutIndsToCutInds)
            obj.V=V;
            obj.T=T;
            obj.pathPairs=pathPairs;
            obj.cutIndsToUncutInds=cutIndsToUncutInds;
            obj.uncutIndsToCutInds=uncutIndsToCutInds;
        end
    
    end
    
end

