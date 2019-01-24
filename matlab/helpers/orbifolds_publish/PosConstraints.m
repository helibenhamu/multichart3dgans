classdef PosConstraints<handle
    
    properties
        A;
        b=[];
        
    end
    
    methods
        function obj=PosConstraints(nvars)
            obj.A=sparse(0,nvars*2);
        end
        function addConstraint(obj,inds,w,rhs)
            assert(length(rhs)==2);
            obj.A(end+1,inds*2-1)=w;
            obj.A(end+1,inds*2)=w;
            
            obj.b=[obj.b;rhs];
        end
        function newConstraint(obj)
            obj.A(end+1,1)=0;
            obj.b(end+1)=0;
        end
        function addLineConstraint(obj,ind,n,offset)
            obj.A(end+1,ind*2-1:ind*2)=n;
            
            obj.b=[obj.b;offset];
        end
        function addTransConstraints(obj,sinds,tinds,T)
            assert(length(sinds)==length(tinds));
%             assert(size(T,1)==length(delta));
            if sinds(end)==tinds(end)
                sinds=sinds(end:-1:1);
                tinds=tinds(end:-1:1);
            end
%             assert(sinds(1)==tinds(1));
            
            
            for ind=2:length(sinds)
                for y=1:2
                    
                    
                    obj.A(end+1,sinds(ind)*2+[-1,0])=T(y,:);
                    obj.A(end,sinds(1)*2+[-1,0])=obj.A(end,sinds(1)*2+[-1,0])-T(y,:);
                    obj.A(end,tinds(ind)*2+y-2)=obj.A(end,tinds(ind)*2+y-2)-1;
                    obj.A(end,tinds(1)*2+y-2)=obj.A(end,tinds(1)*2+y-2)+1;
                    
                    obj.b=[obj.b;0];
                end
            end
        end
    end
    
end

