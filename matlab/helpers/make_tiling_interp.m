
function [V_flat_merged, T_merged] = make_tiling_interp(V_flat, T)
% axis equal
xs=[0,4,0,-4,4,-4,-4,0 ,4];
ys=[0,4,4,0, 0, 4,-4,-4,-4] ;
V_flat_merged = [];
T_merged = [];
T_shift = size(V_flat,1);
count = 0;
for j=1:size(xs,2)
    x=xs(j);
    y=ys(j);
    
    
    for i=1:4
        cx=x ;
        cy=y ;
        ang = (1-i)*pi/2;
        R=[cos(ang) -sin(ang) ; sin(ang) cos(ang)];
        V_flat_rot = R*V_flat' ;
        trans = zeros(size(V_flat_rot));
        trans(1,:)=cx;
        trans(2,:)=cy;
        if i>2
            cx=cx-2;
            trans(1,:)=cx;
        end
        if i>1&&i<4
            cy=cy-2;
            trans(2,:)=cy ;
        end
        
        V_flat_rt = V_flat_rot+trans;
        V_flat_merged = [V_flat_merged ,V_flat_rt];
        T_merged = [T_merged ; T+count*T_shift];
        count = count+1;
    end
   
end

end

