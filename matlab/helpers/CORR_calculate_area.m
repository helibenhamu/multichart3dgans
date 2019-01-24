function [area, E] = CORR_calculate_area(F,V1)

% Calculate the area of the surface

E=zeros(size(F,1),1);

for k=1:size(F,1)
   f = F(k,:);
   
   v1 = V1(f(1),:);
   v2 = V1(f(2),:);
   v3 = V1(f(3),:);
   
   %new code - like in GC
   u = v2-v1;
   v = v3-v1;
   
   if size(u,2)==2
       u = [u,1];
       v =  [v,1];
   end
   E(k) = 0.5*norm(cross(u,v));   
   
end

area = sum(E);