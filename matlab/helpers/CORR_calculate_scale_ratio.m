function [area, E] = CORR_calculate_scale_ratio(F,V,U)

% Calculate the area of the surface

E=zeros(size(F,1),1);

for k=1:size(F,1)
   f = F(k,:);
   
   %source triangle
   v1 = V(f(1),:);
   v2 = V(f(2),:);
   v3 = V(f(3),:);
   
   %target triangle
   u1 = U(f(1),:);
   u2 = U(f(2),:);
   u3 = U(f(3),:);
   
   %source triangle
   v12 = v2-v1;
   v13 = v3-v1;
   
   %target triangle
   u12 = u2-u1;
   u13 = u3-u1;
   
   Au = [u12' u13'];
   
   Av = [v12' v13'];
   
   %the desired differential
%    A = pinv(Av)*Au;
   A = Au*pinv(Av);
   
   
   if(sum(sum(isnan(A))) || sum(sum(isinf(A))))
      E(k) =  Inf;
   else
      [~,tS,~] = svd(A);
      E(k) = tS(1,1)*tS(2,2);%sqrt(cond(A'*A));
   end
   
end

area = sum(E);