function [uxy,tn,al2,al3]=mytri2grid(V,t,u,tn,al2,al3)
%mytri2grid Interpolate from triangular mesh to rectangular grid.
% Based on the matlab function tri2grid 
%
%       UXY=TRI2GRID(V,T,U,X,Y) computes the function values UXY
%       over the grid defined by the vectors X and Y, from the
%       function U with values on the triangular mesh defined by V and T.
%       Values are computed using linear interpolation in
%       the triangle containing the grid point. Vectors X and Y
%       must be increasing.
%
%       [UXY,TN,A2,A3]=TRI2GRID(V,T,U,X,Y) lists
%       the index TN of the triangle containing each grid point,
%       and interpolation coefficients A2 and A3.
%
%       UXY=TRI2GRID(V,T,U,TN,A2,A3), with TN, A2, and A3 computed
%       in an earlier call to TRI2GRID, interpolates using
%       the same grid as in the earlier call. This variant is,
%       however, much faster if several functions have to be
%       interpolated using the same grid.
%
%       For grid points outside of the triangular mesh, NaN is
%       returned in UXY, TN, A2, and A3.
%
%       See INITMESH, REFINEMESH, ASSEMPDE

%       A. Nordmark 6-07-94, AN 8-01-94.
%       Copyright 1994-2003 The MathWorks, Inc.

small=10000*eps;

if nargin<5
    error('not enough input arguments')
elseif nargin==5,
    search=1;
    x=tn;
    y=al2;
else
    search=0;
    [ny,nx]=size(tn);
end

if search==1

  nt=size(t,2);
  nx=length(x);
  ny=length(y);

  tt=zeros(3,nt);
  tt(:)=V(1,t(1:3,:));
  xmin=min(tt);
  xmax=max(tt);
  tt(:)=V(2,t(1:3,:));
  ymin=min(tt);
  ymax=max(tt);
  
  [xmin,ix]=sort(xmin);
  j=1;
  for i=1:nt,
    if j<=nx,
      while ((x(j)<xmin(i)) && (abs(x(j)-xmin(i))>10*eps)),
        j=j+1;
        if j>nx,
          break
        end
      end
    end
    xmin(i)=j;
  end
  xmin(ix)=xmin;

  [xmax,ix]=sort(xmax);
  j=nx;
  for i=nt:-1:1,
    if j>=1,
      while ((x(j)>xmax(i)) && (abs(x(j)-xmax(i))>10*eps)),
        j=j-1;
        if j<1,
          break
        end
      end
    end
    xmax(i)=j;
  end
  xmax(ix)=xmax;

  [ymin,ix]=sort(ymin);
  j=1;
  for i=1:nt,
    if j<=ny,
      while ((y(j)<ymin(i)) && (abs(y(j)-ymin(i))>10*eps)),
        j=j+1;
        if j>ny,
          break
        end
      end
    end
    ymin(i)=j;
  end
  ymin(ix)=ymin;

  [ymax,ix]=sort(ymax);
  j=ny;
  for i=nt:-1:1,
    if j>=1,
      while ((y(j)>ymax(i)) && (abs(y(j)-ymax(i))>10*eps)),
        j=j-1;
        if j<1,
          break
        end
      end
    end
    ymax(i)=j;
  end
  ymax(ix)=ymax;

  tn=NaN*ones(ny,nx);
  al2=tn;
  al3=tn;
  for i=1:nt,
    if xmin(i)<=xmax(i) && ymin(i)<=ymax(i),
      a2=V(1:2,t(2,i))-V(1:2,t(1,i));
      a3=V(1:2,t(3,i))-V(1:2,t(1,i));
      a = inv([a2,a3]);
      b2 = a(1,:);
      b3 = a(2,:);
      for j=xmin(i):xmax(i),
        for k=ymin(i):ymax(i),
          if isnan(tn(k,j)),
            r1p=[x(j);y(k)]-V(1:2,t(1,i));
            d2=b2*r1p;
            if d2>=-small && d2<=1+small,
              d3=b3*r1p;
              if d3>=-small && d2+d3<=1+small,
                tn(k,j)=i;
                al2(k,j)=d2;
                al3(k,j)=d3;
                continue;
              end
            end
          end
        end
      end
    end
  end

end                                     %search

i=find(~isnan(tn));
uxy=NaN*ones(size(tn));
if ~isempty(i)
  if ny==1,
    uxy(i)=(1-al2(i)-al3(i))'.*u(t(1,tn(i)))+ ...
        al2(i)'.*u(t(2,tn(i)))+al3(i)'.*u(t(3,tn(i)));
  else
    uxy(i)=(1-al2(i)-al3(i)).*u(t(1,tn(i)))+ ...
        al2(i).*u(t(2,tn(i)))+al3(i).*u(t(3,tn(i)));
  end
end

