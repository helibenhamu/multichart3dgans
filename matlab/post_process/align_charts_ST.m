function [ new_charts ] = align_charts_ST( data, triplets_table, fixed_chart)
%ALIGN_CHARTS_ST aligns the charts by scale and translation
%   Input:
%       data            - spatial_dim x spatial_dim x triplets*3
%       triplets_table  - table of triplets


 if nargin<3
     fixed_chart = 1;
 end
        
% unique cones in triplets
uCones = unique(triplets_table);

% extract cones from charts
N = size(data,1);
center  = N/2+1;
cones = [ squeeze(data(center,center,:)), squeeze(data(1,1,:)), squeeze(data(center,1,:))];

% create cone blocks
% [ xi 1 0 0 ;
%   yi 0 1 0 ;
%   zi 0 0 1 ]
CB = {};
for ii = 1:size(triplets_table,1)
    for jj = 1: size(triplets_table,2)
        CB{ii,jj} = [ cones(3*ii-2:3*ii,jj), eye(3)];
    end
end

% find all cone coincidences
triplet_pairs = [];
point_pairs = [];
for ii = 1:length(uCones)
    flags = find(triplets_table == uCones(ii));
    if length(flags)>1
        [R ,C] = ind2sub(size(triplets_table),flags);
        triplet_pairs = [triplet_pairs ; nchoosek(R,2)]; %#ok<AGROW>
        point_pairs = [point_pairs ; nchoosek(C,2)];    %#ok<AGROW>
    end
end

% construct A
A = zeros(3*size(triplet_pairs,1),4*size(triplets_table,1));
for ii = 1:size(triplet_pairs,1)
    A(3*(ii-1)+1:3*ii,4*(triplet_pairs(ii,1)-1)+1:4*triplet_pairs(ii,1)) = CB{triplet_pairs(ii,1),point_pairs(ii,1)};
    A(3*(ii-1)+1:3*ii,4*(triplet_pairs(ii,2)-1)+1:4*triplet_pairs(ii,2)) = -CB{triplet_pairs(ii,2),point_pairs(ii,2)};
end
const_chart = zeros(4,size(triplets_table,1)*4);
D = eye(4);
const_chart(:,4*(fixed_chart-1)+(1:4)) = D;

A = [ const_chart; A];
b = [1 ; zeros(size(A,1)-1,1)] ;

% solve LS
x = A\b;
S = x(1:4:end);
T = x; T(1:4:end)=[];

% apply transformation to charts
new_charts = zeros(size(data));
for ii = 1:size(triplets_table,1)
    new_charts(:,:,3*(ii-1)+1:3*ii) = data(:,:,3*(ii-1)+1:3*ii)*S(ii)+repmat(reshape(T((ii-1)*3+1:(ii-1)*3+3),1,1,3),N,N);
end

end

