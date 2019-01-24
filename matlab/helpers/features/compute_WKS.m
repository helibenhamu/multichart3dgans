% This is the implementation of the Wave Kernel Signature described
% in the paper:
% 
%    The Wave Kernel Signature: A Quantum Mechanical Approach To Shape Analysis 
%    M. Aubry, U. Schlickewei, D. Cremers
%    In IEEE International Conference on Computer Vision (ICCV) - Workshop on 
%    Dynamic Shape Capture and Analysis (4DMOD), 2011
% 
% Please refer to the publication above if you use this software. 
% 
% This work is licensed under a Creative Commons
% Attribution-NonCommercial 3.0 Unported License  
% ( http://creativecommons.org/licenses/by-nc/3.0/ )
% 
% The WKS is patented and violation to the license agreement would
% imply a patent infringement.
%
% THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY EXPRESSED OR IMPLIED WARRANTIES
% OF ANY KIND, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THIS SOFTWARE OR THE USE OR OTHER DEALINGS IN
% THIS SOFTWARE.


function [WKS,E,PHI,L] = compute_WKS(vertices,faces,N)

% function [WKS,E,PHI] = compute_WKS(vertices,faces) compute
%   the Wave Kernel Signature of triangle mesh given by
%   [vertices,faces]
%   
%   INPUT:
%   vertices is (number of vertices) x 3 matrix
%   faces is a (number of faces) x 3 matrix
%   
%   OUTPUT:
%   WKS is the (number of vertices) x 100 WKS matrix 
%   E is the vector of LB eigenvalues (by default of size 300 x 1)
%   PHI is the (number of vertices x 300) matrix of LB eigenfunctions 
%   L is the cotan Laplace-Beltrami matrix
%
%   The main parameter to adjust depending on your task is wks_variance


%% parameters

n_eigenvalues=300; % number of eigenvalues used for computations
% depending on the application, you can use less than 300
if nargin<3
N = 100; % number of evaluations of WKS
end
wks_variance = 6; % variance of the WKS gaussian (wih respect to the 
% difference of the two first eigenvalues). For easy or precision tasks 
% (eg. matching with only isometric deformations) you can take it smaller



%% basic quantities

num_vertices = size(vertices,1);
num_faces = size(faces,1);



%% detect boundary vertices

% Calculate the (directed) adjacency matrix. adjacency_matrix(m,n) = 1 if the oriented
% boundary of a triangle contains the directed edge from vertex m to vertex
% n, and 0 otherwise. This matrix is not quite symmetric due to boundary edges.
adjacency_matrix = sparse([faces(:,1); faces(:,2); faces(:,3)], ...
                         [faces(:,2); faces(:,3); faces(:,1)], ...
    	                 ones(3 * num_faces, 1), ...
                         num_vertices, num_vertices, 3 * num_faces);
if any(any(adjacency_matrix > 1))
    error('Triangles must be oriented consistently.')
end
clear adjacency_matrix;


%% compute cotan LB matrix

fprintf('Computing Laplace-Beltrami operator...');

% first compute inner face angles and squared edge lengthes
pp = zeros(num_faces,3);
qq = zeros(num_faces,3);
angles = 0*faces;
squared_edge_length = 0*faces;

for i=1:3
    i1 = mod(i-1,3)+1;
    i2 = mod(i  ,3)+1;
    i3 = mod(i+1,3)+1;
    pp = vertices(faces(:,i2),:) - vertices(faces(:,i1),:);
    qq = vertices(faces(:,i3),:) - vertices(faces(:,i1),:);
    % normalize the vectors
    pp = pp ./ repmat( max(sqrt(sum(pp.^2,2)),eps), [1 3] );
    qq = qq ./ repmat( max(sqrt(sum(qq.^2,2)),eps), [1 3] );
    % compute angles
    angles(:,i1) = acos(sum(pp.*qq,2));
    squared_edge_length(:,i1) = sum((vertices(faces(:,i2)) - vertices(faces(:,i3))).^2,2);
end
clear pp qq;

%then compute L
L = sparse(num_vertices,num_vertices);
for i=1:3
    i1 = mod(i-1,3)+1;
    i2 = mod(i  ,3)+1;
    i3 = mod(i+1,3)+1;
    L = L + sparse(faces(:,i1),faces(:,i2),-cot(angles(:,i3)),...
        num_vertices,num_vertices,num_faces);       
end

L = 1/2 * (L + L');
L = sparse(1:num_vertices,1:num_vertices,-sum(L,2),num_vertices,num_vertices,...
    num_vertices) + L;

fprintf('done. \n');
%% compute Voronoi areas for vertices (following the algorithm decribed in 
%   Discrete Differential-Geometry Operators for Triangulated 2-Manifolds, 
%   Mark Meyer, Mathieu Desbrun, Peter Schroeder and Alan H. Barr, VisMath 2002. 

fprintf('Computing area of vertex Voronoi cells...');


% compute area of triangles
faces_area = zeros(num_faces,1);
for i = 1:3
    faces_area = faces_area +1/4 * (squared_edge_length(:,i).*1./tan(angles(:,i)));
end

% compute area of Voronoi cells
A = zeros(num_vertices,1);
for j = 1:num_vertices
    for i = 1:3
        i1 = mod(i-1,3)+1;
        i2 = mod(i,3)+1;
        i3 = mod(i+1,3)+1;
        ind_j = find(faces(:,i1) == j);
        for l = 1:size(ind_j,1)
            face_index = ind_j(l);
            if (max(angles(face_index,:)) <= pi/2)
                A(j) = A(j) + 1/8 * (1/tan(angles(face_index,i2))* ...
                    squared_edge_length(face_index,i2) + ... 
                    1/tan(angles(face_index,i3))*...
                    squared_edge_length(face_index,i3));
            elseif angles(face_index,i1) > pi/2
                A(j) = A(j) + faces_area(face_index)/2;
            else
                A(j) = A(j) + faces_area(face_index)/4;
            end
        end        
    end
end

A = max(A,1e-8);
area = sum(A);
A = A/area;
Am = sparse([1:length(A)], [1:length(A)], A);

clear angles faces_area;
fprintf('done. \n');

%% compute LB eigenvalues and eigenfunctions 

fprintf('Computing Laplace-Beltrami eigenfunctions...');

% solve generalized eigenvalue problem
options.disp = 0;
[PHI,E] = eigs(L,Am,n_eigenvalues,-1e-5,options);
E = diag(E);
E=abs(real(E));
[E,idx] = sort(E);
PHI = PHI(:,idx);
PHI=real(PHI);

fprintf('done. \n');

%% compute WKS 

fprintf('Computing WKS...');

WKS=zeros(num_vertices,N);

log_E=log(max(abs(E),1e-6))';
e=linspace(log_E(2),(max(log_E))/1.02,N);  
sigma=(e(2)-e(1))*wks_variance;

C = zeros(1,N); %weights used for the normalization of f_E

for i = 1:N
    WKS(:,i) = sum(PHI.^2.*...
        repmat( exp((-(e(i) - log_E).^2) ./ (2*sigma.^2)),num_vertices,1),2);
    C(i) = sum(exp((-(e(i)-log_E).^2)/(2*sigma.^2)));
end

% normalize WKS
WKS(:,:) = WKS(:,:)./repmat(C,num_vertices,1);

fprintf('done. \n');
