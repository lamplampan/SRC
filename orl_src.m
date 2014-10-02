%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% SRC, Sparse Representation-based Classification, 
% an algorithm in Pattern Recognition area
% 
% Author:
%   Written by Denglong Pan,
%   pandenglong@gmail.com
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
t0=clock;
Image_row_NUM=112;Image_column_NUM=92; 
NN=Image_row_NUM*Image_column_NUM;%

Class_Train_NUM=5;
Class_Sample_NUM=10; % 
Class_Test_NUM=Class_Sample_NUM-Class_Train_NUM;

Class_NUM=40;
Train_NUM=Class_NUM*Class_Train_NUM; % 
Test_NUM=Class_NUM*(Class_Sample_NUM-Class_Train_NUM); % 

Eigen_NUM=80;%
Disc_NUM=Eigen_NUM;

% for Kernel Fisherface and CKFD regular and irregular
Dim_Begin=10; Dim_End=Disc_NUM; Dim_Interval=1; 

% is same as (Dim_End-Dim_Begin+Dim_Interval)/Dim_Interval
Dim_Total_NUM=(Dim_End-Dim_Begin)/Dim_Interval+1; 

%read training samples
Train_DAT=zeros(NN,Train_NUM);
s=1;

for r=1:Class_NUM
   for t=1:Class_Train_NUM
   
     % Need to configure the location of orl path here .
     string=['E:\ORL_face\orlnumtotal\s' int2str(r) '_' int2str(t)];
     A=imread(string,'bmp');
     B=im2double(A);
     
     Train_DAT(:,s)=B(:);
     s=s+1;
   end
end

%read testing samples
Test_DAT=zeros(NN,Test_NUM);
s=1;

for r=1:Class_NUM
   for t=Class_Train_NUM+1:Class_Sample_NUM
   
     % Need to configure the location of orl path here .
     string=['E:\ORL_face\orlnumtotal\s' int2str(r) '_' int2str(t)];
     A=imread(string,'bmp');
     B=im2double(A);
     
     Test_DAT(:,s)=B(:);
     s=s+1;
   end
end

% to center the each training sample and testing sample
% !!! Note that: Centralization have great effection when Cos distance is
% used, but it has no impact when L2 or L1 distance is used
% remove out the mean value.
Mean_Image=mean(Train_DAT,2);  

Train_DAT=Train_DAT-Mean_Image*ones(1,Train_NUM);
Test_DAT=Test_DAT-Mean_Image*ones(1,Test_NUM);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find the max Disc_NUM principal component and corresponding eigenvector
% Disc_NUM is the number of principal component,
% Projection_Matrix is the number of Projection Direction
[Projection_Matrix,disc_value]=Eigenface_f(Train_DAT, Disc_NUM);   

% LLE_DP Transform:
% Project the training simples in principal component
Train_SET=Projection_Matrix'*Train_DAT; % size of (Disc_NUM,Train_NUM); % PCA-based 
Test_SET=Projection_Matrix'*Test_DAT;   % size of (Disc_NUM,Test_NUM); % PCA-based

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Train_SET=reshape(Train_SET,[Disc_NUM,Class_Train_NUM,Class_NUM]);
%Test_SET=reshape(Test_SET,[Disc_NUM,Class_Test_NUM,Class_NUM]);

A=Train_SET;

% normalization of training set
for count=1:1:200
    A(:,count)=A(:,count)/norm(A(:,count),2);
end

% Record vectors, classes the simples belong to.
classification=zeros(200,1);

% The wrong classified classes numbers.
miss=0;

% do classification work
for t=1:1:200
    y=Test_SET(:,t);
    
    % The pseudo-inverse of A
    x0=inv(A'*A)*A'*y;
    
    %solve the L1_minimization through y-Ax<=ε
    xp=l1eq_pd(x0,A,[],y,1e-3);
    % xp
    % L=norm(xp,1);% 1 Norm of xp .
    
    % 2 Norm , Record the k residual
    test=zeros(40,1);
    
    for k=1:1:40
        % Rx is the Solution of the vector, which is projected by eigenfunction.
        % The coefficient of relevant classes is non-zero. Others will be zero.
        Rx=zeros(200,1);
        
        % Copy the non-zero value in k class from xp to Rx. 
        % The other positions in Rx is zero because it is not copied. 
        % This equal to mapped by eigenfunction δ.
        Rx(5*(k-1)+1:5*k)=xp(5*(k-1)+1:5*k);
        
        % res=residual(y)
        res=y-A*Rx;
        
        % The 2 norm of residual in k class.
        test(k)=norm(res,2);
    end
    [value,order]=min(test);
    classification(t)=order;
   if t<(order-1)*5+1|t>order*5
       miss=miss+1;
   end
end
classification=reshape(classification,5,40);
classification
miss
Recognition_rate=(Test_NUM-miss)/Test_NUM;
Recognition_rate

etime(clock,t0)

