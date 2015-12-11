% This example code applies LDA with variational inference to AP dataset
% from Blei. Documents were pre-processed to be a single matrix.
clear all
addpath('../lib');
addpath('../data/ap/LDA_input');
load('ZWD.mat'); % load pre-processed documents
load('WO.mat'); % load the vocabulary
W=ZWD;
D=size(W,1); % number of documents
N=size(W,2); % maximum length of documents
K=3; % number of topics
V=size(MO,1); % vocabulary size

% Initialize alpha.
alpha=zeros(K,1)+0.01;

% Initialize phi.
phi=zeros(D,N,K)+1/K;

% Initialize gamma.
gamma=bsxfun(@plus,zeros(D,K),alpha')+N/K;

% Initialize eta and beta.
eta=1;
my_eta=ones(1,V);
beta=zeros(K,V);
for k=1:K
    beta(k,:)=randg(my_eta)+realmin;
    beta(k,:)=beta(k,:)/sum(beta(k,:));
end

% Initialize lambda.
lambda=ones(K,V)/V;

% Set up stopping criteria.
thresh=0.01;
iter=0;

% Apply LDA with mean-field variational inference.
[theta,z,beta,L,i]=LDA_VI(lambda,gamma,phi,W,alpha,eta,beta,thresh,iter);

% Show results.
plot(L);






