% Example using GibbsSamplingLDA for topic extraction
% EECS 545 Final Project
% Xiang Li
% 12/11/2015


% read data
original_path = pwd;
cd('../data/ap/LDA_input/')
load 'WS.mat'
load 'DS.mat'
load 'WO.mat'
load 'WS_test.mat'
load 'DS_test.mat'
cd(original_path)

% set parameters
T = 2;          % number of topics
N_round = 30;   % number of iterations
ALPHA = 5;
BETA = 0.1;
SEED = 1;       % seed for random numbers

% learning
tic

cd('../lib/')
[ WP,DP,Z,Per ] = GibbsSamplerLDA( WS , DS , T , N_round , ALPHA , BETA, SEED);
Per_test = perplexity(WS_test, DS_test, WP, DP, T, ALPHA , BETA);
Per_test
cd(original_path)

toc

% write topics
cd('../lib/')
WO = cellstr(WO);
WriteTopics(WP, WO, [original_path, '/ap_gs_10_topics.txt']);
cd(original_path)

% plot
figure
plot(Per)
xlabel('Interation')
ylabel('Perplexity')
savefig('ap_gs_10_perplexity.fig')




