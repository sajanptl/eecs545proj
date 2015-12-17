% Example using GibbsSamplingLDA for topic extraction
% EECS 545 Final Project
% Xiang Li
% 12/11/2015

original_path = pwd;

% read data
cd('../data/ap/LDA_input/')
load 'WS.mat'
load 'DS.mat'
load 'WO.mat'

load 'ZWD_test.mat'


cd(original_path)

T = 5;          % number of topics
N_round = 30;   % number of iterations
ALPHA = 5;
BETA = 0.1;
SEED = 3;       % seed 

cd('../lib/')
tic
[ WP,DP,Z,Per ] = GibbsSamplerLDA( WS , DS , WO, WS_test, DS_test, ...
    T , N_round , ALPHA , BETA);
toc
cd(original_path)

% cd('../lib/')
% WO = cellstr(WO);
% WriteTopics(WP, WO, '../examples/topics.txt');
% cd(original_path)


