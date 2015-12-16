% Example using GibbsSamplingLDA for topic extraction
% EECS 545 Final Project
% Xiang Li
% 12/11/2015

original_path = pwd;

% read data
cd('../data/ap/LDA_input/')
load 'WS.mat'
load 'DS.mat'
load 'ZWD.mat'
load 'WO.mat'
load 'WS_test.mat'
load 'DS_test.mat'
load 'ZWD_test.mat'


cd(original_path)

T = 5;
N_round = 30;
ALPHA = 5;
BETA = 0.01;
SEED = 3;

cd('../lib/')
tic
[ WP,DP,Z,Per ] = GibbsSamplerLDA( WS , DS , ZWD, WO, WS_test, DS_test, ...
    ZWD_test, T , N_round , ALPHA , BETA);
toc
cd(original_path)

cd('../lib/')
WO = cellstr(WO);
WriteTopics(WP, WO, '../examples/topics.txt');
cd(original_path)


