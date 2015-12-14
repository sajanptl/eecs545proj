% Example using GibbsSamplingLDA for topic extraction
% EECS 545 Final Project
% Xiang Li
% 12/11/2015

original_path = pwd

% read data
cd('../data/ap/LDA_input/')
load 'WS.mat'
load 'DS.mat'
load 'ZWD.mat'
load 'WO.mat'

% create training set and test set (for calculating perplexity)


cd(original_path)

T = 10;
N_round = 1;
ALPHA = 1;
BETA = 1;
SEED = 3;

cd('../lib/')
tic
[ WP,DP,Z ] = GibbsSamplerLDA( WS , DS , ZWD, T , N_round , ALPHA , BETA , SEED)
toc

WO = cellstr(WO)

