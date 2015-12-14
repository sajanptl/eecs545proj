%% Example 1 of running basic topic model (LDA)
%
% This example shows how to run the LDA Gibbs sampler on a small dataset to
% extract a set of topics and shows the most likely words per topic. It
% also writes the results to a file

%%
% Choose the dataset
dataset = 1; % 1 = psych review abstracts 2 = NIPS papers

% if (dataset == 1)
%     % load the psych review data in bag of words format
%     load 'bagofwords_psychreview'; 
%     % Load the psych review vocabulary
%     load 'words_psychreview'; 
% elseif (dataset == 2)
%     % load the nips dataset
%     load 'bagofwords_nips'; 
%     % load the nips vocabulary
%     load 'words_nips'; 
% end


load 'WS.mat'
load 'DS.mat'
load 'ZWD.mat'
load 'WO.mat'

% create training set and test set (for calculating perplexity)
WSS = double(WS);
DSS = double(DS);
%WO = double(WO);
WO = cellstr(WO);
T = 5;
N = 5;
ALPHA = 5;
BETA = 0.01;
SEED = 3;

%%
% Set the number of topics

%%
% Set the hyperparameters
%BETA=0.01;
%ALPHA=50/T;

%%
% The number of iterations
%N = 300; 

%%
% The random seed
SEED = 3;

%%
% What output to show (0=no output; 1=iterations; 2=all output)
OUTPUT = 1;

% load 'WS'
% load 'WO'
% load 'DS'
% WSS = double(WS);
% DSS = double(DS);
%%
% This function might need a few minutes to finish
tic
[ WP,DP,Z ] = GibbsSamplerLDA( WSS , DSS , T , N , ALPHA , BETA , SEED , OUTPUT );
toc

%%
% Just in case, save the resulting information from this sample 
% if (dataset==1)
%     save 'ldasingle_psychreview' WP DP Z ALPHA BETA SEED N;
% end
% 
% if (dataset==2)
%     save 'ldasingle_nips' WP DP Z ALPHA BETA SEED N;
% end
%%
% Put the most 7 likely words per topic in cell structure S
%[S] = WriteTopics( WP , BETA , WO , 10 , 0.7 );

fprintf( '\n\nMost likely words in the first ten topics:\n' );

%%
% Show the most likely words in the first ten topics
% S( 1:10 )  

%%
% Write the topics to a text file
WriteTopics( WP , BETA , WO , 10 , 0.7 , 5 , 'topics11.txt' );
%WriteTopics( WP , BETA , WO , 'topics5.txt' );

fprintf( '\n\nInspect the file ''topics.txt'' for a text-based summary of the topics\n' ); 
