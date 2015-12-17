%% Runs the Unigram model on the AP Corpus dataset
%  Sajan Patel (sajanptl)
%  December 2015


%% Load data
load ../data/ap/LDA_input/W.mat
load ../data/ap/LDA_input/W_test.mat
load ../data/ap/LDA_input/WO.mat

%% smoothing parameter
lambda = 0.95;

%% run model
[Ptrain, Ptest, PPlexTrain, PPlexTest, WplexTrain, WplexTest] = Unigram(W, W_test, lambda);

%% print results
fprintf(['-----------\n']);
fprintf(['Total Training Perplexity = %f\n'], PPlexTrain);
fprintf(['Total Testing Perplexity = %f\n'], PPlexTest);
fprintf(['-----------\n']);

fprintf(['Top 10 Training Words: \n']);
[p, idx] = sort(Ptrain, 'descend');
disp(p(1:10));
disp(idx(1:10));
disp(WO(idx(1:10), :));
fprintf(['-----------\n']);

for i = 1:3
    fprintf(['Top 10 Testing Words for Document %d: \n'], i*10);
    [p, idx] = sort(Ptest(i*10,:), 'descend');
    disp(p(1:10));
    disp(idx(1:10));
    disp(WO(idx(1:10), :));
    fprintf(['-----------\n']);
end

%% save results
save unigram_test.mat lambda Ptrain Ptest PPlexTrain PPlexTest WplexTrain WplexTest
