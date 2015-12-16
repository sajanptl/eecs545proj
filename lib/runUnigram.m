load ../data/ap/LDA_input/W.mat
load ../data/ap/LDA_input/W_test.mat

lambda = 0.95;

[Ptrain, Ptest, PPlexTrain, PPlexTest, WplexTrain, WplexTest] = Unigram(W, W_test, lambda);

fprintf(['Total Training Perplexity = %f\n'], PPlexTrain);
fprintf(['Total Testing Perplexity = %f\n'], PPlexTest);

fprintf(['Top 10 Training Words: \n']);
[p, idx] = sort(Ptrain(1,:), 'descend');
disp(p(1:10));
disp(idx(1:10));

fprintf(['Top 10 Testing Words: \n']);
[p, idx] = sort(Ptest(1,:), 'descend');
disp(p(1:10));
disp(idx(1:10));

save unigram_test.mat lambda Ptrain Ptest PPlexTrain PPlexTest WplexTrain WplexTest
