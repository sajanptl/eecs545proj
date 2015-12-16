function [N, W, D, ND, NWD] = prepare_data(WS, ZWD, WO)

N = length(WS);     % N - total number of words
W = length(WO);     % W - number of distinct words in volcabulary
D = size(ZWD, 1);    % D - total number of documents

% number of words (tokens) in each document
ND = sum(ZWD ~= 0, 2);   
% times of word w occuring in document d (except current word)
NWD = zeros(W, D);  
for w = 1:W
    for d = 1:D
        NWD(w, d) = sum(ZWD(d,:) == w);
    end
end

end