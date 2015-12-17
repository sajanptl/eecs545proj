function per = perplexity( WS, DS, WP, DP, T, ALPHA , BETA)

N = length(WS);
W = size(WP, 1);
per = 0;
for i = 1:N       % for every word
    w = WS(i);    % token represented by word i
    d = DS(i);    % document d is the document containing word i
    
    % vectorize
    nj = sum(WP);
    nwj = WP(w, :);
    dp = DP(d,:);
    
    p = (dp + ALPHA) / (sum(dp) + T*ALPHA) .* (nwj + BETA) ./ (nj + W*BETA);

    per = per + log(sum(p));
end
per = exp(-per/N);

end