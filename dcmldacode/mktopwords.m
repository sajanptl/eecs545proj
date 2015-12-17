function topwords = mktopwords(ntop,phi,wordlist)
% for each topic, find words with highest-probability

[nwords ntopics] = size(phi);
if (nwords ~= length(wordlist))
    display('bad wordlist');
end

topwords = cell(ntop,ntopics);
for t=1:ntopics
   [sorted which] = sort(-phi(:,t));
   topwords(:,t) = wordlist(which(1:ntop));
end  
