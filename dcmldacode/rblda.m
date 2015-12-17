% ldafast.m implements a Rao-Blackwellized Gibbs sampler for the basic LDA model.
% INPUT:
% wdmat - sparse word-document count matrix (cl400)
% numtopics - desired number of topics
% samples - number of samples used to estimate the phi/theta distributions
% burnin - number of epochs to go before sampling starts
% samplewait - number of epochs between samples
% alpha - [optional] value of the alpha parameter (default 50/numtopics)

function [master,avgphi,avgtheta] = rblda(wdmat,numtopics,samples,burnin,samplewait,alpha,beta)
    tic
    if (nargin <= 5) alpha = 50/numtopics; end    
    if (nargin <= 6) beta = 0.01; end
    
    % runs through the burnin period, and then through the sampling period
    iters = burnin + samples*samplewait;

    % initialization
    [numdocs numwords] = size(wdmat);
    totalwords = full(sum(sum(wdmat)));
    
    % master lists of word identities, topic assignments, and documents
    maswords = zeros(totalwords,1); 
    mastopics = zeros(totalwords,1);
    masdocs = zeros(totalwords,1);
    
    % initializing the topics & creating the master lists
    currword = 0;
    for d=1:numdocs
        if (d <= 100) truelabel = 1;
        elseif (d <= 200) truelabel = 2;
        else truelabel = 3; end
        currdoc = wdmat(d,:);  % look at each row in the word-doc count matrix
        for w = find(currdoc)    % pull out each non-zero entry
            wc = currdoc(w);     % count for each word-doc pair
            which = currword + (1:wc);
            maswords(which) = w;
            masdocs(which) = d;        
            mastopics(which) = ceil(rand(wc,1).*numtopics);   % initial topic structure
            mastopics(which) = truelabel;
            currword = currword + wc;
        end
    end
    
    reorder = randperm(totalwords);
    maswords = maswords(reorder);
    mastopics = mastopics(reorder);
    masdocs = masdocs(reorder);
    
    masdist = zeros(numtopics,totalwords);
    for w = 1:totalwords
        masdist(mastopics(w),w) = 1;
    end
    
    % one row for each topic and one column for each document or word
    tdmat = accumarray([mastopics,masdocs],1,[numtopics,numdocs]) + alpha;  % topic-document count matrix
    twmat = accumarray([mastopics,maswords],1,[numtopics,numwords]) + beta; % topic-word count matrix
    sumctw = sum(twmat,2);                                                  % number of words in each topic    
    
    % prepare to sample phi & theta
    samplestaken = 0;
    avgtheta = zeros(numtopics,numdocs);
    avgphi = zeros(numwords,numtopics);
    
    for i=1:iters			% each epoch in the Gibbs sampler
        if mod(i,10) == 1
            toc
            sprintf('Beginning iteration %d.',i)
            tic
        end
       
        for w = 1:totalwords               
            currword = maswords(w);		 % identity of current word
            currdoc = masdocs(w);
            currdist = masdist(:,w);

            sumctw = sumctw - currdist;
            twmat(:,currword) = twmat(:,currword) - currdist;
            tdmat(:,currdoc) = tdmat(:,currdoc) - currdist; 

	        % calculate vector of topic probabilites
            topicprobs = twmat(:,currword) .* tdmat(:,currdoc) ./ sumctw;
            
            denom = topicprobs(1);
            for t = 2:numtopics
                denom = denom + topicprobs(t);  
            end
            currdist = topicprobs ./ denom;
              
            masdist(:,w) = currdist;
            sumctw = sumctw + currdist;
            twmat(:,currword) = twmat(:,currword) + currdist;
            tdmat(:,currdoc) = tdmat(:,currdoc) + currdist;                
        end
        
        % is it time to sample the distribution?
        if (i >= burnin && mod(i,samplewait) == 0)
    
            theta = tdmat;        % pmf over topics for each document
            for d=1:numdocs
                theta(:,d) = theta(:,d) ./ sum(theta(:,d));
            end

            phi = twmat';         % pmf over words for each topic
            for t=1:numtopics
                phi(:,t) = phi(:,t) ./ sum(phi(:,t));
            end            

	    % include the current estimation of phi/theta in the sampled distribution
            avgphi = (avgphi*samplestaken + phi) / (samplestaken + 1);
            avgtheta = (avgtheta*samplestaken + theta) / (samplestaken + 1); 
            samplestaken = samplestaken + 1;
        end
    end
    toc
    % invariant: sumctw == sum(twmat,2) == sum(tdmat,2)
    
    %Combining the master list of words, topics, and documents.
    master=[maswords,mastopics,masdocs];
end
            