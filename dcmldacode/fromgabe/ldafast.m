% lda.m implements a Gibbs sampler for the basic LDA model.
% Reads in a (sparse) count matrix of words and documents
% INPUT:
% wdmat - word-document count matrix (cl400)
% numtopics - desired number of topics
% samples - number of samples used to estimate the phi/theta distributions
% burnin - number of epochs to go before sampling starts
% samplewait - number of epochs between samples
% alpha - [optional] value of the alpha parameter (default 50/numtopics)

function [master,avgphi,avgtheta] = ldafast(wdmat,numtopics,samples,burnin,samplewait,alpha)
    %Constants
    beta = 0.01;
    if (nargin == 5)
        alpha = 50/numtopics;
    end
    
    %Runs through the burnin period, and then through the sampling period
    iters = burnin + samples*samplewait;

    %Initialization
    numdocs=size(wdmat,1);
    numwords=size(wdmat,2);
    totalwords = sum(sum(wdmat));
    if issparse(wdmat)
        totalwords = totalwords(1,1);   %Converts from 1x1 sparse array to double
    end
    
    %Master lists of word identities, topics, and documents
    maswords=zeros(totalwords,1); %word,topic,document for each word
    mastopics=zeros(totalwords,1);
    masdocs=zeros(totalwords,1);
    
    %Initializing the topics & creating the master lists
    currword=0;
    for d=1:numdocs
        currdoc=wdmat(d,:);  %Looking at each row in the word-doc count matrix
        for w=find(currdoc)  %Pulling out each non-zero entry
            wc=currdoc(w);   %Count for each word-doc pair
            maswords(currword+(1:wc)) = w;
            mastopics(currword+(1:wc)) = ceil(rand(wc,1)*numtopics); %Initializes the topic structure
            masdocs(currword+(1:wc)) = d;
            currword = currword+wc;
        end
    end
    
    tdmat = sparse(mastopics,masdocs,1,numtopics,numdocs);     %Topic-document count matrix
    twmat = sparse(mastopics,maswords,1,numtopics,numwords);   %Topic-word count matrix
    
    %Preparing it to sample phi & theta
    samplestaken=0;
    avgphi = zeros(numtopics,numwords);
    avgtheta = zeros(numdocs,numtopics);
    
    tic
    
    for i=1:iters			%Each epoch in the Gibbs sampler
        if mod(i,10)==0
            toc
            sprintf('Beginning iteration %d.',i)
            tic
        end
        sumctw = sum(twmat,2) + totalwords*beta;   %Number of words in each topic
        sumctd = sum(tdmat,1) + numtopics*alpha;   %Number of words in each document
        for w=1:totalwords
            currword = maswords(w);		%For each word in the corpus,
            currtopic = mastopics(w);           %getting its word number, topic, and doc
            d = masdocs(w);

            twvect = twmat(:,currword) + beta;  %Vector of topic counts for given word
            tdvect = tdmat(:,d) + alpha;	%Vector of topic counts for given document

            topicmatchvect = zeros(numtopics,1);  %Vector to account for the removal of the topic z_i
            topicmatchvect(currtopic) = -1;

	    %Calculating the vector of topic probabilites
            topicprobs = ((twvect-topicmatchvect).*(tdvect-topicmatchvect))./((sumctw-topicmatchvect).*(sumctd(d)-topicmatchvect));  %From steyvers-griffiths (3)

	    %Choosing a new topic for the word
            cumtopicprobs = cumsum(topicprobs);
            cumtopicprobs = cumtopicprobs/cumtopicprobs(numtopics);
            r = rand(1);
            mastopics(w) = find(r < cumtopicprobs,1);
        end

	%Updating the topic counts at the end of each epoch
        tdmat = sparse(mastopics,masdocs,1,numtopics,numdocs);
        twmat = sparse(mastopics,maswords,1,numtopics,numwords);

        %Is it time to sample the distribution?
        if (i>=burnin && mod(i,samplewait)==0)
    
            sumcwt = sum(twmat,2)+numwords*beta;
            sumcdt = sum(tdmat,1)+numtopics*alpha;
            
            theta = tdmat' + alpha;
            phi = twmat + beta;
            for d=1:numdocs
                theta(d,:) = theta(d,:)./(sumcdt(d) + numtopics*alpha); %From S-G (4)
            end
            for t=1:numtopics
                phi(t,:) = phi(t,:)./(sumcwt(t) + numwords*beta); %From S-G (4)
            end            

	    %Note: those calculations could stand to be optimized, but they aren't big time sinks right now
	    
	    %Including the current estimation of phi/theta in the sampled distribution
            avgphi = (avgphi*samplestaken + phi);
            avgphi = avgphi / (samplestaken+1);
            avgtheta = (avgtheta*samplestaken + theta);
            avgtheta = avgtheta / (samplestaken+1); 
            samplestaken = samplestaken + 1;
        end
    end
    
    %Combining the master list of words, topics, and documents.
    master=[maswords,mastopics,masdocs];
end
            