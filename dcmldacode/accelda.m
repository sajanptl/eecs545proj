% accelda.m implements an APPROXIMATE RAO-BLACKWELLIZED Gibbs sampler for the basic LDA model.
% INPUT:
% wdmat - sparse word-document count matrix (cl400)
% numtopics - desired number of topics
% samples - number of samples used to estimate the phi/theta distributions
% burnin - number of epochs to go before sampling starts
% samplewait - number of epochs between samples
% alpha - [optional] value of the alpha parameter (default 50/numtopics)

function [master,avgphi,avgtheta] = accelda(wdmat,numtopics,samples,burnin,samplewait,alpha,beta)
    tic
    if (nargin <= 5) alpha = 50/numtopics; end;  
    if (nargin <= 6) beta = 0.01; end;
    
    % runs through the burnin period, and then through the sampling period
    iters = burnin + samples*samplewait;

    % initialization
    [numdocs numwords] = size(wdmat);
    
    % master lists of word identities, topic assignments, and documents
    [masdocs maswords mascount] = find(wdmat);
    entries = length(masdocs);
    masdist = rand(numtopics,entries);
    
    reorder = randperm(entries);
    masdocs = full(masdocs(reorder));
    maswords = full(maswords(reorder));
    mascount = full(mascount(reorder));
    masdist = masdist(:,reorder);
    
    % prepare to sample phi & theta
    samplestaken = 0;
    avgtheta = zeros(numtopics,numdocs);
    avgphi = zeros(numwords,numtopics);
    
    [top wh] = find(masdist);
    wwh = [top maswords(wh)];
    dwh = [top masdocs(wh)];
    
    for i=1:iters			% each epoch in the Gibbs sampler
        if mod(i,100) == 1
            toc
            sprintf('Beginning iteration %d.',i)
            tic
        end
        
        lengths = sum(masdist,1) .* mascount';
        divs = ones(numtopics,1) * lengths;
        masdist = masdist ./ divs;
        
        twmat = accumarray(wwh,masdist(:),[numtopics,numwords]) + beta;
        tdmat = accumarray(dwh,masdist(:),[numtopics,numdocs]) + alpha;
        sumtw = sum(twmat,2);
        
        tdmat = tdmat ./ (sumtw * ones(1,numdocs));
        masdist = twmat(:,maswords) .* tdmat(:,masdocs);
        
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
    % invariant: sumtw == sum(twmat,2)
    
    %Combining the master list of words, topics, and documents.
    master=[maswords masdocs masdist'];
end
            