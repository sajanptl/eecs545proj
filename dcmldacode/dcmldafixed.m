% dcmldafixed.m implements a Gibbs sampler for the basic LDA model
% informal EM to learn alpha and beta parameter vectors
% INPUT:
% wdmat - sparse word-document count matrix 
% numtopics - desired number of topics
% samples - number of M-steps of EM
% burnin - number of epochs before sampling starts
% samplewait - number of epochs between M-steps
% alpha - [optional] initial value (default 50/numtopics)
% beta - [optional] initial value (default 0.01)

function [master,alphas,betas] = dcmldafixed(wdmat,numtopics,samples,burnin,samplewait,alpha,beta)
    tic
    if (nargin <= 5) alpha = 50/numtopics; end    
    if (nargin <= 6) beta = 0.01; end
    
    % initialization
    [numdocs numwords] = size(wdmat);       % numwords is vocab size (types)
    totalwords = full(sum(sum(wdmat)));     % totalwords is token count
    
    % expand beta to be a matrix (allowing for asymmetry)
    if size(beta) == 1
        beta = beta*ones(numtopics,numwords);
    end
    
    % runs through the burnin period, and then through the EM period
    iters = burnin + samples*samplewait;
    
    % test of alphas coming out
    alphas(1) = alpha;
    betas{1} = beta;

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
        currdoc = wdmat(d,:);    % look at each row in the word-doc count matrix
        for w = find(currdoc)    % pull out each non-zero entry
            wc = currdoc(w);     % count for each word-doc pair
            which = currword + (1:wc);
            maswords(which) = w;
            mastopics(which) = ceil(rand(wc,1).*numtopics);   % initial topic structure
            %mastopics(which) = truelabel;
            masdocs(which) = d;
            currword = currword + wc;
        end
    end
    
    % make sure the documents are Gibbs sampled in random order
    reorder = randperm(totalwords);
    maswords = maswords(reorder);
    mastopics = mastopics(reorder);
    masdocs = masdocs(reorder);
    
    % one row for each topic and one column for each document or word
    tdmat = full(sparse(mastopics,masdocs,1,numtopics,numdocs)) + alpha;  % topic-document count matrix
    twmat = full(sparse(mastopics,maswords,1,numtopics,numwords)) + beta; % topic-word count matrix

    for i = 1:iters			      % for each epoch of the Gibbs sampler
        sumctw = sum(twmat,2);    % number of words in each topic
        if mod(i,10) == 1
            sprintf('Used time %g, beginning iteration %d.',toc,i)
            tic
        end
        for w = 1:totalwords               
            currword = maswords(w);		 % identity of current word
            currtopic = mastopics(w);    
            currdoc = masdocs(w);

	    % removing the current word from the dataset (z_{-i})
            sumctw(currtopic) = sumctw(currtopic) - 1;
            twmat(currtopic,currword) = twmat(currtopic,currword) - 1;
            tdmat(currtopic,currdoc) = tdmat(currtopic,currdoc) - 1;
            
	        % calculate vector of topic probabilites
            topicprobs = twmat(:,currword) .* tdmat(:,currdoc) ./ sumctw;
            denom = topicprobs(1);
            for t = 2:numtopics
                denom = denom + topicprobs(t);  
            end
            
            % choose new topic for the word
            denom = rand(1)*denom; 
            bound = topicprobs(1);
            for currtopic = 1:numtopics
                if (denom <= bound) break; end
                bound = bound + topicprobs(currtopic+1);   % error message if no topic assigned
            end
            
	    % re-adding the word to the dataset
            mastopics(w) = currtopic;
            sumctw(currtopic) = sumctw(currtopic) + 1;
            twmat(currtopic,currword) = twmat(currtopic,currword) + 1;
            tdmat(currtopic,currdoc) = tdmat(currtopic,currdoc) + 1;
        end % for each word

        % is it time to do an M-step?
        if (i > burnin && mod(i-burnin,samplewait) == 1)
            theta = full(sparse(mastopics,masdocs,1,numtopics,numdocs)) + 1e-6;
            for d = 1:numdocs
                theta(:,d) = theta(:,d) ./ sum(theta(:,d));
            end
            adisc = checkgrad('dcmldalpha', alpha, 1e-10, theta);
            sprintf('Max relative discrepancy %g in alpha gradient.',adisc)
            newalpha = minimize(1,'dcmldalpha',10,theta); 
               
            newbeta = beta;
            for k = 1:numtopics
                which = find(mastopics == k);
                phi = full(sparse(maswords(which),masdocs(which),1,numwords,numdocs)) + 1e-6; % word-doc counts
                for d = 1:numdocs
                    phi(:,d) = phi(:,d) ./ sum(phi(:,d));
                end

                b = ones(numwords,1);
%                bdisc = checkgrad('dcmldbeta', b, 1e-5, phi);
%                sprintf('Max relative discrepancy %g in beta gradient.',bdisc)
                newb = minimize(b,'dcmldbeta',10,phi); 
                newbeta(k,:) = newb';
            end
                
            tdmat = tdmat - alpha + newalpha;
            twmat = twmat - beta + newbeta;
            alpha = newalpha;
            beta = newbeta;

            alphas(end+1) = alpha;
            betas{end+1} = beta;
        end % of M-step
    end % for each epoch
    toc
    % invariant: sumctw == sum(twmat,2) == sum(tdmat,2)
    
    %Combining the master list of words, topics, and documents.
    master = [maswords,mastopics,masdocs];
end
            