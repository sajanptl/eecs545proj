function WriteTopics( WP , BETA , WO , FILENAME )
%% Function WriteTopics
%%
% |[ S ] = WriteTopics( WP , BETA , WO , K , E , M , FILENAME )|
% writes the |K| most likely entities per topic to a text file |FILENAME|
% with |M| columns. |E| is a threshold on the topic listings in |S|. Only
% entities that do not exceed this cumulative probability are listed.
%
%%
% |WriteTopics( WP , BETA , WO , K , E , M , FILENAME )| writes the |K|
% most likely entities per topic without producing a cell array of strings
%
%
% |WP(i,j)| contains the number of times word |i| has
% been assigned to topic |j|
%%
% where |DP(i,j)| contains the number of times a word in
% document |d| has been assigned to topic |j|
%
% |Z(k)| contains the topic assignment for
% token k.
% Example
%
%%
%     |WriteTopics( WP , BETA , WO , 10 , 1.0 , 4 ,
%     'topics50_psychreview.txt' )|
%%
% will write 10 most likely words per topic to a four column text file

wordsPerTopics = 10;
numWords = size( WP , 1 );
totalTopics = size( WP , 2 );
textFile = fopen( FILENAME , 'W' );
% sum up words occurency pertopics (1 x 10) matrix
sumWordsOccurancyPerTopic = sum( WP , 1 ) + BETA * numWords; 
probabilityTopicInAllTopics = sumWordsOccurancyPerTopic / sum( sumWordsOccurancyPerTopic );

sortProbabilityPerWordsPerTopic = zeros( wordsPerTopics , totalTopics );
Index_P_w_z = zeros( wordsPerTopics , totalTopics );

for t = 1:totalTopics
   [ temp1 , temp2 ] = sort( -WP( : , t ) );
   sortProbabilityPerWordsPerTopic( : , t )  = ( full( -temp1( 1:wordsPerTopics )) + BETA ) ./ ( repmat( sumWordsOccurancyPerTopic( t ) , wordsPerTopics , 1 ));
   Index_P_w_z( : , t )   = temp2( 1:wordsPerTopics );
end

for indexOfTopics = 1:totalTopics
    for c = indexOfTopics:totoalTopics
        nameOfTopicTitles = sprintf( '%s_%d' , 'TOPIC' , c );
        fprintf( textFile , '%25s\t%6.5f\t' , nameOfTopicTitles , probabilityTopicInAllTopics( c ) );     
    end
    for r=1:wordsPerTopics
        for c=indexOfTopics:totoalTopics
            index = Index_P_w_z( r , c );
            prob = sortProbabilityPerWordsPerTopic( r , c );
            fprintf( textFile , '%25s\t%6.5f\t' , WO{ index } , prob );           
        end
        fprintf( textFile , '\r\n' );
    end
    fprintf( textFile , '\r\n\r\n' );
end
fclose( textFile );
