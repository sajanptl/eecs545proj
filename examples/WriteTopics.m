function WriteTopics( WP , BETA , WO , FILENAME )

wordsPerTopics = 10;
numWords = size( WP , 1 );
totalTopics = size( WP , 2 );
textFile = fopen( FILENAME , 'W' );

sumWP = sum( WP , 1 ) + BETA*numWords;
probtopic = sumWP / sum( sumWP );

sortProbabilityPerWordsPerTopics = zeros( wordsPerTopics , totalTopics );
indexPerWordPerTopics = zeros( wordsPerTopics , totalTopics );

for t=1:totalTopics
   [ temp1 , temp2 ] = sort( -WP( : , t ) );
   sortProbabilityPerWordsPerTopics( : , t )  = ( full( -temp1( 1:wordsPerTopics )) + BETA ) ./ ( repmat( sumWP( t ) , wordsPerTopics , 1 ));
   indexPerWordPerTopics( : , t )   = temp2( 1:wordsPerTopics );
end

for indexOfTopics=1:totalTopics
    topicTitle = sprintf( '%s_%d' , 'TOPIC' , indexOfTopics );
    fprintf( textFile , '%25s\t%6.5f\t' , topicTitle , probtopic( indexOfTopics ) );
end

for r=1:wordsPerTopics
    for indexOfTopics = 1:totalTopics
        index = indexPerWordPerTopics( r , indexOfTopics );
        prob = sortProbabilityPerWordsPerTopics( r , indexOfTopics );
        fprintf( textFile , '%25s\t%6.5f\t' , WO{ index } , prob );
    end
    fprintf( textFile , '\r\n' );
end

fclose( textFile );
