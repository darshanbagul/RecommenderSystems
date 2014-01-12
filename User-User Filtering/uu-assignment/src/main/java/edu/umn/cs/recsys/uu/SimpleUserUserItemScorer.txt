package edu.umn.cs.recsys.uu;

import it.unimi.dsi.fastutil.longs.LongArrayList;
import it.unimi.dsi.fastutil.longs.LongSet;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.data.dao.ItemEventDAO;
import org.grouplens.lenskit.data.dao.UserEventDAO;
import org.grouplens.lenskit.data.event.Rating;
import org.grouplens.lenskit.data.history.History;
import org.grouplens.lenskit.data.history.RatingVectorUserHistorySummarizer;
import org.grouplens.lenskit.data.history.UserHistory;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;
import org.grouplens.lenskit.vectors.similarity.CosineVectorSimilarity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.inject.Inject;

/**
 * User-user item scorer.
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class SimpleUserUserItemScorer extends AbstractItemScorer {
    private static final Logger logger = LoggerFactory.getLogger(SimpleUserUserItemScorer.class);

    private final UserEventDAO userDao;
    private final ItemEventDAO itemDao;

    @Inject
    public SimpleUserUserItemScorer(UserEventDAO udao, ItemEventDAO idao) {
        userDao = udao;
        itemDao = idao;
    }

    @Override
    public void score(long user, @Nonnull MutableSparseVector scores) {
        SparseVector userVector = getUserRatingVector(user);

        // TODO Score items for this user using user-user collaborative filtering

         CosineVectorSimilarity cosSimilarity = new CosineVectorSimilarity();


        
        double userMeanRating = userVector.mean(); 							//User's mean rating mu_u
        MutableSparseVector userMeanVector = userVector.mutableCopy();
        
        userMeanVector.add( userMeanRating * -1.0);							// Computing mean-centered ratings of the user


        
        for (VectorEntry e: scores.fast(VectorEntry.State.EITHER)) {		// This is the loop structure to iterate over items to score
            
            long i = e.getKey();											// get item (movie ID)
            logger.debug("item {}", i);

            double sum = 0;
            double weight = 0;


            
            LongSet potentialNeighbors =  itemDao.getUsersForItem(i);		//Get all users who have rated the item(movie)


            MutableSparseVector neighborSimilarities =  MutableSparseVector.create(potentialNeighbors);
            neighborSimilarities.clear();									// Initialize a vector to save the neighbor's similarities

            for (long v : potentialNeighbors){
                if( v != user){                								// Exclude the user for which the scoring is to be done

                   SparseVector neighborVector = getUserRatingVector(v);    // Get neighbor's ratings

                    
                    double neighborMeanRating = neighborVector.mean();		// Neighbor's mean rating mu_v


                    MutableSparseVector neighborMeanVector = neighborVector.mutableCopy();
                    neighborMeanVector.add(neighborMeanRating * -1.0);		// Compute mean-centered ratings of the neighbour


                    double sim = cosSimilarity.similarity(userMeanVector, neighborMeanVector);		// Compute the cosine similarities
                    neighborSimilarities.set(v, sim);	

                   // logger.info("Potential neighbor: user id = {} cosineSimilarity = {}", v, sim);
                }

            }


            LongArrayList neighbors = neighborSimilarities.keysByValue(true);	// Sort the neighbors based on similarity values to get top 30
            
            for (long v : neighbors.subList(0, 30)){						// And then use the top 30 neighbours
                double sim = neighborSimilarities.get(v);

                SparseVector neighborVector  = getUserRatingVector(v);
                
                double neighborMeanRating = neighborVector.mean();			// Neighbor's mean rating mu_v

                
                double neighborItemRating = neighborVector.get(i);			// Neighbor's item rating r_v,i

                double diffRating = neighborItemRating - neighborMeanRating;

                logger.debug("Neighbor: user id = {} cosineSimilarity = {} diffRating = {}", v, sim, diffRating);
                sum += sim * diffRating;									//Numerator-> summation of sim(u,v)*(r_v,i-mu_v)
                weight += Math.abs(sim);									//Denominator-> summation of similarities
            }

            scores.set(e, userMeanRating +  sum / weight);					//add the item score to the scores vector
        }
    }

    /**
     * Get a user's rating vector.
     * @param user The user ID.
     * @return The rating vector.
     */
	private SparseVector getUserRatingVector(long user) {
        UserHistory<Rating> history = userDao.getEventsForUser(user, Rating.class);
        if (history == null) {
        	history = History.forUser(user);
        }
        return RatingVectorUserHistorySummarizer.makeRatingVector(history);
    }
}
