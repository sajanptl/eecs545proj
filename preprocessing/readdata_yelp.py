# EECS 545 Final Project - LDA
# Read Yelp Dataset (json file) and convert it the format usable by the LDA API
# 11/30/2015
# Xiang Li

import json
import random
from readdata_ap import *

random.seed(1)


# process business info
print 'Reading business information'
businesses = []
restaurant_ids = []
num_line = 0
num_restaurant = 0

for line in open('data/yelp/yelp_academic_dataset_business.json', 'r'):
    business =  json.loads(line)
    if ('Restaurants' in business['categories']):   # if the business is restaurant
        num_restaurant = num_restaurant + 1
        if num_restaurant % 1000 == 0:
            print 'Restaurant number ', num_restaurant
        businesses.append(business)   # add to restaurant list
        restaurant_ids.append(business['business_id'])  # add to restaurant ids

restaurant_ids_set = set(restaurant_ids)


# process reviews
print 'Reading reviews'
reviews = []
num_line = 0
num_restaurant_review = 0

for line in open('data/yelp/yelp_academic_dataset_review.json', 'r'):
    review_line = json.loads(line)
    if (review_line['business_id'] in restaurant_ids_set):
        num_restaurant_review = num_restaurant_review + 1
        if random.randrange(100) == 0:      # random get 1 review from 200
            print 'Restaurant number ', num_restaurant_review
            review = string_to_token(review_line['text'])
            reviews.append(review)

# vectorize
print 'Vectorize'
X = vectorize(reviews, 'data/yelp/LDA_input/')
Xtrain, Xtest = split_data(X)
save_data(Xtrain, '', 'data/yelp/LDA_input/')
save_data(Xtest, '_test', 'data/yelp/LDA_input/')
