# EECS 545 Final Project - LDA
# Read Yelp Dataset (json file) and convert it the format usable by the LDA API
# 11/30/2015

import json

# read business info
print 'Reading business information'
businesses = []
restaurant_ids = []
num_line = 0
num_restaurant = 0

for line in open('../data/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json', 'r'):
    num_line = num_line + 1
    print 'Reading business line ', num_line
    business =  json.loads(line)
    if ('Restaurants' in business['categories']):
        num_restaurant = num_restaurant + 1
        print 'Restaurant number ', num_restaurant
        businesses.append(business)
        restaurant_ids.append(business['business_id'])


print 'Reading business information'
reviews = []
num_line = 0
num_restaurant_review = 0

for line in open('../data/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json', 'r'):
    num_line = num_line + 1
    print 'Reading review line ', num_line
    review = json.loads(line)
    if (review['business_id'] in restaurant_ids):
        num_restaurant_review = num_restaurant_review + 1
        print 'Restaurant number ', num_restaurant_review
        reviews.append(review)

print 'Done'
