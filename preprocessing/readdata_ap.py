# EECS 545 Final Project - LDA
# Read Sample AP Dataset (txt file) and convert it the format usable by the LDA API
# 12/02/2015

num_line = 0
num_document = 0

output = open('../data/ap/ap_cleaned.txt', 'w')

with open('../data/ap/ap.txt', 'r') as f:
    for line in f:
        num_line = num_line + 1
        if num_line % 6 == 4:   # the line with the document
            num_document = num_document + 1
            print 'Document ', num_document
            print line
            output.write(line)

output.close()
