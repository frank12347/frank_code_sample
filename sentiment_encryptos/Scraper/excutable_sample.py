#!/usr/bin/env python3

from twitter_api import TwitterSearch
import sys
import json

"""
An excutable script for tweets retrieval
arg1: output file path
"""

if __name__ == '__main__':
    output = sys.argv[1] # file path as an argument
    bearer_token = '!!![YOUR TOKEN]'
    query = '#bitcoin -is:retweet lang:en'
    agent = TwitterSearch(bearer_token, query, last_day=True)
    print(agent)
    print('Search parameters:')
    for key, value in agent.paras.items():
        print(key, ':', value)
    result = agent.collect_tweets(wait_seconds=5, max_tweets=20000)
    with open(output, 'w') as file:
        json.dump(result, file)

    print('Tweets saved at file', output)
