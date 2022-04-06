import json
from utilities import *
import time

"""
The TwitterSearch class for twitter data extraction.
It includes topic | keywords search, user search and tweet id search.

!!!You need to apply for a bearer token to request access for twitter API
"""


class TwitterSearch:
    # API endpoint links
    __search_url = "https://api.twitter.com/2/tweets/search/recent"
    __count_url = "https://api.twitter.com/2/tweets/counts/recent"
    __id_search_url = "https://api.twitter.com/2/tweets"
    __user_search_url = "https://api.twitter.com/2/users"


    def __init__(self, bearer_token, query=None, max_per_page=100,
        max_tweets=1000000, last_day=True):
        """
        Required parameters: bearer_token, query string
        Optional parameters: maximum tweets per page, maximum tweets to get
        """
        self.max_per_page = max_per_page
        self.next_token = None
        self.session = None
        self.bearer_token = bearer_token
        self.paras = {'query':query, 'max_results':max_per_page, 'next_token':None}
        if last_day:
            start_time, end_time = get_last_day()
            self.paras.update({'start_time':start_time, 'end_time':end_time})
        if max_tweets != None:
            self.max_tweets = max_tweets


    def search(self, query=None):
        """
        Function to request context by calling get on self.session
        """
        url = TwitterSearch.__search_url
        # start a new session if no session exists
        if self.session == None:
            self.session_init()

        # start a new query if new query input
        if query != None:
            paras = self.paras.copy()
            paras['query'] = query
            response = self.session.get(url=url, params=paras)

        else:
            response = self.session.get(url=url, params=self.paras)
        if response.status_code != 200:
            raise Exception(response.status_code, response.text)
        return response.json()


    def search_by_tweet_id(self, ids, expansions=None, tweet_fields=None):
        """
        Search for tweets by tweet ids
        Input: ids -- a list of string ids; expansions -- additional fields
        Output: a json object
        """
        ids = ','.join(ids)
        response = requests.get(url=TwitterSearch.__id_search_url, \
        params={'ids':ids, 'expansions':expansions, 'tweet.fields':tweet_fields},
            headers=bearer_oauth(self.bearer_token))
        if response.status_code != 200:
            raise Exception(response.status_code, response.text)
        return response.json()


    def search_by_user_id(self, ids, expansions=None, user_fields=None):
        """
        Search for users by user ids
        Input: ids -- a list of string ids; expansions -- additional fields
        Output: a json object
        """
        ids = ','.join(ids)
        response = requests.get(url=TwitterSearch.__user_search_url, \
        params={'ids':ids, 'expansions':expansions, 'user.fields':user_fields},\
            headers=bearer_oauth(self.bearer_token))
        if response.status_code != 200:
            raise Exception(response.status_code, response.text)
        return response.json()


    def session_init(self):
        """Initilize a new session if no session exists"""
        if self.session:
            self.session.close()
        else:
            self.session = make_session(self.bearer_token)
            print('Session initilized')


    def iterator(self, wait_seconds=0):
        """
        Iterable to loop over all pages of results. The function will hold up a
            few seconds because of twitter query limits.
        Input: wait_seconds -- computed by the query limits / time limits
        """
        self.session_init()
        counter = 0
        while True:
            response = self.search()
            self.paras['next_token'] = response['meta'].get('next_token', None)
            yield response['data']
            time.sleep(wait_seconds)

        self.session.close()

    def collect_tweets(self, wait_seconds=5, max_tweets=None):
        """
        Clean text data from tweets by the iterator, store them together to one
            list of json objects.
        Input: wait_seconds due to twitter limits of requests, default 5s;
            max_tweets to limit the number of tweet returned, default as initilized.

        """
        cap = max_tweets if max_tweets != None else self.max_tweets
        tweets = []
        counter = 0
        for page in self.iterator(wait_seconds):
            if (counter >= cap) | (self.paras['next_token'] == None):
                break
            #tweets.extend([clean_text(x['text']) for x in page])
            tweets.extend([x['text'] for x in page])
            counter += len(page)

        self.session.close()
        return tweets


    def __str__(self):
        """
        Print the number of results returned
        """
        paras = {'query':self.paras['query'], 'granularity':'day'}
        response = requests.get(url=TwitterSearch.__count_url, \
        headers=bearer_oauth(self.bearer_token),\
        params=paras).json()
        total_count = response['data'][-2]
        result = 'Total {} tweets from {} to {}'.format(total_count['tweet_count'],\
            total_count['start'], total_count['end'])
        return result
