from datetime import datetime, timedelta
import re
import requests

regex = re.compile(r'([#$@]\S+)|(http\S+)|[^\w\s]')

def clean_text(x):
    """Remove non-letter characters, hashtag, url, and convert it to lowercase"""
    return re.sub(regex, '', x.lower())

def get_last_day():
    """
    Get start and end utc time as twitter format "%Y-%m-%dT%H:%M:%SZ"
    Output: start time as 00:00 am of last day, end time as 11:59 pm
    """
    from datetime import datetime, timedelta
    last_day = (datetime.utcnow() - timedelta(days=1))
    return last_day.strftime('%Y-%m-%dT00:00:00Z'),\
    last_day.strftime('%Y-%m-%dT23:59:59Z')


def parse_datetime(year, month, day, minute=0, second=0):
    """
    Parse UTC time to twitter format of "%Y-%m-%dT%H:%M:%SZ"
    Input: year, month, day, minute(optional), second(optional)
    Output: time as "%Y-%m-%dT%H:%M:%SZ"
    """
    try:
        output = datetime(year, month, day, minute, second)
        return output.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        print("Please input at least correct utc time year, month and day!")


def bearer_oauth(token):
    """
    Method to create a header, which does authentication
    Input: Bearer token
    Output: headers when sending a one time request
    """
    headers = {}
    headers["Authorization"] = f"Bearer {token}"
    headers["User-Agent"] = "v2RecentSearchPython"
    return headers


def make_session(bearer_token):
    """
    Make a new session and setup authentication
    Input: Bearer token
    Output: A requests session for reuse purposes.
    """
    session = requests.session()
    headers = {'User-Agent': "v2RecentSearchPython",
        "Authorization": f"Bearer {bearer_token}"}
    session.headers = headers
    return session
