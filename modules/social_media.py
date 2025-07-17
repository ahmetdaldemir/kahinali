import os
import re
import praw
import tweepy
import pandas as pd
from textblob import TextBlob
import logging
from config import Config

class SocialMediaSentiment:
    def __init__(self):
        # Reddit
        try:
            self.reddit = praw.Reddit(
                client_id=Config.REDDIT_CLIENT_ID,
                client_secret=Config.REDDIT_CLIENT_SECRET,
                user_agent=Config.REDDIT_USER_AGENT
            )
        except Exception as e:
            self.reddit = None
            logging.warning(f"Reddit bağlantısı kurulamadı: {e}")
        
        # Twitter
        try:
            self.twitter_auth = tweepy.OAuth1UserHandler(
                Config.TWITTER_API_KEY,
                Config.TWITTER_API_SECRET,
                Config.TWITTER_ACCESS_TOKEN,
                Config.TWITTER_ACCESS_TOKEN_SECRET
            )
            self.twitter_api = tweepy.API(self.twitter_auth, wait_on_rate_limit=True)
        except Exception as e:
            self.twitter_api = None
            logging.warning(f"Twitter bağlantısı kurulamadı: {e}")
            
        self.logger = logging.getLogger(__name__)

    def fetch_reddit_posts(self, subreddits=['CryptoCurrency', 'Bitcoin', 'CryptoMarkets'], limit=100):
        posts = []
        if not self.reddit:
            return posts
            
        for subreddit in subreddits:
            try:
                for post in self.reddit.subreddit(subreddit).hot(limit=limit):
                    posts.append({
                        'source': 'reddit',
                        'subreddit': subreddit,
                        'title': post.title,
                        'created_utc': post.created_utc
                    })
            except Exception as e:
                self.logger.error(f"Reddit {subreddit} alınamadı: {e}")
        return posts

    def fetch_twitter_posts(self, query='crypto', count=100):
        tweets = []
        if not self.twitter_api:
            return tweets
            
        try:
            for tweet in tweepy.Cursor(self.twitter_api.search_tweets, q=query, lang='en', tweet_mode='extended').items(count):
                tweets.append({
                    'source': 'twitter',
                    'user': tweet.user.screen_name,
                    'text': tweet.full_text,
                    'created_at': tweet.created_at
                })
        except Exception as e:
            self.logger.error(f"Twitter araması alınamadı: {e}")
        return tweets

    def clean_text(self, text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)
        text = text.lower()
        return text

    def sentiment_score(self, text):
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception:
            return 0

    def analyze_sentiment(self, posts, text_key='title'):
        for post in posts:
            text = post.get(text_key, '')
            clean = self.clean_text(text)
            post['sentiment'] = self.sentiment_score(clean)
        return posts

    def extract_coin_mentions(self, posts, coin_list):
        coin_counts = {coin: 0 for coin in coin_list}
        for post in posts:
            text = self.clean_text(post.get('title', '') + ' ' + post.get('text', ''))
            for coin in coin_list:
                if coin.lower() in text:
                    coin_counts[coin] += 1
        # En çok bahsedilen ilk 10 coin
        sorted_coins = sorted(coin_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_coins[:10]

    def get_social_sentiment(self, coin_list, reddit_limit=100, twitter_count=100):
        reddit_posts = self.fetch_reddit_posts(limit=reddit_limit)
        twitter_posts = self.fetch_twitter_posts(count=twitter_count)
        all_posts = reddit_posts + twitter_posts
        
        # Eğer hiç post yoksa, varsayılan değerler döndür
        if not all_posts:
            return {coin: 0 for coin in coin_list}, []
            
        all_posts = self.analyze_sentiment(all_posts, text_key='title' if 'title' in all_posts[0] else 'text')
        # Coin mention analizi
        top_coins = self.extract_coin_mentions(all_posts, coin_list)
        # Her coin için ortalama sentiment
        coin_sentiments = {}
        for coin, _ in top_coins:
            coin_posts = [p for p in all_posts if coin.lower() in self.clean_text(p.get('title', '') + ' ' + p.get('text', ''))]
            if coin_posts:
                avg_sentiment = sum(p['sentiment'] for p in coin_posts) / len(coin_posts)
            else:
                avg_sentiment = 0
            coin_sentiments[coin] = avg_sentiment
        return coin_sentiments, top_coins 