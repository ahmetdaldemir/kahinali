import feedparser
import requests
import pandas as pd
from textblob import TextBlob
import logging
from config import Config

class NewsAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.news_api_key = Config.NEWS_API_KEY
        self.rss_feeds = [
            'https://cointelegraph.com/rss',
            'https://news.bitcoin.com/feed/',
            'https://cryptopotato.com/feed/',
            'https://www.coindesk.com/arc/outboundfeeds/rss/'
        ]

    def fetch_rss_news(self, limit=50):
        news = []
        for url in self.rss_feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:limit]:
                    news.append({
                        'source': url,
                        'title': entry.title,
                        'published': entry.published if 'published' in entry else '',
                        'summary': entry.summary if 'summary' in entry else ''
                    })
            except Exception as e:
                self.logger.error(f"RSS haber alınamadı: {e}")
        return news

    def fetch_newsapi_news(self, query='crypto', limit=50):
        if not self.news_api_key:
            return []
        url = f'https://newsapi.org/v2/everything?q={query}&language=en&apiKey={self.news_api_key}&pageSize={limit}'
        try:
            resp = requests.get(url)
            data = resp.json()
            articles = data.get('articles', [])
            news = []
            for article in articles:
                news.append({
                    'source': article['source']['name'],
                    'title': article['title'],
                    'published': article['publishedAt'],
                    'summary': article['description']
                })
            return news
        except Exception as e:
            self.logger.error(f"NewsAPI haber alınamadı: {e}")
            return []

    def clean_text(self, text):
        import re
        if text is None:
            return ""
        text = str(text)  # String'e çevir
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)
        text = text.lower()
        return text

    def sentiment_score(self, text):
        try:
            if text is None or text == "":
                return 0
            blob = TextBlob(str(text))
            return blob.sentiment.polarity
        except Exception:
            return 0

    def analyze_sentiment(self, news, text_key='title'):
        for item in news:
            text = item.get(text_key, '')
            if text is None:
                text = ""
            clean = self.clean_text(text)
            item['sentiment'] = self.sentiment_score(clean)
        return news

    def get_news_impact(self, coin_list, news_limit=50):
        try:
            rss_news = self.fetch_rss_news(limit=news_limit)
            api_news = self.fetch_newsapi_news(limit=news_limit)
            all_news = rss_news + api_news
            all_news = self.analyze_sentiment(all_news, text_key='title')
            
            # Coin mention analizi
            coin_impacts = {}
            for coin in coin_list:
                if coin is None:
                    continue
                coin_lower = str(coin).lower()
                coin_news = []
                for n in all_news:
                    title = n.get('title', '') or ""
                    summary = n.get('summary', '') or ""
                    combined_text = title + ' ' + summary
                    if coin_lower in self.clean_text(combined_text):
                        coin_news.append(n)
                
                if coin_news:
                    avg_sentiment = sum(n.get('sentiment', 0) for n in coin_news) / len(coin_news)
                    impact_score = avg_sentiment * len(coin_news)
                else:
                    avg_sentiment = 0
                    impact_score = 0
                coin_impacts[coin] = {
                    'avg_sentiment': avg_sentiment, 
                    'impact_score': impact_score, 
                    'news_count': len(coin_news)
                }
            return coin_impacts, all_news
        except Exception as e:
            self.logger.error(f"Haber analizi hatası: {e}")
            # Hata durumunda boş sonuç döndür
            coin_impacts = {}
            for coin in coin_list:
                if coin is not None:
                    coin_impacts[coin] = {
                        'avg_sentiment': 0, 
                        'impact_score': 0.5, 
                        'news_count': 0
                    }
            return coin_impacts, [] 