"""Pure functions for cleaning, reference retrieval, and a bounded RSS reader."""
import html
import re
import unicodedata
import urllib.request
from urllib.parse import urlparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FEEDS = {
    'Teknologi': 'https://www.antaranews.com/rss/tekno.xml',
    'Ekonomi': 'https://www.antaranews.com/rss/ekonomi.xml',
    'Lingkungan': 'https://www.antaranews.com/rss/warta-bumi.xml',
    'Terkini': 'https://www.antaranews.com/rss/terkini.xml',
}

def clean(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', html.unescape(text))).strip().lower()

def search_cases(query, cases, topic='Semua'):
    if not query.strip():
        return [dict(c, similarity=None) for c in cases if topic=='Semua' or c['topic']==topic]
    texts=[c['title']+' '+c['claim']+' '+c['topic'] for c in cases]
    vec=TfidfVectorizer(ngram_range=(1,2))
    mat=vec.fit_transform(texts)
    scores=cosine_similarity(vec.transform([clean(query)]),mat).ravel()
    out=[dict(c,similarity=float(s)) for c,s in zip(cases,scores)
         if s>0 and (topic=='Semua' or c['topic']==topic)]
    return sorted(out,key=lambda c:c['similarity'],reverse=True)

def parse_feed(raw):
    if len(raw)>1_500_000 or b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
        raise ValueError('Umpan tidak sesuai format yang diizinkan.')
    root=ET.fromstring(raw)
    items=[]; seen=set()
    for node in root.findall('./channel/item'):
        title=html.unescape(node.findtext('title','')).strip()
        url=node.findtext('link','').strip()
        parsed=urlparse(url)
        host=parsed.hostname or ''
        if parsed.scheme not in ('https','http') or not (host=='antaranews.com' or host.endswith('.antaranews.com')):
            continue
        if not title or url in seen: continue
        seen.add(url)
        date=node.findtext('pubDate','').strip()
        try:
            dt=parsedate_to_datetime(date)
            if dt.tzinfo is None: dt=dt.replace(tzinfo=timezone.utc)
            timestamp=dt.timestamp()
            date=dt.astimezone().isoformat()
        except (ValueError,TypeError,OverflowError): timestamp=0
        items.append(dict(title=title,url=url,date=date,timestamp=timestamp))
    return sorted(items,key=lambda x:x['timestamp'],reverse=True)[:18]

def fetch_feed(channel):
    url=FEEDS[channel]
    req=urllib.request.Request(url,headers={'User-Agent':'TELISIK-Education/2.0 RSS Reader'})
    with urllib.request.urlopen(req,timeout=12) as response:
        raw=response.read(1_500_001)
    items=parse_feed(raw)
    if not items: raise ValueError('Sumber belum mengirim judul berita yang dapat dibaca.')
    return {'items':items,'fetched_at':datetime.now(timezone.utc).isoformat(),'feed_url':url}
