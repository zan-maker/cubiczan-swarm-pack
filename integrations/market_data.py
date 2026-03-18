"""
Market Data Integration — Financial data feeds for the Financial Markets domain.
All open-source / free-tier APIs.
"""

from dataclasses import dataclass


@dataclass
class DataFeed:
    name: str
    provider: str
    api_type: str  # "rest", "websocket", "library"
    free_tier: bool
    install: str
    description: str


FINANCIAL_FEEDS = [
    DataFeed(
        name="Stock/ETF Prices",
        provider="Yahoo Finance",
        api_type="library",
        free_tier=True,
        install="pip install yfinance",
        description="Historical and real-time stock, ETF, options data via yfinance",
    ),
    DataFeed(
        name="Crypto Prices",
        provider="CoinGecko",
        api_type="rest",
        free_tier=True,
        install="pip install pycoingecko",
        description="Cryptocurrency prices, market cap, volume. 10-30 calls/min free.",
    ),
    DataFeed(
        name="Economic Indicators",
        provider="FRED (Federal Reserve)",
        api_type="rest",
        free_tier=True,
        install="pip install fredapi",
        description="GDP, CPI, unemployment, interest rates. Free API key required.",
    ),
    DataFeed(
        name="News Sentiment",
        provider="NewsAPI",
        api_type="rest",
        free_tier=True,
        install="pip install newsapi-python",
        description="Global news headlines. 100 requests/day free tier.",
    ),
    DataFeed(
        name="Alternative Data (GDELT)",
        provider="GDELT Project",
        api_type="rest",
        free_tier=True,
        install="pip install gdeltdoc",
        description="Global event database. News tone, themes, geo-coding.",
    ),
    DataFeed(
        name="SEC Filings",
        provider="SEC EDGAR",
        api_type="rest",
        free_tier=True,
        install="pip install sec-edgar-downloader",
        description="10-K, 10-Q, 8-K filings. Rate limit: 10 req/sec.",
    ),
]

CYBERSECURITY_FEEDS = [
    DataFeed(
        name="CVE Database",
        provider="NVD (NIST)",
        api_type="rest",
        free_tier=True,
        install="pip install nvdlib",
        description="National Vulnerability Database. CVE details, CVSS scores.",
    ),
    DataFeed(
        name="Threat Intelligence",
        provider="AlienVault OTX",
        api_type="rest",
        free_tier=True,
        install="pip install OTXv2",
        description="Open Threat Exchange. IoCs, pulses, reputation data.",
    ),
    DataFeed(
        name="IP Reputation",
        provider="AbuseIPDB",
        api_type="rest",
        free_tier=True,
        install="# REST API, no SDK needed",
        description="IP abuse reports. 1000 checks/day free.",
    ),
    DataFeed(
        name="Internet Census",
        provider="Shodan",
        api_type="rest",
        free_tier=True,
        install="pip install shodan",
        description="Internet-connected device search. Limited free queries.",
    ),
]

BUSINESS_INTEL_FEEDS = [
    DataFeed(
        name="Company Data",
        provider="SEC EDGAR",
        api_type="rest",
        free_tier=True,
        install="pip install sec-edgar-downloader",
        description="Public company filings, financial statements.",
    ),
    DataFeed(
        name="Trends",
        provider="Google Trends",
        api_type="library",
        free_tier=True,
        install="pip install pytrends",
        description="Search interest over time. Unofficial API.",
    ),
    DataFeed(
        name="Patent Data",
        provider="USPTO",
        api_type="rest",
        free_tier=True,
        install="# REST API",
        description="Patent applications and grants. Bulk download available.",
    ),
]


def get_feeds_for_domain(domain: str) -> list[DataFeed]:
    """Return available data feeds for a given domain."""
    domain_map = {
        "financial-markets": FINANCIAL_FEEDS,
        "cybersecurity": CYBERSECURITY_FEEDS,
        "business-intelligence": BUSINESS_INTEL_FEEDS,
    }
    return domain_map.get(domain, [])


def install_all_feeds(domain: str) -> list[str]:
    """Generate pip install commands for all feeds in a domain."""
    feeds = get_feeds_for_domain(domain)
    return [f.install for f in feeds if f.install.startswith("pip")]
