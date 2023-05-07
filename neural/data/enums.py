from enum import Enum
from neural.common.constants import CALENDAR
from abc import ABC
from neural.client.alpaca import AlpacaDataClient


class FeatureType(Enum):

    """
    Enumeration class that defines constants for the different types of features in data.
    Typically used to define the feature schema of a dataset or stream. This means that a boolean mask
    is created for each feature type that indicates the location corresponding columns is in the dataset or stream.
    features types are universal and can be found in any data source and dataset.This enumeration is also useful for 
    feature engineering where a certain type of input is required to commpute a derivative feature such as MACD or RSI financial indicators.
    Downloaders use the value of feature enums as a filter to generate boolean mask for corresponding features.
    This means that dowlnloaders look for columns with name equal to the value of the enum, for example 'OPEN'
    corresponding to ASSET_OPEN_PRICE enum and then return a boolean mask that indicates the location of any 
    column names in the dataset/stream that matches the name 'OPEN'. Column names in Output of downloaders are 
    guaranteed to be consistent with this method to create valid boolean masks.


    Attributes:
    --------------
        ASSET_OPEN_PRICE (str): The type of column for opening price of an asset within an interval.
        ASSET_HIGH_PRICE (str): The type of column for high price data of asset within an interval.
        ASSET_LOW_PRICE (str): The type of column for low price data.
        ASSET_CLOSE_PRICE (str): The type of column for closing price data. by default training 
            environments use this column as the price of asset for placing orders at the end of each time interval.
        ASSET_BID_PRICE (str): The type of column for bid price data.
        ASSET_ASK_PRICE (str): The type of column for ask price data.
        SENTIMENT (str): The type of column for sentiment data. usualy a float between -1 and 1.
        1 meaning very positive and -1 meaning very negative. However it is possible to have
        a different range of values, or multiple sentiment columns.
        EMBEDDING (str): The type of column for word embedding data. This is a vector of floats
        that is the result of a word embedding algorithm such as word2vec or GloVe aggregated
        over a time interval.
        TEXT (str): The type of column for text data. This is a string that is the result of
        concatenation of all the text data in a time interval.
    
    Notes:
    --------------
        This is not a representative list of all the feature types that can be used in datasets and streams.
        This is just a list of the most common feature types that are frequently used in a way that
        needs to be distinguished them from other feature types, for example for feature engineering, 
        using OHLC prices, or for filtering text columns for passing to large language models.
        This list can be extended to include other feature types that are not included here using
        the FeatureType.MY_FEATURE_TYPE = 'KEY_WORD' syntax. Feature schema will automatically
        look for columns that contain name 'KEY_WORD' and create a boolean mask for them in feature shema.
    """

    ASSET_OPEN_PRICE = 'OPEN'
    ASSET_HIGH_PRICE = 'HIGH'
    ASSET_LOW_PRICE = 'LOW'
    ASSET_CLOSE_PRICE = 'CLOSE'
    ASSET_BID_PRICE = 'BID'
    ASSET_ASK_PRICE = 'ASK'
    ASSET_TEXT_EMBEDDING = 'EMBEDDING'
    ASSET_SENTIMENT_SCORE = 'SENTIMENT'
    ASSET_TEXT = 'TEXT'


class AssetType(str, Enum):

    """
    An enum class representing different categories of financial instruments.

    STOCK: Represents ownership in a publicly traded corporation. Stocks can be traded on stock exchanges, 
    and their value can fluctuate based on various factors such as the company's financial performance and market conditions.

    CURRENCY: Represents a unit of currency, such as the US dollar, Euro, or Japanese yen. Currencies can be traded on the 
    foreign exchange market (Forex), and their value can fluctuate based on various factors such as interest rates, g
    eopolitical events, and market sentiment.

    CRYPTOCURRENCY: Represents a digital or virtual currency that uses cryptography for security and operates 
    independently of a central bank. Cryptocurrencies can be bought and sold on various cryptocurrency exchanges, 
    and their value can fluctuate based on various factors such as supply and demand, adoption rates, and regulatory developments.

    FUTURES: Represents a standardized contract to buy or sell an underlying asset at a predetermined price and date in the future. 
    Futures can be traded on various futures exchanges, and their value can fluctuate based on various factors such as supply and demand, 
    geopolitical events, and market sentiment.

    OPTIONS: Represents a financial contract that gives the buyer the right, but not the obligation, to buy or sell an underlying 
    asset at a specified price on or before a specified date. Options can be traded on various options exchanges, 
    and their value can fluctuate based on various factors such as the price of the underlying asset, 
    the time until expiration, and market volatility.

    BOND: Represents debt issued by a company or government entity. Bonds can be traded on various bond markets, 
    and their value can fluctuate based on various factors such as interest rates, creditworthiness, and market conditions.

    EXCHANGE_TRADED_FUND: Represents a type of investment fund traded on stock exchanges, similar to mutual funds, 
    but with shares that can be bought and sold like individual stocks. ETFs can provide exposure to a wide range of asset classes, 
    such as stocks, bonds, and commodities, and their value can fluctuate based on various factors such as the performance of 
    the underlying assets and market conditions.

    MUTUAL_FUND: Represents a professionally managed pool of money from many investors, used to purchase a diversified mix of stocks, 
    bonds, or other assets. Mutual funds can be bought and sold through fund companies or brokerages, and their value can 
    fluctuate based on various factors such as the performance of the underlying assets and market conditions.

    COMMODITY: Represents a physical or virtual product that can be bought or sold, such as gold, oil, or currencies. 
    Commodities can be traded on various commodity exchanges, and their value can fluctuate based on various factors 
    such as supply and demand, geopolitical events, and market sentiment.
    """

    STOCK = 'STOCK'
    CURRENCY = 'CURRENCY'
    CRYPTO = 'CRYPTOCURRENCY'
    FUTURES = 'FUTURES'
    OPTIONS = 'OPTIONS'
    BOND = 'BOND'
    EXCHANGE_TRADED_FUND = 'ETF'
    MUTUAL_FUND = 'MUTUAL_FUND'
    COMMODITY = 'COMMODITY'