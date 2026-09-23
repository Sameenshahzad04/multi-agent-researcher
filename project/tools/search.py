import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from config.setting import config

def get_web_search_tool():
    """
    Returns TavilySearchResults if TAVILY_API_KEY is valid,
    otherwise automatically falls back to DuckDuckGoSearchResults (100% free).
    """
    tavily_key = config.TAVILY_API_KEY

    if tavily_key and tavily_key.strip() != "" and tavily_key != "your_tavily_api_key":
        os.environ["TAVILY_API_KEY"] = tavily_key
        return TavilySearchResults(
            max_results=3,
            name="web_search",
            description=(
                "Search the live web for current events, news, or facts you are not "
                "confident about. Input should be a short search query string."
            ),
        )
    else:
        # region="us-en" fixes the wt.wikipedia.org DNS error
        wrapper = DuckDuckGoSearchAPIWrapper(region="us-en", max_results=3)
        return DuckDuckGoSearchResults(
            api_wrapper=wrapper,
            max_results=3,
            name="web_search",
            description=(
                "Search the live web for current events, news, or facts you are not "
                "confident about. Input should be a short plain search query string."
            ),
        )

web_search_tool = get_web_search_tool()