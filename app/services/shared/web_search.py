from langchain_community.utilities import GoogleSerperAPIWrapper

from app.core.config import logger


# 공용 웹 검색 유틸을 실행해 최신성 보강용 텍스트 결과를 반환한다.
def run_web_search(query: str) -> str:
    logger.warning("Web search fallback triggered | query=%r", query)
    return GoogleSerperAPIWrapper().run(query)
