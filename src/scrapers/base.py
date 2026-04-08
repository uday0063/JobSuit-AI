from abc import ABC, abstractmethod

class BaseScraper(ABC):
    """
    Interface for all job scrapers.
    """
    @abstractmethod
    def scrape(self, query: str, location: str, max_results: int = 20) -> list[dict]:
        """
        Execute the scrape and return a list of jobs.
        Each job should follow the schema:
        {
            "title": str,
            "company": str,
            "location": str,
            "description": str,
            "url": str,
            "source": str
        }
        """
        pass
