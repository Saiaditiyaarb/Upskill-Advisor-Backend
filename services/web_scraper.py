"""
Advanced Web Scraper for Course Data Collection

This module provides a comprehensive web scraping framework for collecting course data
from multiple educational platforms with proper rate limiting, error handling, and
ethical scraping practices.

Features:
- Multi-platform support (Coursera, edX, Udemy, Khan Academy, FutureLearn, etc.)
- Intelligent rate limiting and retry mechanisms
- Robots.txt compliance
- User-agent rotation
- Proxy support
- Data validation and standardization
- Comprehensive logging and monitoring
"""

import asyncio
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import aiohttp
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from schemas.course import Course

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ScrapingConfig:
    """Configuration for web scraping operations"""
    base_delay: float = 1.0  # Base delay between requests in seconds
    max_delay: float = 10.0  # Maximum delay for exponential backoff
    max_retries: int = 3  # Maximum number of retries for failed requests
    timeout: int = 30  # Request timeout in seconds
    respect_robots_txt: bool = True  # Whether to check robots.txt
    use_proxy: bool = False  # Whether to use proxy rotation
    proxy_list: List[str] = None  # List of proxy URLs
    user_agent_rotation: bool = True  # Whether to rotate user agents
    concurrent_requests: int = 5  # Maximum concurrent requests
    rate_limit_per_minute: int = 60  # Maximum requests per minute


@dataclass
class CourseData:
    """Standardized course data structure"""
    title: str
    provider: str
    url: str
    description: Optional[str] = None
    instructor: Optional[str] = None
    duration: Optional[str] = None
    difficulty: Optional[str] = None
    price: Optional[str] = None
    rating: Optional[float] = None
    enrollment_count: Optional[int] = None
    skills: List[str] = None
    category: Optional[str] = None
    language: Optional[str] = None
    certificate_available: Optional[bool] = None
    scraped_at: Optional[datetime] = None

    def __post_init__(self):
        if self.skills is None:
            self.skills = []
        if self.scraped_at is None:
            self.scraped_at = datetime.now()


class RateLimiter:
    """Rate limiter to control request frequency"""

    def __init__(self, max_requests: int = 60, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire permission to make a request"""
        async with self._lock:
            now = time.time()
            # Remove old requests outside the time window
            self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]

            if len(self.requests) >= self.max_requests:
                # Calculate wait time
                oldest_request = min(self.requests)
                wait_time = self.time_window - (now - oldest_request)
                if wait_time > 0:
                    logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)

            self.requests.append(now)


class RobotsChecker:
    """Check robots.txt compliance for websites"""

    def __init__(self):
        self._cache = {}

    def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        """Check if URL can be fetched according to robots.txt"""
        try:
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

            if base_url not in self._cache:
                robots_url = urljoin(base_url, "/robots.txt")
                rp = RobotFileParser()
                rp.set_url(robots_url)
                rp.read()
                self._cache[base_url] = rp

            return self._cache[base_url].can_fetch(user_agent, url)
        except Exception as e:
            logger.warning(f"Could not check robots.txt for {url}: {e}")
            return True  # Default to allowing if robots.txt check fails


class BaseScraper(ABC):
    """Abstract base class for platform-specific scrapers"""

    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.session = None
        self.rate_limiter = RateLimiter(config.rate_limit_per_minute, 60)
        self.robots_checker = RobotsChecker()
        self.user_agent = UserAgent() if config.user_agent_rotation else None

    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(limit=self.config.concurrent_requests)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def get_headers(self) -> Dict[str, str]:
        """Get headers for HTTP requests"""
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        if self.user_agent:
            headers['User-Agent'] = self.user_agent.random
        else:
            headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

        return headers

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def fetch_page(self, url: str) -> Optional[str]:
        """Fetch a web page with rate limiting and error handling"""
        # Check robots.txt compliance
        if self.config.respect_robots_txt and not self.robots_checker.can_fetch(url):
            logger.warning(f"Robots.txt disallows fetching {url}")
            return None

        # Apply rate limiting
        await self.rate_limiter.acquire()

        # Add random delay to mimic human behavior
        delay = random.uniform(self.config.base_delay, self.config.base_delay * 2)
        await asyncio.sleep(delay)

        try:
            headers = self.get_headers()
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    content = await response.text()
                    logger.debug(f"Successfully fetched {url}")
                    return content
                elif response.status == 429:
                    # Rate limited - wait longer
                    retry_after = response.headers.get('Retry-After', '60')
                    wait_time = int(retry_after)
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds")
                    await asyncio.sleep(wait_time)
                    raise aiohttp.ClientError(f"Rate limited: {response.status}")
                else:
                    logger.error(f"HTTP {response.status} for {url}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            raise

    @abstractmethod
    async def search_courses(self, query: str, limit: int = 50) -> List[CourseData]:
        """Search for courses on the platform"""
        pass

    @abstractmethod
    async def get_course_details(self, course_url: str) -> Optional[CourseData]:
        """Get detailed information about a specific course"""
        pass

    @abstractmethod
    def get_platform_name(self) -> str:
        """Get the name of the platform"""
        pass


class CourseraScraper(BaseScraper):
    """Scraper for Coursera courses"""

    def get_platform_name(self) -> str:
        return "Coursera"

    async def search_courses(self, query: str, limit: int = 50) -> List[CourseData]:
        """Search for courses on Coursera"""
        courses = []
        try:
            # Coursera search URL
            search_url = f"https://www.coursera.org/search?query={query.replace(' ', '%20')}"
            content = await self.fetch_page(search_url)

            if not content:
                return courses

            soup = BeautifulSoup(content, 'html.parser')

            # Find course cards (this selector may need updating based on Coursera's current structure)
            course_cards = soup.find_all('div', class_='cds-9')[:limit]

            for card in course_cards:
                try:
                    title_elem = card.find('h3') or card.find('h2')
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)

                    # Extract course URL
                    link_elem = card.find('a', href=True)
                    course_url = urljoin("https://www.coursera.org", link_elem['href']) if link_elem else None

                    # Extract additional metadata
                    instructor_elem = card.find('span', class_='instructor-name')
                    instructor = instructor_elem.get_text(strip=True) if instructor_elem else None

                    rating_elem = card.find('span', class_='rating')
                    rating = None
                    if rating_elem:
                        try:
                            rating = float(rating_elem.get_text(strip=True))
                        except ValueError:
                            pass

                    course_data = CourseData(
                        title=title,
                        provider=self.get_platform_name(),
                        url=course_url,
                        instructor=instructor,
                        rating=rating
                    )
                    courses.append(course_data)

                except Exception as e:
                    logger.error(f"Error parsing Coursera course card: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error searching Coursera courses: {e}")

        return courses

    async def get_course_details(self, course_url: str) -> Optional[CourseData]:
        """Get detailed information about a Coursera course"""
        try:
            content = await self.fetch_page(course_url)
            if not content:
                return None

            soup = BeautifulSoup(content, 'html.parser')

            # Extract course details (selectors may need updating)
            title = soup.find('h1')
            title = title.get_text(strip=True) if title else "Unknown Title"

            description_elem = soup.find('div', class_='description')
            description = description_elem.get_text(strip=True) if description_elem else None

            # Extract skills
            skills = []
            skills_section = soup.find('div', class_='skills')
            if skills_section:
                skill_tags = skills_section.find_all('span')
                skills = [tag.get_text(strip=True) for tag in skill_tags]

            return CourseData(
                title=title,
                provider=self.get_platform_name(),
                url=course_url,
                description=description,
                skills=skills
            )

        except Exception as e:
            logger.error(f"Error getting Coursera course details: {e}")
            return None


class EdXScraper(BaseScraper):
    """Scraper for edX courses"""

    def get_platform_name(self) -> str:
        return "edX"

    async def search_courses(self, query: str, limit: int = 50) -> List[CourseData]:
        """Search for courses on edX"""
        courses = []
        try:
            search_url = f"https://www.edx.org/search?q={query.replace(' ', '+')}"
            content = await self.fetch_page(search_url)

            if not content:
                return courses

            soup = BeautifulSoup(content, 'html.parser')

            # Find course cards
            course_cards = soup.find_all('div', class_='discovery-card')[:limit]

            for card in course_cards:
                try:
                    title_elem = card.find('h3') or card.find('h2')
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)

                    link_elem = card.find('a', href=True)
                    course_url = urljoin("https://www.edx.org", link_elem['href']) if link_elem else None

                    course_data = CourseData(
                        title=title,
                        provider=self.get_platform_name(),
                        url=course_url
                    )
                    courses.append(course_data)

                except Exception as e:
                    logger.error(f"Error parsing edX course card: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error searching edX courses: {e}")

        return courses

    async def get_course_details(self, course_url: str) -> Optional[CourseData]:
        """Get detailed information about an edX course"""
        try:
            content = await self.fetch_page(course_url)
            if not content:
                return None

            soup = BeautifulSoup(content, 'html.parser')

            title = soup.find('h1')
            title = title.get_text(strip=True) if title else "Unknown Title"

            return CourseData(
                title=title,
                provider=self.get_platform_name(),
                url=course_url
            )

        except Exception as e:
            logger.error(f"Error getting edX course details: {e}")
            return None


class UdemyScraper(BaseScraper):
    """Scraper for Udemy courses"""

    def get_platform_name(self) -> str:
        return "Udemy"

    async def search_courses(self, query: str, limit: int = 50) -> List[CourseData]:
        """Search for courses on Udemy"""
        courses = []
        try:
            search_url = f"https://www.udemy.com/courses/search/?q={query.replace(' ', '+')}"
            content = await self.fetch_page(search_url)

            if not content:
                return courses

            soup = BeautifulSoup(content, 'html.parser')

            # Find course cards
            course_cards = soup.find_all('div', class_='course-card--container')[:limit]

            for card in course_cards:
                try:
                    title_elem = card.find('h3')
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)

                    link_elem = card.find('a', href=True)
                    course_url = urljoin("https://www.udemy.com", link_elem['href']) if link_elem else None

                    # Extract price
                    price_elem = card.find('span', class_='price-text')
                    price = price_elem.get_text(strip=True) if price_elem else None

                    course_data = CourseData(
                        title=title,
                        provider=self.get_platform_name(),
                        url=course_url,
                        price=price
                    )
                    courses.append(course_data)

                except Exception as e:
                    logger.error(f"Error parsing Udemy course card: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error searching Udemy courses: {e}")

        return courses

    async def get_course_details(self, course_url: str) -> Optional[CourseData]:
        """Get detailed information about a Udemy course"""
        try:
            content = await self.fetch_page(course_url)
            if not content:
                return None

            soup = BeautifulSoup(content, 'html.parser')

            title = soup.find('h1')
            title = title.get_text(strip=True) if title else "Unknown Title"

            return CourseData(
                title=title,
                provider=self.get_platform_name(),
                url=course_url
            )

        except Exception as e:
            logger.error(f"Error getting Udemy course details: {e}")
            return None


class KhanAcademyScraper(BaseScraper):
    """Scraper for Khan Academy courses"""

    def get_platform_name(self) -> str:
        return "Khan Academy"

    async def search_courses(self, query: str, limit: int = 50) -> List[CourseData]:
        """Search for courses on Khan Academy"""
        courses = []
        try:
            # Khan Academy uses a different structure - search through their API or browse pages
            search_url = f"https://www.khanacademy.org/search?page_search_query={query.replace(' ', '%20')}"
            content = await self.fetch_page(search_url)

            if not content:
                return courses

            soup = BeautifulSoup(content, 'html.parser')

            # Find course/lesson cards
            course_cards = soup.find_all('div', class_='_1hr8vhm')[:limit]

            for card in course_cards:
                try:
                    title_elem = card.find('h3') or card.find('h2') or card.find('a')
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)

                    link_elem = card.find('a', href=True)
                    course_url = urljoin("https://www.khanacademy.org", link_elem['href']) if link_elem else None

                    course_data = CourseData(
                        title=title,
                        provider=self.get_platform_name(),
                        url=course_url,
                        price="Free",  # Khan Academy is free
                        certificate_available=False
                    )
                    courses.append(course_data)

                except Exception as e:
                    logger.error(f"Error parsing Khan Academy course card: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error searching Khan Academy courses: {e}")

        return courses

    async def get_course_details(self, course_url: str) -> Optional[CourseData]:
        """Get detailed information about a Khan Academy course"""
        try:
            content = await self.fetch_page(course_url)
            if not content:
                return None

            soup = BeautifulSoup(content, 'html.parser')

            title = soup.find('h1')
            title = title.get_text(strip=True) if title else "Unknown Title"

            description_elem = soup.find('div', class_='description') or soup.find('p')
            description = description_elem.get_text(strip=True) if description_elem else None

            return CourseData(
                title=title,
                provider=self.get_platform_name(),
                url=course_url,
                description=description,
                price="Free",
                certificate_available=False
            )

        except Exception as e:
            logger.error(f"Error getting Khan Academy course details: {e}")
            return None


class FutureLearnScraper(BaseScraper):
    """Scraper for FutureLearn courses"""

    def get_platform_name(self) -> str:
        return "FutureLearn"

    async def search_courses(self, query: str, limit: int = 50) -> List[CourseData]:
        """Search for courses on FutureLearn"""
        courses = []
        try:
            search_url = f"https://www.futurelearn.com/search?q={query.replace(' ', '+')}"
            content = await self.fetch_page(search_url)

            if not content:
                return courses

            soup = BeautifulSoup(content, 'html.parser')

            # Find course cards
            course_cards = soup.find_all('article', class_='m-card')[:limit]

            for card in course_cards:
                try:
                    title_elem = card.find('h3') or card.find('h2')
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)

                    link_elem = card.find('a', href=True)
                    course_url = urljoin("https://www.futurelearn.com", link_elem['href']) if link_elem else None

                    # Extract duration
                    duration_elem = card.find('span', class_='duration')
                    duration = duration_elem.get_text(strip=True) if duration_elem else None

                    course_data = CourseData(
                        title=title,
                        provider=self.get_platform_name(),
                        url=course_url,
                        duration=duration
                    )
                    courses.append(course_data)

                except Exception as e:
                    logger.error(f"Error parsing FutureLearn course card: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error searching FutureLearn courses: {e}")

        return courses

    async def get_course_details(self, course_url: str) -> Optional[CourseData]:
        """Get detailed information about a FutureLearn course"""
        try:
            content = await self.fetch_page(course_url)
            if not content:
                return None

            soup = BeautifulSoup(content, 'html.parser')

            title = soup.find('h1')
            title = title.get_text(strip=True) if title else "Unknown Title"

            description_elem = soup.find('div', class_='course-description')
            description = description_elem.get_text(strip=True) if description_elem else None

            return CourseData(
                title=title,
                provider=self.get_platform_name(),
                url=course_url,
                description=description
            )

        except Exception as e:
            logger.error(f"Error getting FutureLearn course details: {e}")
            return None


class PluralsightScraper(BaseScraper):
    """Scraper for Pluralsight courses"""

    def get_platform_name(self) -> str:
        return "Pluralsight"

    async def search_courses(self, query: str, limit: int = 50) -> List[CourseData]:
        """Search for courses on Pluralsight"""
        courses = []
        try:
            search_url = f"https://www.pluralsight.com/search?q={query.replace(' ', '%20')}"
            content = await self.fetch_page(search_url)

            if not content:
                return courses

            soup = BeautifulSoup(content, 'html.parser')

            # Find course cards
            course_cards = soup.find_all('div', class_='search-result')[:limit]

            for card in course_cards:
                try:
                    title_elem = card.find('h3') or card.find('a')
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)

                    link_elem = card.find('a', href=True)
                    course_url = urljoin("https://www.pluralsight.com", link_elem['href']) if link_elem else None

                    # Extract level/difficulty
                    level_elem = card.find('span', class_='level')
                    difficulty = level_elem.get_text(strip=True) if level_elem else None

                    # Extract duration
                    duration_elem = card.find('span', class_='duration')
                    duration = duration_elem.get_text(strip=True) if duration_elem else None

                    course_data = CourseData(
                        title=title,
                        provider=self.get_platform_name(),
                        url=course_url,
                        difficulty=difficulty,
                        duration=duration
                    )
                    courses.append(course_data)

                except Exception as e:
                    logger.error(f"Error parsing Pluralsight course card: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error searching Pluralsight courses: {e}")

        return courses

    async def get_course_details(self, course_url: str) -> Optional[CourseData]:
        """Get detailed information about a Pluralsight course"""
        try:
            content = await self.fetch_page(course_url)
            if not content:
                return None

            soup = BeautifulSoup(content, 'html.parser')

            title = soup.find('h1')
            title = title.get_text(strip=True) if title else "Unknown Title"

            description_elem = soup.find('div', class_='course-description')
            description = description_elem.get_text(strip=True) if description_elem else None

            return CourseData(
                title=title,
                provider=self.get_platform_name(),
                url=course_url,
                description=description
            )

        except Exception as e:
            logger.error(f"Error getting Pluralsight course details: {e}")
            return None


class LinkedInLearningScraper(BaseScraper):
    """Scraper for LinkedIn Learning courses"""

    def get_platform_name(self) -> str:
        return "LinkedIn Learning"

    async def search_courses(self, query: str, limit: int = 50) -> List[CourseData]:
        """Search for courses on LinkedIn Learning"""
        courses = []
        try:
            search_url = f"https://www.linkedin.com/learning/search?keywords={query.replace(' ', '%20')}"
            content = await self.fetch_page(search_url)

            if not content:
                return courses

            soup = BeautifulSoup(content, 'html.parser')

            # Find course cards
            course_cards = soup.find_all('div', class_='search-result__wrapper')[:limit]

            for card in course_cards:
                try:
                    title_elem = card.find('h3') or card.find('h4')
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)

                    link_elem = card.find('a', href=True)
                    course_url = link_elem['href'] if link_elem else None
                    if course_url and not course_url.startswith('http'):
                        course_url = urljoin("https://www.linkedin.com", course_url)

                    # Extract instructor
                    instructor_elem = card.find('span', class_='instructor-name')
                    instructor = instructor_elem.get_text(strip=True) if instructor_elem else None

                    # Extract duration
                    duration_elem = card.find('span', class_='course-duration')
                    duration = duration_elem.get_text(strip=True) if duration_elem else None

                    course_data = CourseData(
                        title=title,
                        provider=self.get_platform_name(),
                        url=course_url,
                        instructor=instructor,
                        duration=duration
                    )
                    courses.append(course_data)

                except Exception as e:
                    logger.error(f"Error parsing LinkedIn Learning course card: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error searching LinkedIn Learning courses: {e}")

        return courses

    async def get_course_details(self, course_url: str) -> Optional[CourseData]:
        """Get detailed information about a LinkedIn Learning course"""
        try:
            content = await self.fetch_page(course_url)
            if not content:
                return None

            soup = BeautifulSoup(content, 'html.parser')

            title = soup.find('h1')
            title = title.get_text(strip=True) if title else "Unknown Title"

            description_elem = soup.find('div', class_='course-description')
            description = description_elem.get_text(strip=True) if description_elem else None

            return CourseData(
                title=title,
                provider=self.get_platform_name(),
                url=course_url,
                description=description
            )

        except Exception as e:
            logger.error(f"Error getting LinkedIn Learning course details: {e}")
            return None


class WebScrapingManager:
    """Main manager for coordinating web scraping across multiple platforms"""

    def __init__(self, config: ScrapingConfig = None):
        self.config = config or ScrapingConfig()
        self.scrapers = {
            'coursera': CourseraScraper,
            'edx': EdXScraper,
            'udemy': UdemyScraper,
            'khan_academy': KhanAcademyScraper,
            'futurelearn': FutureLearnScraper,
            'pluralsight': PluralsightScraper,
            'linkedin_learning': LinkedInLearningScraper,
        }

    async def search_all_platforms(self, query: str, platforms: List[str] = None, limit_per_platform: int = 20) -> List[CourseData]:
        """Search for courses across multiple platforms"""
        if platforms is None:
            platforms = list(self.scrapers.keys())

        all_courses = []

        # Create tasks for concurrent scraping
        tasks = []
        for platform in platforms:
            if platform in self.scrapers:
                scraper_class = self.scrapers[platform]
                task = self._search_platform(scraper_class, query, limit_per_platform)
                tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        for result in results:
            if isinstance(result, list):
                all_courses.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error in platform search: {result}")

        return all_courses

    async def _search_platform(self, scraper_class, query: str, limit: int) -> List[CourseData]:
        """Search a specific platform"""
        try:
            async with scraper_class(self.config) as scraper:
                return await scraper.search_courses(query, limit)
        except Exception as e:
            logger.error(f"Error searching {scraper_class.__name__}: {e}")
            return []

    def convert_to_course_schema(self, course_data: CourseData) -> Course:
        """Convert CourseData to Course schema"""
        # Generate a unique course ID
        course_id = f"{course_data.provider.lower()}_{hash(course_data.url) % 1000000}"

        # Map difficulty
        difficulty_mapping = {
            'beginner': 'beginner',
            'intermediate': 'intermediate',
            'advanced': 'advanced',
            'expert': 'advanced'
        }
        difficulty = difficulty_mapping.get(course_data.difficulty, 'beginner')

        # Estimate duration in weeks (default to 4 if not available)
        duration_weeks = 4
        if course_data.duration:
            # Try to extract weeks from duration string
            import re
            weeks_match = re.search(r'(\d+)\s*weeks?', course_data.duration.lower())
            if weeks_match:
                duration_weeks = int(weeks_match.group(1))

        # Create metadata dictionary
        metadata = {
            'description': course_data.description,
            'instructor': course_data.instructor,
            'price': course_data.price,
            'rating': course_data.rating,
            'enrollment_count': course_data.enrollment_count,
            'category': course_data.category,
            'language': course_data.language,
            'certificate_available': course_data.certificate_available,
            'scraped_at': course_data.scraped_at.isoformat() if course_data.scraped_at else None,
            'original_duration': course_data.duration
        }

        # Remove None values from metadata
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return Course(
            course_id=course_id,
            title=course_data.title,
            skills=course_data.skills or [],
            difficulty=difficulty,
            duration_weeks=duration_weeks,
            provider=course_data.provider,
            url=course_data.url,
            metadata=metadata
        )


# Example usage and testing functions
async def main():
    """Example usage of the web scraping system"""
    config = ScrapingConfig(
        base_delay=2.0,
        max_retries=3,
        rate_limit_per_minute=30,
        respect_robots_txt=True,
        user_agent_rotation=True
    )

    manager = WebScrapingManager(config)

    # Search for Python courses across all platforms
    courses = await manager.search_all_platforms("Python programming", limit_per_platform=10)

    print(f"Found {len(courses)} courses:")
    for course in courses[:5]:  # Show first 5
        print(f"- {course.title} ({course.provider})")
        print(f"  URL: {course.url}")
        print(f"  Skills: {course.skills}")
        print()


if __name__ == "__main__":
    asyncio.run(main())