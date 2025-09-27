import requests
from bs4 import BeautifulSoup
import json
import logging
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, quote
import time
import random
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def _extract_skills_from_text(text: str) -> List[str]:
    """Extract potential skills from course description text using enhanced keyword matching."""
    if not text:
        return []

    # Enhanced skill patterns with more comprehensive coverage
    skill_patterns = [
        # Programming languages
        r'\b(?:python|javascript|java|c\+\+|c#|php|ruby|go|rust|swift|kotlin|typescript|scala|perl|r|matlab|julia)\b',
        # Data science & Analytics
        r'\b(?:machine learning|data science|data analysis|statistics|pandas|numpy|matplotlib|scikit-learn|tensorflow|pytorch|keras|tableau|power bi|excel|spss|sas)\b',
        # Web development
        r'\b(?:html|css|react|angular|vue|node\.?js|express|django|flask|spring|laravel|bootstrap|jquery|sass|less)\b',
        # Mobile development
        r'\b(?:android|ios|flutter|react native|xamarin|swift|kotlin|objective-c)\b',
        # Databases
        r'\b(?:sql|mysql|postgresql|mongodb|redis|elasticsearch|oracle|sqlite|cassandra|dynamodb)\b',
        # Cloud & DevOps
        r'\b(?:aws|azure|gcp|google cloud|docker|kubernetes|jenkins|git|github|gitlab|terraform|ansible|chef|puppet|ci/cd)\b',
        # Hardware & Embedded
        r'\b(?:firmware|embedded|microcontroller|arduino|raspberry pi|fpga|verilog|vhdl|c programming|assembly|rtos|iot|electronics)\b',
        # Cybersecurity
        r'\b(?:cybersecurity|security|penetration testing|ethical hacking|cryptography|network security|malware|vulnerability)\b',
        # Design & Creative
        r'\b(?:ui/ux|user experience|user interface|graphic design|photoshop|illustrator|figma|sketch|adobe|animation)\b',
        # Business & Management
        r'\b(?:project management|agile|scrum|kanban|leadership|communication|business analysis|marketing|sales|finance)\b',
        # AI & ML specific
        r'\b(?:artificial intelligence|neural networks|deep learning|nlp|computer vision|reinforcement learning|chatgpt|openai)\b',
        # Networking
        r'\b(?:networking|cisco|ccna|ccnp|tcp/ip|routing|switching|firewall|vpn)\b',
    ]

    skills = set()
    text_lower = text.lower()

    for pattern in skill_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        skills.update(matches)

    # Clean up and normalize skills
    normalized_skills = []
    for skill in skills:
        # Normalize common variations
        skill = skill.replace('node.js', 'nodejs').replace('node js', 'nodejs')
        skill = skill.replace('c++', 'cpp').replace('c#', 'csharp')
        skill = skill.replace('ui/ux', 'ui ux design')
        normalized_skills.append(skill)

    return list(set(normalized_skills))


def _determine_difficulty(text: str) -> str:
    """Determine course difficulty from text content."""
    if not text:
        return "intermediate"

    text_lower = text.lower()

    if any(word in text_lower for word in ['beginner', 'introduction', 'intro', 'basics', 'fundamentals', 'getting started']):
        return "beginner"
    elif any(word in text_lower for word in ['advanced', 'expert', 'master', 'professional', 'deep dive']):
        return "advanced"
    else:
        return "intermediate"


def _estimate_duration(text: str) -> int:
    """Estimate course duration in weeks from text content."""
    if not text:
        return 4  # Default

    # Look for duration patterns
    duration_patterns = [
        r'(\d+)\s*weeks?',
        r'(\d+)\s*months?',
        r'(\d+)\s*hours?',
    ]

    for pattern in duration_patterns:
        match = re.search(pattern, text.lower())
        if match:
            value = int(match.group(1))
            if 'week' in pattern:
                return value
            elif 'month' in pattern:
                return value * 4  # Convert months to weeks
            elif 'hour' in pattern:
                return max(1, value // 10)  # Rough conversion: 10 hours = 1 week

    return 4  # Default duration


def _calculate_relevance_score(course_title: str, course_description: str, skills: List[str], search_query: str) -> float:
    """Calculate relevance score between course and search query to filter out irrelevant results."""
    if not search_query:
        return 1.0

    score = 0.0
    query_lower = search_query.lower().strip()
    title_lower = course_title.lower()
    desc_lower = course_description.lower() if course_description else ""
    skills_lower = [skill.lower() for skill in skills]

    # Define domain-specific keywords for better matching
    domain_keywords = {
        'firmware': ['firmware', 'embedded', 'microcontroller', 'hardware', 'c programming', 'assembly', 'rtos', 'electronics'],
        'embedded': ['embedded', 'firmware', 'microcontroller', 'hardware', 'c programming', 'assembly', 'rtos', 'iot'],
        'data science': ['data science', 'machine learning', 'statistics', 'python', 'data analysis', 'pandas', 'numpy'],
        'web development': ['web development', 'javascript', 'html', 'css', 'react', 'frontend', 'backend'],
        'mobile development': ['mobile', 'android', 'ios', 'flutter', 'react native', 'app development'],
        'cybersecurity': ['cybersecurity', 'security', 'penetration testing', 'ethical hacking', 'network security'],
        'devops': ['devops', 'docker', 'kubernetes', 'ci/cd', 'jenkins', 'terraform', 'aws', 'cloud'],
        'ai': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks', 'nlp'],
        'blockchain': ['blockchain', 'cryptocurrency', 'smart contracts', 'ethereum', 'bitcoin'],
        'game development': ['game development', 'unity', 'unreal', 'gaming', 'game design'],
    }

    # Extract key terms from search query
    query_words = set(query_lower.split())

    # Check for exact query match in title (highest score)
    if query_lower in title_lower:
        score += 10.0

    # Check for query words in title
    title_words = set(title_lower.split())
    title_matches = len(query_words & title_words)
    score += title_matches * 3.0

    # Check for query words in skills
    skills_text = ' '.join(skills_lower)
    for word in query_words:
        if word in skills_text:
            score += 2.0

    # Check for domain relevance
    for domain, keywords in domain_keywords.items():
        if domain in query_lower or any(keyword in query_lower for keyword in keywords):
            # This is a domain-specific search
            domain_score = 0
            for keyword in keywords:
                if keyword in title_lower or keyword in desc_lower or keyword in skills_text:
                    domain_score += 1.0

            # If it's a domain-specific search but course has no domain keywords, penalize heavily
            if domain_score == 0:
                score -= 5.0
            else:
                score += domain_score
            break

    # Check for description relevance (lower weight)
    if desc_lower:
        desc_words = set(desc_lower.split())
        desc_matches = len(query_words & desc_words)
        score += desc_matches * 0.5

    # Penalize courses that seem completely unrelated
    # Check for conflicting domains
    conflicting_domains = {
        'firmware': ['web development', 'frontend', 'react', 'javascript', 'html', 'css'],
        'web development': ['firmware', 'embedded', 'hardware', 'microcontroller'],
        'data science': ['web development', 'frontend', 'mobile development'],
        'mobile development': ['web development', 'backend', 'server'],
    }

    for domain, conflicts in conflicting_domains.items():
        if domain in query_lower:
            for conflict in conflicts:
                if conflict in title_lower or conflict in skills_text:
                    score -= 3.0

    # Normalize score to 0-1 range
    return max(0.0, min(1.0, score / 10.0))


def _crawl_coursera(query: str, max_courses: int) -> List[Dict[str, Any]]:
    """Crawl Coursera courses for the given query."""
    courses = []
    try:
        logger.info(f"Searching Coursera for: {query}")
        encoded_query = quote(query)
        url = f"https://www.coursera.org/search?query={encoded_query}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Correct selectors based on actual HTML structure
        course_selectors = [
            "div.cds-CommonCard-clickArea",
            "div[class*='cds-CommonCard-clickArea']"
        ]

        course_cards = []
        for selector in course_selectors:
            course_cards = soup.select(selector)
            if course_cards:
                break

        logger.info(f"Found {len(course_cards)} Coursera course cards")

        # Debug: Log the first few cards for inspection
        if course_cards:
            logger.debug(f"First card HTML preview: {str(course_cards[0])[:200]}...")

        for i, card in enumerate(course_cards[:max_courses]):
            try:
                # Extract title from the correct structure
                title_elem = card.select_one("h3.cds-CommonCard-title")
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)
                if not title or len(title) <= 5:
                    continue

                # Extract description from the body content
                desc_elem = card.select_one("div.cds-CommonCard-bodyContent")
                description = desc_elem.get_text(strip=True) if desc_elem else ""

                # Extract URL from the main link
                link_elem = card.select_one("a[href]")
                course_url = None
                if link_elem:
                    href = link_elem.get('href')
                    if href and ('/learn/' in href or '/specializations/' in href or '/professional-certificates/' in href):
                        if href.startswith('http'):
                            course_url = href
                        else:
                            course_url = urljoin("https://www.coursera.org", href)

                # Extract skills and metadata
                full_text = f"{title} {description}"
                skills = _extract_skills_from_text(full_text)
                difficulty = _determine_difficulty(full_text)
                duration = _estimate_duration(full_text)

                # Calculate relevance score to filter out irrelevant courses
                relevance_score = _calculate_relevance_score(title, description, skills, query)

                # Only include courses with sufficient relevance (threshold: 0.3)
                if relevance_score < 0.3:
                    logger.debug(f"Skipping irrelevant Coursera course: {title} (relevance: {relevance_score:.2f})")
                    continue

                course_data = {
                    "title": title,
                    "description": description,
                    "skills": skills,
                    "difficulty": difficulty,
                    "duration_weeks": duration,
                    "provider": "Coursera",
                    "url": course_url,
                    "source": "Coursera",
                    "search_query": query,
                    "relevance_score": relevance_score
                }

                courses.append(course_data)
                logger.debug(f"Extracted Coursera course: {title} (relevance: {relevance_score:.2f})")

            except Exception as e:
                logger.warning(f"Error processing Coursera course card {i}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error crawling Coursera: {e}")

    return courses


def _crawl_udemy(query: str, max_courses: int) -> List[Dict[str, Any]]:
    """Crawl Udemy courses for the given query."""
    courses = []
    try:
        logger.info(f"Searching Udemy for: {query}")
        encoded_query = quote(query)
        url = f"https://www.udemy.com/courses/search/?q={encoded_query}"

        # Enhanced headers to avoid 403 Forbidden
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }

        # Add session for better request handling
        session = requests.Session()
        session.headers.update(headers)

        # Add longer delay before Udemy request
        time.sleep(random.uniform(0.5, 1.0))

        response = session.get(url, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # More specific selectors for Udemy course cards
        course_selectors = [
            "div[data-purpose='course-card-wrapper']",
            "div[data-purpose='course-card']",
            "div.course-card--container--1QM2W",
            "div[class*='course-card--container']",
            "div.course-list--container--FP33M div[data-purpose='course-card']"
        ]

        course_cards = []
        for selector in course_selectors:
            course_cards = soup.select(selector)
            if course_cards:
                break

        logger.info(f"Found {len(course_cards)} Udemy course cards")

        for i, card in enumerate(course_cards[:max_courses]):
            try:
                # Extract title
                title_selectors = ["h3", "h2", ".ud-heading-md", "[data-testid='course-title']"]
                title = None
                for selector in title_selectors:
                    title_elem = card.select_one(selector)
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        break

                if not title:
                    continue

                # Extract description
                desc_selectors = ["p", ".ud-text-sm", "[data-testid='course-headline']"]
                description = ""
                for selector in desc_selectors:
                    desc_elem = card.select_one(selector)
                    if desc_elem:
                        description = desc_elem.get_text(strip=True)
                        break

                # Extract URL with improved selectors
                link_selectors = [
                    "a[href*='/course/']",
                    "a[data-testid='course-title']",
                    "h3 a", "h2 a", "a[href]"
                ]
                course_url = None
                for selector in link_selectors:
                    link_elem = card.select_one(selector)
                    if link_elem:
                        href = link_elem.get('href')
                        if href:
                            if href.startswith('http'):
                                course_url = href
                            else:
                                course_url = urljoin("https://www.udemy.com", href)
                            break

                # Extract skills and metadata
                full_text = f"{title} {description}"
                skills = _extract_skills_from_text(full_text)
                difficulty = _determine_difficulty(full_text)
                duration = _estimate_duration(full_text)

                course_data = {
                    "title": title,
                    "description": description,
                    "skills": skills,
                    "difficulty": difficulty,
                    "duration_weeks": duration,
                    "provider": "Udemy",
                    "url": course_url,
                    "source": "Udemy",
                    "search_query": query
                }

                courses.append(course_data)
                logger.debug(f"Extracted Udemy course: {title}")

            except Exception as e:
                logger.warning(f"Error processing Udemy course card {i}: {e}")
                continue

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            logger.warning(f"Udemy blocked request (403 Forbidden). Skipping Udemy for this search.")
        else:
            logger.error(f"HTTP error crawling Udemy: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error crawling Udemy: {e}")
    except Exception as e:
        logger.error(f"Unexpected error crawling Udemy: {e}")

    return courses


def _crawl_edx(query: str, max_courses: int) -> List[Dict[str, Any]]:
    """Crawl edX courses for the given query."""
    courses = []
    try:
        logger.info(f"Searching edX for: {query}")
        encoded_query = quote(query)
        url = f"https://www.edx.org/search?q={encoded_query}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Enhanced selectors for course cards
        course_selectors = [
            "div[data-testid='discovery-card']",
            "div.discovery-card",
            "div[class*='course-card']",
            "div[class*='CourseCard']",
            "div.course-item",
            "article[class*='course']",
            "div[class*='search-result']",
            "div.course-listing-item",
            "div[data-testid='course-item']"
        ]

        course_cards = []
        for selector in course_selectors:
            course_cards = soup.select(selector)
            if course_cards:
                break

        logger.info(f"Found {len(course_cards)} edX course cards")

        for i, card in enumerate(course_cards[:max_courses]):
            try:
                # Extract title
                title_selectors = ["h3", "h2", "[data-testid='discovery-card-title']"]
                title = None
                for selector in title_selectors:
                    title_elem = card.select_one(selector)
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        break

                if not title:
                    continue

                # Extract description
                desc_selectors = ["p", "[data-testid='discovery-card-description']"]
                description = ""
                for selector in desc_selectors:
                    desc_elem = card.select_one(selector)
                    if desc_elem:
                        description = desc_elem.get_text(strip=True)
                        break

                # Extract URL with improved selectors
                link_selectors = [
                    "a[href*='/course/']",
                    "a[href*='/learn/']",
                    "a[data-testid='discovery-card-title']",
                    "h3 a", "h2 a", "a[href]"
                ]
                course_url = None
                for selector in link_selectors:
                    link_elem = card.select_one(selector)
                    if link_elem:
                        href = link_elem.get('href')
                        if href:
                            if href.startswith('http'):
                                course_url = href
                            else:
                                course_url = urljoin("https://www.edx.org", href)
                            break

                # Extract skills and metadata
                full_text = f"{title} {description}"
                skills = _extract_skills_from_text(full_text)
                difficulty = _determine_difficulty(full_text)
                duration = _estimate_duration(full_text)

                course_data = {
                    "title": title,
                    "description": description,
                    "skills": skills,
                    "difficulty": difficulty,
                    "duration_weeks": duration,
                    "provider": "edX",
                    "url": course_url,
                    "source": "edX",
                    "search_query": query
                }

                courses.append(course_data)
                logger.debug(f"Extracted edX course: {title}")

            except Exception as e:
                logger.warning(f"Error processing edX course card {i}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error crawling edX: {e}")

    return courses


def crawl_courses(query: str = None, max_courses: int = 10) -> List[Dict[str, Any]]:
    """
    Enhanced course crawler that searches for courses based on query and extracts detailed information.
    Now uses concurrent execution for improved performance.

    Args:
        query: Search query for courses (e.g., "python", "data science", "machine learning")
        max_courses: Maximum number of courses to return

    Returns:
        List of course dictionaries with enhanced metadata
    """
    if not query:
        query = "programming"  # Default query

    # Calculate courses per provider
    courses_per_provider = max(max_courses // 3, 2)

    # Use ThreadPoolExecutor for concurrent crawling
    all_courses = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all crawling tasks concurrently
        future_to_provider = {
            executor.submit(_crawl_coursera, query, courses_per_provider): "Coursera",
            executor.submit(_crawl_udemy, query, courses_per_provider): "Udemy",
            executor.submit(_crawl_edx, query, courses_per_provider): "edX"
        }

        # Collect results as they complete
        for future in as_completed(future_to_provider):
            provider = future_to_provider[future]
            try:
                provider_courses = future.result()
                all_courses.extend(provider_courses)
                logger.debug(f"Completed crawling {provider}: {len(provider_courses)} courses")
            except Exception as e:
                logger.error(f"Error in {provider} crawling thread: {e}")
                continue

    # Filter and return only quality courses (no fallback generation)
    quality_courses = [c for c in all_courses if c.get('title') and len(c.get('title', '')) > 5]

    logger.info(f"Total courses crawled: {len(quality_courses)}")
    return quality_courses[:max_courses]


def _get_fallback_courses(query: str, max_courses: int) -> List[Dict[str, Any]]:
    """
    Generate fallback courses when scraping fails or returns insufficient results.
    This creates realistic course suggestions based on the query.
    """
    fallback_courses = []

    # Define course templates based on common queries
    course_templates = {
        'data science': [
            {
                'title': 'Complete Data Science Bootcamp',
                'skills': ['python', 'data analysis', 'machine learning', 'statistics', 'pandas', 'numpy'],
                'difficulty': 'intermediate',
                'duration_weeks': 12,
                'provider': 'DataCamp',
                'description': 'Comprehensive data science course covering Python, statistics, and machine learning'
            },
            {
                'title': 'Python for Data Science and Machine Learning',
                'skills': ['python', 'machine learning', 'data science', 'scikit-learn', 'tensorflow'],
                'difficulty': 'beginner',
                'duration_weeks': 8,
                'provider': 'Udemy',
                'description': 'Learn Python programming for data science and machine learning applications'
            },
            {
                'title': 'Advanced Data Analysis with R',
                'skills': ['r programming', 'data analysis', 'statistics', 'data visualization'],
                'difficulty': 'advanced',
                'duration_weeks': 6,
                'provider': 'Coursera',
                'description': 'Advanced statistical analysis and data visualization using R'
            }
        ],
        'machine learning': [
            {
                'title': 'Machine Learning A-Z',
                'skills': ['machine learning', 'python', 'algorithms', 'supervised learning', 'unsupervised learning'],
                'difficulty': 'intermediate',
                'duration_weeks': 10,
                'provider': 'Udemy',
                'description': 'Complete machine learning course covering all major algorithms and techniques'
            },
            {
                'title': 'Deep Learning Specialization',
                'skills': ['deep learning', 'neural networks', 'tensorflow', 'keras', 'python'],
                'difficulty': 'advanced',
                'duration_weeks': 16,
                'provider': 'Coursera',
                'description': 'Comprehensive deep learning course with hands-on projects'
            }
        ],
        'web development': [
            {
                'title': 'Full Stack Web Development',
                'skills': ['javascript', 'react', 'node.js', 'html', 'css', 'mongodb'],
                'difficulty': 'intermediate',
                'duration_weeks': 14,
                'provider': 'freeCodeCamp',
                'description': 'Complete full stack web development course'
            },
            {
                'title': 'Modern JavaScript Development',
                'skills': ['javascript', 'es6', 'react', 'vue', 'typescript'],
                'difficulty': 'intermediate',
                'duration_weeks': 8,
                'provider': 'Udemy',
                'description': 'Modern JavaScript development with latest frameworks'
            }
        ],
        'python': [
            {
                'title': 'Python Programming Masterclass',
                'skills': ['python', 'programming', 'object-oriented programming', 'data structures'],
                'difficulty': 'beginner',
                'duration_weeks': 6,
                'provider': 'Udemy',
                'description': 'Complete Python programming course from basics to advanced'
            },
            {
                'title': 'Advanced Python Programming',
                'skills': ['python', 'advanced programming', 'decorators', 'generators', 'async programming'],
                'difficulty': 'advanced',
                'duration_weeks': 4,
                'provider': 'Pluralsight',
                'description': 'Advanced Python concepts and best practices'
            }
        ],
        'public speaking': [
            {
                'title': 'Complete Public Speaking Masterclass',
                'skills': ['public speaking', 'communication', 'presentation skills', 'confidence building', 'storytelling'],
                'difficulty': 'beginner',
                'duration_weeks': 8,
                'provider': 'Udemy',
                'description': 'Master the art of public speaking with confidence and charisma'
            },
            {
                'title': 'Advanced Presentation Skills',
                'skills': ['presentation skills', 'public speaking', 'visual communication', 'audience engagement'],
                'difficulty': 'intermediate',
                'duration_weeks': 6,
                'provider': 'Coursera',
                'description': 'Advanced techniques for engaging presentations and public speaking'
            },
            {
                'title': 'Professional Communication and Speaking',
                'skills': ['professional communication', 'public speaking', 'business communication', 'leadership communication'],
                'difficulty': 'intermediate',
                'duration_weeks': 10,
                'provider': 'LinkedIn Learning',
                'description': 'Develop professional communication and public speaking skills for career advancement'
            }
        ],
        'communication': [
            {
                'title': 'Effective Communication Skills',
                'skills': ['communication', 'interpersonal skills', 'active listening', 'verbal communication'],
                'difficulty': 'beginner',
                'duration_weeks': 4,
                'provider': 'Coursera',
                'description': 'Build strong communication skills for personal and professional success'
            },
            {
                'title': 'Business Communication Essentials',
                'skills': ['business communication', 'professional writing', 'presentation skills', 'meeting facilitation'],
                'difficulty': 'intermediate',
                'duration_weeks': 6,
                'provider': 'edX',
                'description': 'Essential business communication skills for the modern workplace'
            },
            {
                'title': 'Storytelling for Impact',
                'skills': ['storytelling', 'narrative techniques', 'persuasive communication', 'content creation'],
                'difficulty': 'intermediate',
                'duration_weeks': 5,
                'provider': 'MasterClass',
                'description': 'Learn to craft compelling stories that engage and persuade audiences'
            }
        ],
        'leadership': [
            {
                'title': 'Leadership Fundamentals',
                'skills': ['leadership', 'team management', 'decision making', 'emotional intelligence'],
                'difficulty': 'intermediate',
                'duration_weeks': 8,
                'provider': 'Coursera',
                'description': 'Develop essential leadership skills for managing teams and driving results'
            },
            {
                'title': 'Executive Communication',
                'skills': ['executive communication', 'strategic communication', 'influence', 'public speaking'],
                'difficulty': 'advanced',
                'duration_weeks': 6,
                'provider': 'Harvard Business School Online',
                'description': 'Advanced communication strategies for senior leaders and executives'
            }
        ]
    }

    # Find matching templates with improved matching logic
    query_lower = query.lower() if query else ''
    matching_templates = []

    # Define related terms for better matching
    related_terms = {
        'public speaking': ['speaker', 'presentation', 'communication', 'speaking', 'public speaker'],
        'communication': ['speaker', 'speaking', 'presentation', 'interpersonal', 'verbal'],
        'leadership': ['leader', 'management', 'executive', 'team lead'],
        'data science': ['data scientist', 'analytics', 'data analysis'],
        'machine learning': ['ml', 'ai', 'artificial intelligence'],
        'web development': ['web developer', 'frontend', 'backend', 'fullstack'],
        'python': ['programming', 'coding']
    }

    # First, try exact and related term matching
    for key, templates in course_templates.items():
        # Check exact key match
        if key in query_lower:
            matching_templates.extend(templates)
            continue

        # Check related terms
        if key in related_terms:
            for term in related_terms[key]:
                if term in query_lower:
                    matching_templates.extend(templates)
                    break

        # Check if any word in the key matches
        if any(word in query_lower for word in key.split()):
            matching_templates.extend(templates)

    # If no specific matches, use general programming courses
    if not matching_templates:
        matching_templates = [
            {
                'title': 'Introduction to Programming',
                'skills': ['programming', 'problem solving', 'algorithms'],
                'difficulty': 'beginner',
                'duration_weeks': 6,
                'provider': 'edX',
                'description': 'Fundamental programming concepts and problem-solving skills'
            },
            {
                'title': 'Software Development Fundamentals',
                'skills': ['software development', 'programming', 'version control', 'testing'],
                'difficulty': 'intermediate',
                'duration_weeks': 8,
                'provider': 'Coursera',
                'description': 'Essential software development skills and practices'
            }
        ]

    # Create fallback courses
    for i, template in enumerate(matching_templates[:max_courses]):
        course_id = f"fallback-{abs(hash(template['title'])) % 10000:04d}"

        fallback_course = {
            'title': template['title'],
            'description': template['description'],
            'skills': template['skills'],
            'difficulty': template['difficulty'],
            'duration_weeks': template['duration_weeks'],
            'provider': template['provider'],
            'url': None,
            'source': 'Fallback',
            'search_query': query
        }

        fallback_courses.append(fallback_course)

    logger.info(f"Generated {len(fallback_courses)} fallback courses for query: {query}")
    return fallback_courses