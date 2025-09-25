import requests
from bs4 import BeautifulSoup
import json
import logging
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, quote
import time
import random

logger = logging.getLogger(__name__)


def _extract_skills_from_text(text: str) -> List[str]:
    """Extract potential skills from course description text using keyword matching."""
    if not text:
        return []

    # Common skill keywords to look for
    skill_patterns = [
        # Programming languages
        r'\b(?:python|javascript|java|c\+\+|c#|php|ruby|go|rust|swift|kotlin|typescript)\b',
        # Data science
        r'\b(?:machine learning|data science|data analysis|statistics|pandas|numpy|matplotlib|scikit-learn|tensorflow|pytorch|keras)\b',
        # Web development
        r'\b(?:html|css|react|angular|vue|node\.?js|express|django|flask|spring|laravel)\b',
        # Databases
        r'\b(?:sql|mysql|postgresql|mongodb|redis|elasticsearch)\b',
        # Cloud & DevOps
        r'\b(?:aws|azure|gcp|docker|kubernetes|jenkins|git|github|gitlab)\b',
        # General skills
        r'\b(?:project management|communication|leadership|problem solving|critical thinking)\b',
    ]

    skills = set()
    text_lower = text.lower()

    for pattern in skill_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        skills.update(matches)

    return list(skills)


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


def crawl_courses(query: str = None, max_courses: int = 10) -> List[Dict[str, Any]]:
    """
    Enhanced course crawler that searches for courses based on query and extracts detailed information.

    Args:
        query: Search query for courses (e.g., "python", "data science", "machine learning")
        max_courses: Maximum number of courses to return

    Returns:
        List of course dictionaries with enhanced metadata
    """
    courses = []

    if not query:
        query = "programming"  # Default query

    # Add random delay to avoid being blocked
    def random_delay():
        time.sleep(random.uniform(0.5, 1.5))

    # Enhanced Coursera scraping
    try:
        logger.info(f"Searching Coursera for: {query}")
        encoded_query = quote(query)
        url = f"https://www.coursera.org/search?query={encoded_query}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Try multiple selectors for course cards
        course_selectors = [
            "div[data-testid='search-result-card']",
            "div.cds-9",
            "div.rc-SearchCard",
            "div[class*='search-result']"
        ]

        course_cards = []
        for selector in course_selectors:
            course_cards = soup.select(selector)
            if course_cards:
                break

        logger.info(f"Found {len(course_cards)} Coursera course cards")

        for i, card in enumerate(course_cards[:max_courses//2]):
            try:
                # Extract title
                title_selectors = ["h2", "h3", "[data-testid='search-result-title']", ".cds-119"]
                title = None
                for selector in title_selectors:
                    title_elem = card.select_one(selector)
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        break

                if not title:
                    continue

                # Extract description
                desc_selectors = ["p", ".cds-119", "[data-testid='search-result-description']"]
                description = ""
                for selector in desc_selectors:
                    desc_elem = card.select_one(selector)
                    if desc_elem:
                        description = desc_elem.get_text(strip=True)
                        break

                # Extract URL
                link_elem = card.select_one("a[href]")
                course_url = None
                if link_elem:
                    href = link_elem.get('href')
                    if href:
                        course_url = urljoin("https://www.coursera.org", href)

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
                    "provider": "Coursera",
                    "url": course_url,
                    "source": "Coursera",
                    "search_query": query
                }

                courses.append(course_data)
                logger.debug(f"Extracted Coursera course: {title}")

            except Exception as e:
                logger.warning(f"Error processing Coursera course card {i}: {e}")
                continue

        random_delay()

    except Exception as e:
        logger.error(f"Error crawling Coursera: {e}")

    # Enhanced Udemy scraping
    try:
        logger.info(f"Searching Udemy for: {query}")
        encoded_query = quote(query)
        url = f"https://www.udemy.com/courses/search/?q={encoded_query}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Try multiple selectors for course cards
        course_selectors = [
            "div[data-testid='course-card']",
            "div.course-card--main-content--3xT-S",
            "div[class*='course-card']",
            "div.ud-search-form-autocomplete-suggestion-item"
        ]

        course_cards = []
        for selector in course_selectors:
            course_cards = soup.select(selector)
            if course_cards:
                break

        logger.info(f"Found {len(course_cards)} Udemy course cards")

        for i, card in enumerate(course_cards[:max_courses//2]):
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

                # Extract URL
                link_elem = card.select_one("a[href]")
                course_url = None
                if link_elem:
                    href = link_elem.get('href')
                    if href:
                        course_url = urljoin("https://www.udemy.com", href)

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

        random_delay()

    except Exception as e:
        logger.error(f"Error crawling Udemy: {e}")

    # Add edX scraping
    try:
        logger.info(f"Searching edX for: {query}")
        encoded_query = quote(query)
        url = f"https://www.edx.org/search?q={encoded_query}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Try multiple selectors for course cards
        course_selectors = [
            "div[data-testid='discovery-card']",
            "div.discovery-card",
            "div[class*='course-card']"
        ]

        course_cards = []
        for selector in course_selectors:
            course_cards = soup.select(selector)
            if course_cards:
                break

        logger.info(f"Found {len(course_cards)} edX course cards")

        for i, card in enumerate(course_cards[:max_courses//3]):
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

                # Extract URL
                link_elem = card.select_one("a[href]")
                course_url = None
                if link_elem:
                    href = link_elem.get('href')
                    if href:
                        course_url = urljoin("https://www.edx.org", href)

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

    # If we didn't get enough quality courses from scraping, add fallback courses
    quality_courses = [c for c in courses if c.get('title') and len(c.get('title', '')) > 5 and c.get('skills')]

    if len(quality_courses) < max_courses // 2:
        fallback_courses = _get_fallback_courses(query, max_courses - len(quality_courses))
        courses = quality_courses + fallback_courses

    logger.info(f"Total courses crawled: {len(courses)}")
    return courses[:max_courses]


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