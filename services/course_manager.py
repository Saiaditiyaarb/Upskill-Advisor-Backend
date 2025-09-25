"""
Course Management and Dynamic Update System

This module provides comprehensive course data management including:
- Dynamic courses.json updates
- Search and filtering capabilities
- Data persistence and backup
- Incremental updates and synchronization
- Course recommendation and matching
- Analytics and reporting

Features:
- Thread-safe JSON file operations
- Backup and recovery mechanisms
- Search indexing for fast queries
- Duplicate prevention and merging
- Scheduled updates and maintenance
- RESTful API integration
"""

import json
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from pathlib import Path
from dataclasses import asdict
import shutil
import hashlib
from collections import defaultdict
import re

from schemas.course import Course
from services.web_scraper import WebScrapingManager, ScrapingConfig, CourseData
from services.course_data_processor import CourseDataProcessor, ProcessingStats

logger = logging.getLogger(__name__)


class CourseIndex:
    """In-memory search index for fast course queries"""

    def __init__(self):
        self.title_index: Dict[str, Set[str]] = defaultdict(set)
        self.skill_index: Dict[str, Set[str]] = defaultdict(set)
        self.provider_index: Dict[str, Set[str]] = defaultdict(set)
        self.category_index: Dict[str, Set[str]] = defaultdict(set)
        self.difficulty_index: Dict[str, Set[str]] = defaultdict(set)
        self.course_data: Dict[str, Course] = {}

    def add_course(self, course: Course):
        """Add a course to the search index"""
        course_id = course.course_id
        self.course_data[course_id] = course

        # Index title words
        title_words = self._tokenize(course.title)
        for word in title_words:
            self.title_index[word].add(course_id)

        # Index skills
        for skill in course.skills:
            skill_words = self._tokenize(skill)
            for word in skill_words:
                self.skill_index[word].add(course_id)

        # Index provider
        if course.provider:
            provider_words = self._tokenize(course.provider)
            for word in provider_words:
                self.provider_index[word].add(course_id)

        # Index category
        category = course.metadata.get('category')
        if category:
            category_words = self._tokenize(category)
            for word in category_words:
                self.category_index[word].add(course_id)

        # Index difficulty
        self.difficulty_index[course.difficulty].add(course_id)

    def remove_course(self, course_id: str):
        """Remove a course from the search index"""
        if course_id not in self.course_data:
            return

        course = self.course_data[course_id]

        # Remove from title index
        title_words = self._tokenize(course.title)
        for word in title_words:
            self.title_index[word].discard(course_id)

        # Remove from skill index
        for skill in course.skills:
            skill_words = self._tokenize(skill)
            for word in skill_words:
                self.skill_index[word].discard(course_id)

        # Remove from provider index
        if course.provider:
            provider_words = self._tokenize(course.provider)
            for word in provider_words:
                self.provider_index[word].discard(course_id)

        # Remove from category index
        category = course.metadata.get('category')
        if category:
            category_words = self._tokenize(category)
            for word in category_words:
                self.category_index[word].discard(course_id)

        # Remove from difficulty index
        self.difficulty_index[course.difficulty].discard(course_id)

        # Remove from course data
        del self.course_data[course_id]

    def search(self, query: str, filters: Dict[str, Any] = None) -> List[Course]:
        """Search courses using the index"""
        if not query and not filters:
            return list(self.course_data.values())

        matching_ids = set()

        if query:
            query_words = self._tokenize(query)
            query_matches = set()

            for word in query_words:
                # Search in titles
                title_matches = self.title_index.get(word, set())
                # Search in skills
                skill_matches = self.skill_index.get(word, set())
                # Search in providers
                provider_matches = self.provider_index.get(word, set())
                # Search in categories
                category_matches = self.category_index.get(word, set())

                word_matches = title_matches | skill_matches | provider_matches | category_matches
                if not query_matches:
                    query_matches = word_matches
                else:
                    query_matches &= word_matches  # AND operation for multiple words

            matching_ids = query_matches

        # Apply filters
        if filters:
            filtered_ids = set(self.course_data.keys())

            if 'provider' in filters:
                provider_filter = set()
                for provider in filters['provider'] if isinstance(filters['provider'], list) else [filters['provider']]:
                    provider_words = self._tokenize(provider)
                    for word in provider_words:
                        provider_filter |= self.provider_index.get(word, set())
                filtered_ids &= provider_filter

            if 'difficulty' in filters:
                difficulty_filter = set()
                difficulties = filters['difficulty'] if isinstance(filters['difficulty'], list) else [filters['difficulty']]
                for difficulty in difficulties:
                    difficulty_filter |= self.difficulty_index.get(difficulty, set())
                filtered_ids &= difficulty_filter

            if 'skills' in filters:
                skill_filter = set()
                skills = filters['skills'] if isinstance(filters['skills'], list) else [filters['skills']]
                for skill in skills:
                    skill_words = self._tokenize(skill)
                    for word in skill_words:
                        skill_filter |= self.skill_index.get(word, set())
                filtered_ids &= skill_filter

            if 'category' in filters:
                category_filter = set()
                categories = filters['category'] if isinstance(filters['category'], list) else [filters['category']]
                for category in categories:
                    category_words = self._tokenize(category)
                    for word in category_words:
                        category_filter |= self.category_index.get(word, set())
                filtered_ids &= category_filter

            if matching_ids:
                matching_ids &= filtered_ids
            else:
                matching_ids = filtered_ids

        # Return matching courses
        return [self.course_data[course_id] for course_id in matching_ids if course_id in self.course_data]

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for indexing"""
        if not text:
            return []
        # Convert to lowercase and split on non-alphanumeric characters
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if len(word) > 2]  # Filter out very short words

    def rebuild_index(self, courses: List[Course]):
        """Rebuild the entire index from a list of courses"""
        self.clear()
        for course in courses:
            self.add_course(course)

    def clear(self):
        """Clear all indexes"""
        self.title_index.clear()
        self.skill_index.clear()
        self.provider_index.clear()
        self.category_index.clear()
        self.difficulty_index.clear()
        self.course_data.clear()


class CourseManager:
    """Main course management system"""

    def __init__(self, courses_file_path: str = "courses.json", backup_dir: str = "backups"):
        self.courses_file_path = Path(courses_file_path)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)

        self.index = CourseIndex()
        self.scraping_manager = WebScrapingManager()
        self.data_processor = CourseDataProcessor()

        # Thread safety
        self._lock = threading.RLock()

        # Load existing courses
        self.load_courses()

    def load_courses(self) -> List[Course]:
        """Load courses from JSON file"""
        with self._lock:
            try:
                if self.courses_file_path.exists():
                    with open(self.courses_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    courses = []
                    if isinstance(data, list):
                        # Handle list format
                        for course_data in data:
                            try:
                                course = Course(**course_data)
                                courses.append(course)
                            except Exception as e:
                                logger.error(f"Error loading course: {e}")
                    elif isinstance(data, dict) and 'courses' in data:
                        # Handle object format with metadata
                        for course_data in data['courses']:
                            try:
                                course = Course(**course_data)
                                courses.append(course)
                            except Exception as e:
                                logger.error(f"Error loading course: {e}")

                    # Rebuild search index
                    self.index.rebuild_index(courses)
                    logger.info(f"Loaded {len(courses)} courses from {self.courses_file_path}")
                    return courses
                else:
                    logger.info(f"No existing courses file found at {self.courses_file_path}")
                    return []
            except Exception as e:
                logger.error(f"Error loading courses: {e}")
                return []

    def save_courses(self, courses: List[Course] = None) -> bool:
        """Save courses to JSON file with backup"""
        with self._lock:
            try:
                if courses is None:
                    courses = list(self.index.course_data.values())

                # Create backup before saving
                self._create_backup()

                # Prepare data structure
                courses_data = []
                for course in courses:
                    course_dict = asdict(course)
                    courses_data.append(course_dict)

                # Create comprehensive data structure
                data = {
                    'metadata': {
                        'total_courses': len(courses),
                        'last_updated': datetime.now().isoformat(),
                        'version': '1.0',
                        'providers': list(set(course.provider for course in courses if course.provider)),
                        'categories': list(set(course.metadata.get('category') for course in courses if course.metadata.get('category'))),
                        'difficulties': list(set(course.difficulty for course in courses)),
                        'total_skills': len(set(skill for course in courses for skill in course.skills))
                    },
                    'courses': courses_data
                }

                # Write to file
                with open(self.courses_file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                logger.info(f"Saved {len(courses)} courses to {self.courses_file_path}")
                return True

            except Exception as e:
                logger.error(f"Error saving courses: {e}")
                return False

    def _create_backup(self):
        """Create a backup of the current courses file"""
        try:
            if self.courses_file_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = self.backup_dir / f"courses_backup_{timestamp}.json"
                shutil.copy2(self.courses_file_path, backup_file)

                # Keep only last 10 backups
                backups = sorted(self.backup_dir.glob("courses_backup_*.json"))
                if len(backups) > 10:
                    for old_backup in backups[:-10]:
                        old_backup.unlink()

                logger.debug(f"Created backup: {backup_file}")
        except Exception as e:
            logger.error(f"Error creating backup: {e}")

    async def update_courses(self, search_queries: List[str] = None, platforms: List[str] = None,
                           limit_per_query: int = 50) -> Dict[str, Any]:
        """Update courses by scraping new data"""
        if search_queries is None:
            search_queries = [
                "python programming", "data science", "machine learning", "web development",
                "javascript", "react", "artificial intelligence", "cybersecurity",
                "digital marketing", "project management", "ui ux design", "blockchain"
            ]

        update_stats = {
            'start_time': datetime.now().isoformat(),
            'queries_processed': 0,
            'new_courses_found': 0,
            'courses_updated': 0,
            'duplicates_removed': 0,
            'errors': []
        }

        try:
            all_new_courses = []

            # Scrape courses for each query
            for query in search_queries:
                try:
                    logger.info(f"Scraping courses for query: {query}")
                    scraped_courses = await self.scraping_manager.search_all_platforms(
                        query, platforms, limit_per_query // len(search_queries) if len(search_queries) > 1 else limit_per_query
                    )
                    all_new_courses.extend(scraped_courses)
                    update_stats['queries_processed'] += 1

                    # Add delay between queries to be respectful
                    await asyncio.sleep(2)

                except Exception as e:
                    error_msg = f"Error scraping query '{query}': {e}"
                    logger.error(error_msg)
                    update_stats['errors'].append(error_msg)

            # Process and standardize the scraped data
            if all_new_courses:
                processed_courses, processing_stats = await self.data_processor.process_courses(all_new_courses)
                update_stats['duplicates_removed'] = processing_stats.duplicates_removed

                # Convert to Course schema and add to index
                existing_urls = {course.url for course in self.index.course_data.values()}
                new_courses_added = 0
                courses_updated = 0

                for course_data in processed_courses:
                    try:
                        course = self.data_processor.convert_to_course_schema(course_data)

                        if course.url in existing_urls:
                            # Update existing course
                            existing_course = self._find_course_by_url(course.url)
                            if existing_course:
                                self._update_existing_course(existing_course, course)
                                courses_updated += 1
                        else:
                            # Add new course
                            self.index.add_course(course)
                            new_courses_added += 1
                            existing_urls.add(course.url)

                    except Exception as e:
                        error_msg = f"Error processing course {course_data.title}: {e}"
                        logger.error(error_msg)
                        update_stats['errors'].append(error_msg)

                update_stats['new_courses_found'] = new_courses_added
                update_stats['courses_updated'] = courses_updated

                # Save updated courses
                if new_courses_added > 0 or courses_updated > 0:
                    self.save_courses()
                    logger.info(f"Update complete: {new_courses_added} new courses, {courses_updated} updated")

        except Exception as e:
            error_msg = f"Error during course update: {e}"
            logger.error(error_msg)
            update_stats['errors'].append(error_msg)

        update_stats['end_time'] = datetime.now().isoformat()
        return update_stats

    def _find_course_by_url(self, url: str) -> Optional[Course]:
        """Find a course by URL"""
        for course in self.index.course_data.values():
            if course.url == url:
                return course
        return None

    def _update_existing_course(self, existing_course: Course, new_course: Course):
        """Update an existing course with new information"""
        # Update metadata with new information
        existing_course.metadata.update({
            'last_updated': datetime.now().isoformat(),
            'update_count': existing_course.metadata.get('update_count', 0) + 1
        })

        # Merge skills (add new ones)
        existing_skills = set(existing_course.skills)
        new_skills = set(new_course.skills)
        merged_skills = list(existing_skills | new_skills)
        existing_course.skills = merged_skills

        # Update other fields if they were empty or the new data is more recent
        if not existing_course.metadata.get('description') and new_course.metadata.get('description'):
            existing_course.metadata['description'] = new_course.metadata['description']

        if not existing_course.metadata.get('instructor') and new_course.metadata.get('instructor'):
            existing_course.metadata['instructor'] = new_course.metadata['instructor']

        # Update the course in the index
        self.index.remove_course(existing_course.course_id)
        self.index.add_course(existing_course)

    def search_courses(self, query: str = "", filters: Dict[str, Any] = None,
                      limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """Search courses with pagination"""
        try:
            # Perform search
            results = self.index.search(query, filters)

            # Apply sorting (by relevance, then by rating, then by title)
            results.sort(key=lambda x: (
                -(x.metadata.get('rating', 0) or 0),  # Higher rating first
                x.title.lower()  # Alphabetical as tiebreaker
            ))

            # Apply pagination
            total_results = len(results)
            paginated_results = results[offset:offset + limit]

            return {
                'courses': [asdict(course) for course in paginated_results],
                'pagination': {
                    'total': total_results,
                    'limit': limit,
                    'offset': offset,
                    'has_next': offset + limit < total_results,
                    'has_prev': offset > 0
                },
                'filters_applied': filters or {},
                'query': query
            }

        except Exception as e:
            logger.error(f"Error searching courses: {e}")
            return {
                'courses': [],
                'pagination': {'total': 0, 'limit': limit, 'offset': offset, 'has_next': False, 'has_prev': False},
                'error': str(e)
            }

    def get_course_by_id(self, course_id: str) -> Optional[Course]:
        """Get a specific course by ID"""
        return self.index.course_data.get(course_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the course database"""
        courses = list(self.index.course_data.values())

        if not courses:
            return {'total_courses': 0}

        # Provider statistics
        provider_counts = defaultdict(int)
        for course in courses:
            if course.provider:
                provider_counts[course.provider] += 1

        # Category statistics
        category_counts = defaultdict(int)
        for course in courses:
            category = course.metadata.get('category')
            if category:
                category_counts[category] += 1

        # Difficulty statistics
        difficulty_counts = defaultdict(int)
        for course in courses:
            difficulty_counts[course.difficulty] += 1

        # Skill statistics
        skill_counts = defaultdict(int)
        for course in courses:
            for skill in course.skills:
                skill_counts[skill] += 1

        # Free vs paid courses
        free_courses = sum(1 for course in courses if course.metadata.get('price_info', {}).get('is_free', False))
        paid_courses = len(courses) - free_courses

        return {
            'total_courses': len(courses),
            'providers': dict(provider_counts),
            'categories': dict(category_counts),
            'difficulties': dict(difficulty_counts),
            'top_skills': dict(sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
            'pricing': {
                'free_courses': free_courses,
                'paid_courses': paid_courses
            },
            'last_updated': max(
                (datetime.fromisoformat(course.metadata.get('last_updated', '1970-01-01T00:00:00'))
                 for course in courses if course.metadata.get('last_updated')),
                default=datetime.min
            ).isoformat()
        }

    def recommend_courses(self, user_skills: List[str], user_interests: List[str] = None,
                         difficulty_preference: str = None, limit: int = 10) -> List[Course]:
        """Recommend courses based on user profile"""
        try:
            all_courses = list(self.index.course_data.values())
            scored_courses = []

            for course in all_courses:
                score = 0

                # Skill matching
                course_skills = set(skill.lower() for skill in course.skills)
                user_skills_lower = set(skill.lower() for skill in user_skills)
                skill_overlap = len(course_skills & user_skills_lower)
                score += skill_overlap * 10

                # Interest matching (if provided)
                if user_interests:
                    course_text = (course.title + " " + course.metadata.get('description', '')).lower()
                    for interest in user_interests:
                        if interest.lower() in course_text:
                            score += 5

                # Difficulty preference
                if difficulty_preference and course.difficulty == difficulty_preference:
                    score += 3

                # Rating boost
                rating = course.metadata.get('rating')
                if rating:
                    score += rating

                # Free course boost
                if course.metadata.get('price_info', {}).get('is_free', False):
                    score += 2

                scored_courses.append((course, score))

            # Sort by score and return top recommendations
            scored_courses.sort(key=lambda x: x[1], reverse=True)
            return [course for course, score in scored_courses[:limit] if score > 0]

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []

    async def scheduled_update(self, interval_hours: int = 24):
        """Run scheduled updates"""
        while True:
            try:
                logger.info("Starting scheduled course update")
                update_stats = await self.update_courses()
                logger.info(f"Scheduled update completed: {update_stats}")

                # Wait for next update
                await asyncio.sleep(interval_hours * 3600)

            except Exception as e:
                logger.error(f"Error in scheduled update: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying


# Example usage and testing
async def main():
    """Example usage of the course manager"""
    manager = CourseManager("courses.json")

    # Update courses with new data
    print("Updating courses...")
    update_stats = await manager.update_courses(
        search_queries=["python", "data science"],
        limit_per_query=10
    )
    print(f"Update stats: {update_stats}")

    # Search for courses
    print("\nSearching for Python courses...")
    search_results = manager.search_courses("python", limit=5)
    print(f"Found {search_results['pagination']['total']} courses")

    # Get statistics
    print("\nDatabase statistics:")
    stats = manager.get_statistics()
    print(f"Total courses: {stats['total_courses']}")
    print(f"Providers: {list(stats['providers'].keys())}")

    # Get recommendations
    print("\nRecommendations for Python developer:")
    recommendations = manager.recommend_courses(
        user_skills=["python", "programming"],
        difficulty_preference="intermediate",
        limit=3
    )
    for course in recommendations:
        print(f"- {course.title} ({course.provider})")


if __name__ == "__main__":
    asyncio.run(main())