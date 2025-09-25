"""
Course Data Processing and Standardization Module

This module provides comprehensive data processing, validation, and standardization
for course data collected from various educational platforms. It ensures data quality,
consistency, and enrichment before storing in the courses.json file.

Features:
- Data validation and cleaning
- Skill extraction and standardization
- Duration normalization
- Price standardization
- Difficulty level mapping
- Category classification
- Duplicate detection and merging
- Data enrichment with external APIs
"""

import re
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from collections import Counter
import asyncio

from services.web_scraper import CourseData, Course

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for data processing operations"""
    total_courses: int = 0
    valid_courses: int = 0
    duplicates_removed: int = 0
    skills_extracted: int = 0
    categories_assigned: int = 0
    prices_normalized: int = 0
    durations_normalized: int = 0


class SkillExtractor:
    """Extract and standardize skills from course titles and descriptions"""

    def __init__(self):
        # Common programming languages and technologies
        self.programming_languages = {
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby',
            'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'html',
            'css', 'sass', 'less', 'dart', 'perl', 'lua', 'bash', 'powershell'
        }

        # Frameworks and libraries
        self.frameworks = {
            'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express',
            'laravel', 'rails', 'asp.net', 'node.js', 'jquery', 'bootstrap',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
            'matplotlib', 'seaborn', 'opencv', 'unity', 'unreal', 'xamarin'
        }

        # Tools and platforms
        self.tools = {
            'git', 'docker', 'kubernetes', 'jenkins', 'aws', 'azure', 'gcp',
            'firebase', 'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch',
            'tableau', 'power bi', 'excel', 'photoshop', 'illustrator', 'figma',
            'sketch', 'blender', 'autocad', 'solidworks'
        }

        # Soft skills and concepts
        self.concepts = {
            'machine learning', 'deep learning', 'artificial intelligence', 'data science',
            'data analysis', 'web development', 'mobile development', 'game development',
            'cybersecurity', 'blockchain', 'cloud computing', 'devops', 'agile',
            'scrum', 'project management', 'digital marketing', 'seo', 'ui/ux',
            'graphic design', 'photography', 'video editing', 'animation',
            'business analysis', 'finance', 'accounting', 'leadership'
        }

        # Combine all skill categories
        self.all_skills = (
            self.programming_languages |
            self.frameworks |
            self.tools |
            self.concepts
        )

    def extract_skills(self, title: str, description: str = None) -> List[str]:
        """Extract skills from course title and description"""
        text = title.lower()
        if description:
            text += " " + description.lower()

        # Remove special characters and normalize
        text = re.sub(r'[^\w\s\-\+\#\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        extracted_skills = set()

        # Direct skill matching
        for skill in self.all_skills:
            if skill in text:
                extracted_skills.add(skill)

        # Pattern-based extraction for common variations
        patterns = {
            r'\b(react|reactjs|react\.js)\b': 'react',
            r'\b(node|nodejs|node\.js)\b': 'node.js',
            r'\b(vue|vuejs|vue\.js)\b': 'vue',
            r'\b(angular|angularjs)\b': 'angular',
            r'\b(c\+\+|cpp)\b': 'c++',
            r'\b(c#|csharp)\b': 'c#',
            r'\b(js|javascript)\b': 'javascript',
            r'\b(ts|typescript)\b': 'typescript',
            r'\b(py|python)\b': 'python',
            r'\b(ml|machine learning)\b': 'machine learning',
            r'\b(ai|artificial intelligence)\b': 'artificial intelligence',
            r'\b(dl|deep learning)\b': 'deep learning',
            r'\b(ds|data science)\b': 'data science',
            r'\b(da|data analysis)\b': 'data analysis',
            r'\b(ui/ux|ux/ui|user experience|user interface)\b': 'ui/ux',
            r'\b(seo|search engine optimization)\b': 'seo',
            r'\b(aws|amazon web services)\b': 'aws',
            r'\b(gcp|google cloud platform)\b': 'gcp',
        }

        for pattern, skill in patterns.items():
            if re.search(pattern, text):
                extracted_skills.add(skill)

        return list(extracted_skills)


class CategoryClassifier:
    """Classify courses into categories based on content"""

    def __init__(self):
        self.category_keywords = {
            'Programming': [
                'programming', 'coding', 'development', 'software', 'python', 'java',
                'javascript', 'web development', 'mobile development', 'app development'
            ],
            'Data Science': [
                'data science', 'machine learning', 'deep learning', 'artificial intelligence',
                'data analysis', 'statistics', 'analytics', 'big data', 'data mining'
            ],
            'Design': [
                'design', 'ui/ux', 'graphic design', 'web design', 'photoshop',
                'illustrator', 'figma', 'sketch', 'typography', 'branding'
            ],
            'Business': [
                'business', 'management', 'marketing', 'finance', 'accounting',
                'entrepreneurship', 'leadership', 'strategy', 'economics'
            ],
            'Technology': [
                'cloud computing', 'cybersecurity', 'blockchain', 'devops',
                'networking', 'database', 'system administration'
            ],
            'Creative': [
                'photography', 'video editing', 'animation', 'music', 'art',
                'creative writing', 'filmmaking', 'audio production'
            ],
            'Language': [
                'english', 'spanish', 'french', 'german', 'chinese', 'japanese',
                'language learning', 'linguistics', 'communication'
            ],
            'Science': [
                'physics', 'chemistry', 'biology', 'mathematics', 'engineering',
                'research', 'laboratory', 'scientific method'
            ],
            'Health': [
                'health', 'medicine', 'nutrition', 'fitness', 'psychology',
                'mental health', 'wellness', 'healthcare'
            ],
            'Education': [
                'teaching', 'education', 'pedagogy', 'curriculum', 'learning',
                'training', 'instruction', 'academic'
            ]
        }

    def classify(self, title: str, description: str = None, skills: List[str] = None) -> str:
        """Classify course into a category"""
        text = title.lower()
        if description:
            text += " " + description.lower()
        if skills:
            text += " " + " ".join(skills).lower()

        category_scores = {}

        for category, keywords in self.category_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text:
                    score += 1
            category_scores[category] = score

        # Return category with highest score, or 'General' if no matches
        if category_scores and max(category_scores.values()) > 0:
            return max(category_scores, key=category_scores.get)
        return 'General'


class DurationNormalizer:
    """Normalize course duration to weeks"""

    def __init__(self):
        self.patterns = {
            r'(\d+)\s*weeks?': lambda m: int(m.group(1)),
            r'(\d+)\s*months?': lambda m: int(m.group(1)) * 4,
            r'(\d+)\s*days?': lambda m: max(1, int(m.group(1)) // 7),
            r'(\d+)\s*hours?': lambda m: max(1, int(m.group(1)) // 10),  # Assume 10 hours per week
            r'(\d+)\s*minutes?': lambda m: 1,  # Short courses default to 1 week
            r'(\d+)-(\d+)\s*weeks?': lambda m: (int(m.group(1)) + int(m.group(2))) // 2,
            r'(\d+)-(\d+)\s*months?': lambda m: ((int(m.group(1)) + int(m.group(2))) // 2) * 4,
        }

    def normalize(self, duration_str: str) -> int:
        """Convert duration string to weeks"""
        if not duration_str:
            return 4  # Default to 4 weeks

        duration_str = duration_str.lower().strip()

        for pattern, converter in self.patterns.items():
            match = re.search(pattern, duration_str)
            if match:
                try:
                    weeks = converter(match)
                    return max(1, min(weeks, 52))  # Clamp between 1 and 52 weeks
                except (ValueError, AttributeError):
                    continue

        # Default fallback
        return 4


class PriceNormalizer:
    """Normalize course prices"""

    def __init__(self):
        self.currency_symbols = {
            '$': 'USD',
            '€': 'EUR',
            '£': 'GBP',
            '¥': 'JPY',
            '₹': 'INR'
        }

    def normalize(self, price_str: str) -> Dict[str, Any]:
        """Extract and normalize price information"""
        if not price_str:
            return {'amount': None, 'currency': None, 'is_free': False}

        price_str = price_str.strip().lower()

        # Check for free courses
        if any(word in price_str for word in ['free', 'gratis', 'kostenlos', 'gratuit']):
            return {'amount': 0, 'currency': 'USD', 'is_free': True}

        # Extract numeric price
        price_match = re.search(r'[\$€£¥₹]?(\d+(?:\.\d{2})?)', price_str)
        if price_match:
            amount = float(price_match.group(1))

            # Detect currency
            currency = 'USD'  # Default
            for symbol, curr in self.currency_symbols.items():
                if symbol in price_str:
                    currency = curr
                    break

            return {'amount': amount, 'currency': currency, 'is_free': amount == 0}

        return {'amount': None, 'currency': None, 'is_free': False}


class DifficultyMapper:
    """Map difficulty levels to standard format"""

    def __init__(self):
        self.difficulty_mapping = {
            # Beginner variations
            'beginner': 'beginner',
            'basic': 'beginner',
            'intro': 'beginner',
            'introduction': 'beginner',
            'introductory': 'beginner',
            'starter': 'beginner',
            'fundamentals': 'beginner',
            'basics': 'beginner',
            'entry': 'beginner',
            'level 1': 'beginner',

            # Intermediate variations
            'intermediate': 'intermediate',
            'medium': 'intermediate',
            'moderate': 'intermediate',
            'level 2': 'intermediate',
            'continuing': 'intermediate',

            # Advanced variations
            'advanced': 'advanced',
            'expert': 'advanced',
            'professional': 'advanced',
            'master': 'advanced',
            'level 3': 'advanced',
            'senior': 'advanced',
            'high': 'advanced',
        }

    def map_difficulty(self, difficulty_str: str) -> str:
        """Map difficulty string to standard level"""
        if not difficulty_str:
            return 'beginner'

        difficulty_str = difficulty_str.lower().strip()
        return self.difficulty_mapping.get(difficulty_str, 'beginner')


class DuplicateDetector:
    """Detect and handle duplicate courses"""

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold

    def calculate_similarity(self, course1: CourseData, course2: CourseData) -> float:
        """Calculate similarity between two courses"""
        # Title similarity (most important)
        title_similarity = self._text_similarity(course1.title, course2.title)

        # Provider check (same provider = higher chance of duplicate)
        provider_bonus = 0.2 if course1.provider == course2.provider else 0

        # URL similarity
        url_similarity = 0
        if course1.url and course2.url:
            url_similarity = self._text_similarity(course1.url, course2.url) * 0.3

        return min(1.0, title_similarity + provider_bonus + url_similarity)

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using simple word overlap"""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def find_duplicates(self, courses: List[CourseData]) -> List[Tuple[int, int, float]]:
        """Find duplicate courses and return indices with similarity scores"""
        duplicates = []

        for i in range(len(courses)):
            for j in range(i + 1, len(courses)):
                similarity = self.calculate_similarity(courses[i], courses[j])
                if similarity >= self.similarity_threshold:
                    duplicates.append((i, j, similarity))

        return duplicates

    def merge_courses(self, course1: CourseData, course2: CourseData) -> CourseData:
        """Merge two similar courses, keeping the best information"""
        # Prefer course with more complete information
        primary = course1 if self._completeness_score(course1) >= self._completeness_score(course2) else course2
        secondary = course2 if primary == course1 else course1

        # Merge skills
        merged_skills = list(set((primary.skills or []) + (secondary.skills or [])))

        # Use primary course as base and fill in missing information
        merged = CourseData(
            title=primary.title,
            provider=primary.provider,
            url=primary.url,
            description=primary.description or secondary.description,
            instructor=primary.instructor or secondary.instructor,
            duration=primary.duration or secondary.duration,
            difficulty=primary.difficulty or secondary.difficulty,
            price=primary.price or secondary.price,
            rating=primary.rating or secondary.rating,
            enrollment_count=max(primary.enrollment_count or 0, secondary.enrollment_count or 0) or None,
            skills=merged_skills,
            category=primary.category or secondary.category,
            language=primary.language or secondary.language,
            certificate_available=primary.certificate_available if primary.certificate_available is not None else secondary.certificate_available,
            scraped_at=max(primary.scraped_at or datetime.min, secondary.scraped_at or datetime.min)
        )

        return merged

    def _completeness_score(self, course: CourseData) -> int:
        """Calculate completeness score for a course"""
        score = 0
        if course.description: score += 1
        if course.instructor: score += 1
        if course.duration: score += 1
        if course.difficulty: score += 1
        if course.price: score += 1
        if course.rating: score += 1
        if course.skills: score += len(course.skills)
        return score


class CourseDataProcessor:
    """Main processor for course data standardization and enrichment"""

    def __init__(self):
        self.skill_extractor = SkillExtractor()
        self.category_classifier = CategoryClassifier()
        self.duration_normalizer = DurationNormalizer()
        self.price_normalizer = PriceNormalizer()
        self.difficulty_mapper = DifficultyMapper()
        self.duplicate_detector = DuplicateDetector()

    async def process_courses(self, courses: List[CourseData]) -> Tuple[List[CourseData], ProcessingStats]:
        """Process and standardize a list of courses"""
        stats = ProcessingStats(total_courses=len(courses))
        processed_courses = []

        # Step 1: Basic validation and cleaning
        valid_courses = []
        for course in courses:
            if self._validate_course(course):
                cleaned_course = self._clean_course(course)
                valid_courses.append(cleaned_course)
                stats.valid_courses += 1

        # Step 2: Extract and standardize skills
        for course in valid_courses:
            if not course.skills:
                course.skills = self.skill_extractor.extract_skills(course.title, course.description)
                stats.skills_extracted += 1

        # Step 3: Classify categories
        for course in valid_courses:
            if not course.category:
                course.category = self.category_classifier.classify(
                    course.title, course.description, course.skills
                )
                stats.categories_assigned += 1

        # Step 4: Normalize durations
        for course in valid_courses:
            if course.duration:
                # Store original duration in a normalized format
                normalized_weeks = self.duration_normalizer.normalize(course.duration)
                # We'll use this when converting to Course schema
                stats.durations_normalized += 1

        # Step 5: Normalize prices
        for course in valid_courses:
            if course.price:
                # Store normalized price information
                stats.prices_normalized += 1

        # Step 6: Map difficulty levels
        for course in valid_courses:
            if course.difficulty:
                course.difficulty = self.difficulty_mapper.map_difficulty(course.difficulty)

        # Step 7: Remove duplicates
        duplicates = self.duplicate_detector.find_duplicates(valid_courses)

        # Create a set of indices to remove
        indices_to_remove = set()
        for i, j, similarity in duplicates:
            if i not in indices_to_remove and j not in indices_to_remove:
                # Merge the courses and keep the merged version
                merged_course = self.duplicate_detector.merge_courses(valid_courses[i], valid_courses[j])
                valid_courses[i] = merged_course
                indices_to_remove.add(j)
                stats.duplicates_removed += 1

        # Remove duplicate courses
        processed_courses = [course for i, course in enumerate(valid_courses) if i not in indices_to_remove]

        return processed_courses, stats

    def _validate_course(self, course: CourseData) -> bool:
        """Validate that a course has minimum required information"""
        if not course.title or not course.title.strip():
            return False
        if not course.provider or not course.provider.strip():
            return False
        if not course.url or not course.url.strip():
            return False
        return True

    def _clean_course(self, course: CourseData) -> CourseData:
        """Clean and normalize course data"""
        # Clean title
        if course.title:
            course.title = re.sub(r'\s+', ' ', course.title.strip())

        # Clean description
        if course.description:
            course.description = re.sub(r'\s+', ' ', course.description.strip())
            # Limit description length
            if len(course.description) > 1000:
                course.description = course.description[:997] + "..."

        # Clean instructor name
        if course.instructor:
            course.instructor = re.sub(r'\s+', ' ', course.instructor.strip())

        # Validate and clean URL
        if course.url and not course.url.startswith(('http://', 'https://')):
            course.url = 'https://' + course.url

        return course

    def convert_to_course_schema(self, course_data: CourseData) -> Course:
        """Convert processed CourseData to Course schema"""
        # Generate a unique course ID
        course_id = f"{course_data.provider.lower().replace(' ', '_')}_{abs(hash(course_data.url)) % 1000000}"

        # Normalize duration to weeks
        duration_weeks = 4  # Default
        if course_data.duration:
            duration_weeks = self.duration_normalizer.normalize(course_data.duration)

        # Map difficulty
        difficulty = self.difficulty_mapper.map_difficulty(course_data.difficulty or 'beginner')

        # Normalize price
        price_info = self.price_normalizer.normalize(course_data.price or '')

        # Create metadata dictionary
        metadata = {
            'description': course_data.description,
            'instructor': course_data.instructor,
            'original_duration': course_data.duration,
            'rating': course_data.rating,
            'enrollment_count': course_data.enrollment_count,
            'category': course_data.category,
            'language': course_data.language or 'English',
            'certificate_available': course_data.certificate_available,
            'scraped_at': course_data.scraped_at.isoformat() if course_data.scraped_at else datetime.now().isoformat(),
            'price_info': price_info,
            'is_free': price_info.get('is_free', False),
            'last_updated': datetime.now().isoformat()
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


# Example usage
async def main():
    """Example usage of the course data processor"""
    processor = CourseDataProcessor()

    # Sample course data
    sample_courses = [
        CourseData(
            title="Python for Data Science",
            provider="Coursera",
            url="https://coursera.org/python-data-science",
            description="Learn Python programming for data analysis and machine learning",
            duration="6 weeks",
            difficulty="beginner",
            price="$49"
        ),
        CourseData(
            title="Introduction to Python for Data Science",  # Similar to above
            provider="edX",
            url="https://edx.org/intro-python-data-science",
            description="Python programming fundamentals for data science applications",
            duration="1.5 months",
            difficulty="intro",
            price="Free"
        )
    ]

    processed_courses, stats = await processor.process_courses(sample_courses)

    print(f"Processing Stats:")
    print(f"Total courses: {stats.total_courses}")
    print(f"Valid courses: {stats.valid_courses}")
    print(f"Duplicates removed: {stats.duplicates_removed}")
    print(f"Skills extracted: {stats.skills_extracted}")
    print(f"Categories assigned: {stats.categories_assigned}")

    print(f"\nProcessed courses:")
    for course in processed_courses:
        print(f"- {course.title} ({course.provider})")
        print(f"  Skills: {course.skills}")
        print(f"  Category: {course.category}")


if __name__ == "__main__":
    asyncio.run(main())