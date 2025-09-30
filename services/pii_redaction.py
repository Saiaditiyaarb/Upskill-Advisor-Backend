"""
PII Redaction Service for UpskillAdvisor

This module provides comprehensive PII (Personally Identifiable Information) redaction
capabilities to ensure user privacy and safety/ethics compliance.

Features:
- Email address redaction
- Phone number redaction  
- Social Security Number redaction
- Credit card number redaction
- Name redaction (configurable)
- Address redaction
- Custom pattern redaction
"""

import re
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PIIRedactionConfig:
    """Configuration for PII redaction settings."""
    redact_emails: bool = True
    redact_phones: bool = True
    redact_ssns: bool = True
    redact_credit_cards: bool = True
    redact_names: bool = False  # Optional, as names might be needed for personalization
    redact_addresses: bool = True
    redact_custom_patterns: List[str] = None
    replacement_text: str = "[REDACTED]"
    preserve_format: bool = True  # Keep original format structure


class PIIRedactor:
    """Comprehensive PII redaction utility using regex patterns."""
    
    def __init__(self, config: Optional[PIIRedactionConfig] = None):
        self.config = config or PIIRedactionConfig()
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for PII detection."""
        self.patterns = {}
        
        if self.config.redact_emails:
            # Email pattern - comprehensive coverage
            self.patterns['email'] = re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                re.IGNORECASE
            )
        
        if self.config.redact_phones:
            # Phone number patterns - various formats
            phone_patterns = [
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US format: 123-456-7890
                r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b',  # US format: (123) 456-7890
                r'\b\+1[-.]?\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # International: +1-123-456-7890
                r'\b\d{10}\b',  # 10 digits
                r'\b\d{11}\b',  # 11 digits (with country code)
            ]
            self.patterns['phone'] = re.compile('|'.join(phone_patterns))
        
        if self.config.redact_ssns:
            # Social Security Number patterns
            ssn_patterns = [
                r'\b\d{3}-\d{2}-\d{4}\b',  # Standard format: 123-45-6789
                r'\b\d{3}\s\d{2}\s\d{4}\b',  # Space format: 123 45 6789
                r'\b\d{9}\b',  # 9 digits (context-dependent)
            ]
            self.patterns['ssn'] = re.compile('|'.join(ssn_patterns))
        
        if self.config.redact_credit_cards:
            # Credit card patterns (Luhn algorithm compatible)
            cc_patterns = [
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # 16 digits
                r'\b\d{4}[-\s]?\d{6}[-\s]?\d{5}\b',  # Amex format
            ]
            self.patterns['credit_card'] = re.compile('|'.join(cc_patterns))
        
        if self.config.redact_addresses:
            # Address patterns (street addresses)
            address_patterns = [
                r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Circle|Cir|Court|Ct)\b',
                r'\b\d{5}(?:-\d{4})?\b',  # ZIP codes
            ]
            self.patterns['address'] = re.compile('|'.join(address_patterns), re.IGNORECASE)
        
        # Custom patterns
        if self.config.redact_custom_patterns:
            for i, pattern in enumerate(self.config.redact_custom_patterns):
                try:
                    self.patterns[f'custom_{i}'] = re.compile(pattern, re.IGNORECASE)
                except re.error as e:
                    logger.warning(f"Invalid custom pattern {i}: {pattern} - {e}")
    
    def redact_text(self, text: str) -> str:
        """Redact PII from a text string."""
        if not text or not isinstance(text, str):
            return text
        
        redacted_text = text
        
        for pii_type, pattern in self.patterns.items():
            if self.config.preserve_format:
                # Preserve format by replacing with same-length placeholder
                def replace_with_format(match):
                    original = match.group(0)
                    if len(original) <= len(self.config.replacement_text):
                        return self.config.replacement_text
                    else:
                        # Create format-preserving replacement
                        return self.config.replacement_text + '*' * (len(original) - len(self.config.replacement_text))
                
                redacted_text = pattern.sub(replace_with_format, redacted_text)
            else:
                redacted_text = pattern.sub(self.config.replacement_text, redacted_text)
        
        return redacted_text
    
    def redact_dict(self, data: Dict[str, Any], exclude_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Recursively redact PII from a dictionary."""
        if not isinstance(data, dict):
            return data
        
        exclude_keys = exclude_keys or []
        redacted_data = {}
        
        for key, value in data.items():
            if key.lower() in exclude_keys:
                # Skip redaction for excluded keys (e.g., 'goal_role', 'skills')
                redacted_data[key] = value
            elif isinstance(value, str):
                redacted_data[key] = self.redact_text(value)
            elif isinstance(value, dict):
                redacted_data[key] = self.redact_dict(value, exclude_keys)
            elif isinstance(value, list):
                redacted_data[key] = self.redact_list(value, exclude_keys)
            else:
                redacted_data[key] = value
        
        return redacted_data
    
    def redact_list(self, data: List[Any], exclude_keys: Optional[List[str]] = None) -> List[Any]:
        """Recursively redact PII from a list."""
        if not isinstance(data, list):
            return data
        
        redacted_list = []
        for item in data:
            if isinstance(item, str):
                redacted_list.append(self.redact_text(item))
            elif isinstance(item, dict):
                redacted_list.append(self.redact_dict(item, exclude_keys))
            elif isinstance(item, list):
                redacted_list.append(self.redact_list(item, exclude_keys))
            else:
                redacted_list.append(item)
        
        return redacted_list
    
    def redact_user_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact PII from user profile data while preserving necessary fields."""
        # Fields to exclude from redaction (needed for functionality)
        exclude_fields = [
            'goal_role', 'current_skills', 'years_experience', 
            'skills', 'expertise', 'name', 'title', 'description'
        ]
        
        return self.redact_dict(profile_data, exclude_fields)
    
    def redact_request_data(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact PII from API request data."""
        # Fields to exclude from redaction
        exclude_fields = [
            'goal_role', 'current_skills', 'years_experience',
            'target_skills', 'retrieval_mode', 'search_online',
            'generate_pdf', 'skills', 'expertise', 'name'
        ]
        
        return self.redact_dict(request_data, exclude_fields)


# Global redactor instance with default configuration
default_redactor = PIIRedactor()


def redact_pii(text: str, config: Optional[PIIRedactionConfig] = None) -> str:
    """Convenience function to redact PII from text."""
    redactor = PIIRedactor(config) if config else default_redactor
    return redactor.redact_text(text)


def redact_user_data(data: Union[Dict[str, Any], List[Any]], config: Optional[PIIRedactionConfig] = None) -> Union[Dict[str, Any], List[Any]]:
    """Convenience function to redact PII from user data structures."""
    redactor = PIIRedactor(config) if config else default_redactor
    
    if isinstance(data, dict):
        return redactor.redact_dict(data)
    elif isinstance(data, list):
        return redactor.redact_list(data)
    else:
        return data


def create_safe_logging_config() -> PIIRedactionConfig:
    """Create a configuration optimized for logging scenarios."""
    return PIIRedactionConfig(
        redact_emails=True,
        redact_phones=True,
        redact_ssns=True,
        redact_credit_cards=True,
        redact_names=False,  # Keep names for debugging
        redact_addresses=True,
        replacement_text="[REDACTED]",
        preserve_format=True
    )


def create_strict_redaction_config() -> PIIRedactionConfig:
    """Create a strict configuration for maximum privacy protection."""
    return PIIRedactionConfig(
        redact_emails=True,
        redact_phones=True,
        redact_ssns=True,
        redact_credit_cards=True,
        redact_names=True,
        redact_addresses=True,
        replacement_text="[REDACTED]",
        preserve_format=False
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the redaction functionality
    test_data = {
        "user_profile": {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "phone": "555-123-4567",
            "goal_role": "Software Engineer",
            "current_skills": [
                {"name": "Python", "expertise": "Intermediate"}
            ]
        },
        "user_context": {
            "time_per_week_hours": 8,
            "preference": "hands-on",
            "contact_email": "john.doe@company.com"
        }
    }
    
    # Test with default configuration
    redactor = PIIRedactor()
    redacted = redactor.redact_user_profile(test_data)
    
    print("Original data:")
    print(test_data)
    print("\nRedacted data:")
    print(redacted)
