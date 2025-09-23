"""
Job Description (JD) ingestion script for Upskill Advisor.

Responsibilities:
- Read job descriptions from various sources (folder of .txt files, .jsonl file, or structured JSON)
- Extract structured data (job_title, required_skills, years_of_experience_min) using LLM or NLP models
- Generate semantic embeddings for the extracted skills and job title
- Store structured data and embeddings in a dedicated Pinecone index or namespace
- Support batch processing for large JD corpora

Design choices for future-proofing:
- Flexible input formats to accommodate different JD data sources
- LLM-based entity extraction with fallback to rule-based methods
- Separate Pinecone namespace/index for JD data to avoid conflicts with course data
- Structured output schema that can be extended with additional fields
- Re-runnable with deterministic IDs based on content hashing
"""
from __future__ import annotations

import json
import os
import logging
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from hashlib import sha256

from core.config import get_settings

logger = logging.getLogger("ingest_jd")


@dataclass
class JobDescription:
    """Structured representation of a job description."""
    jd_id: str
    job_title: str
    required_skills: List[str]
    years_of_experience_min: Optional[int] = None
    years_of_experience_max: Optional[int] = None
    company: Optional[str] = None
    location: Optional[str] = None
    job_type: Optional[str] = None  # full-time, part-time, contract, etc.
    salary_range: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def _generate_jd_id(content: str) -> str:
    """Generate deterministic ID from job description content."""
    return f"jd_{sha256(content.encode('utf-8')).hexdigest()[:16]}"


def _load_jds_from_folder(folder_path: str) -> List[str]:
    """Load job descriptions from a folder of .txt files."""
    jds = []
    folder = Path(folder_path)

    if not folder.exists():
        logger.warning(f"JD folder not found: {folder_path}")
        return []

    txt_files = list(folder.glob("*.txt"))
    logger.info(f"Found {len(txt_files)} .txt files in {folder_path}")

    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    jds.append(content)
        except Exception as e:
            logger.warning(f"Failed to read {txt_file}: {e}")

    return jds


def _load_jds_from_jsonl(file_path: str) -> List[str]:
    """Load job descriptions from a .jsonl file."""
    jds = []

    if not os.path.exists(file_path):
        logger.warning(f"JSONL file not found: {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    # Extract job description text from various possible fields
                    jd_text = (
                        data.get('description') or
                        data.get('job_description') or
                        data.get('text') or
                        str(data)
                    )
                    if jd_text:
                        jds.append(jd_text)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
    except Exception as e:
        logger.error(f"Failed to read JSONL file {file_path}: {e}")

    logger.info(f"Loaded {len(jds)} job descriptions from {file_path}")
    return jds


def _load_jds_from_json(file_path: str) -> List[str]:
    """Load job descriptions from a structured JSON file."""
    if not os.path.exists(file_path):
        logger.warning(f"JSON file not found: {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        jds = []
        if isinstance(data, list):
            # List of job descriptions
            for item in data:
                if isinstance(item, str):
                    jds.append(item)
                elif isinstance(item, dict):
                    jd_text = (
                        item.get('description') or
                        item.get('job_description') or
                        item.get('text')
                    )
                    if jd_text:
                        jds.append(jd_text)
        elif isinstance(data, dict):
            # Single job description or nested structure
            if 'job_descriptions' in data:
                return _load_jds_from_json_data(data['job_descriptions'])
            else:
                jd_text = (
                    data.get('description') or
                    data.get('job_description') or
                    data.get('text')
                )
                if jd_text:
                    jds.append(jd_text)

        logger.info(f"Loaded {len(jds)} job descriptions from {file_path}")
        return jds

    except Exception as e:
        logger.error(f"Failed to read JSON file {file_path}: {e}")
        return []


def _load_jds_from_json_data(data: List[Dict[str, Any]]) -> List[str]:
    """Helper to extract JD text from structured JSON data."""
    jds = []
    for item in data:
        jd_text = (
            item.get('description') or
            item.get('job_description') or
            item.get('text')
        )
        if jd_text:
            jds.append(jd_text)
    return jds


async def _extract_entities_llm(jd_text: str) -> Optional[JobDescription]:
    """Extract structured entities from job description using LLM."""
    try:
        from langchain.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.exceptions import OutputParserException
    except Exception as e:
        logger.warning(f"LangChain dependencies unavailable for LLM extraction: {e}")
        return None

    try:
        # Enhanced prompt for job description entity extraction
        tmpl = """You are an expert HR analyst. Extract structured information from the following job description.

JOB DESCRIPTION:
{jd_text}

TASK:
Extract the following information and return as JSON:

{{
    "job_title": "extracted job title",
    "required_skills": ["skill1", "skill2", "skill3"],
    "years_of_experience_min": 2,
    "years_of_experience_max": 5,
    "company": "company name if mentioned",
    "location": "location if mentioned",
    "job_type": "full-time/part-time/contract/etc",
    "salary_range": "salary range if mentioned"
}}

GUIDELINES:
1. Extract specific technical and soft skills mentioned in requirements
2. Look for experience requirements (e.g., "2+ years", "3-5 years experience")
3. Normalize skill names (e.g., "JavaScript" not "JS", "Python" not "python")
4. If information is not available, use null
5. Focus on hard requirements, not nice-to-haves
6. Extract 5-15 most important skills

Extract the information:"""

        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template=tmpl,
            input_variables=["jd_text"]
        )

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        chain = prompt | llm | parser

        result = await chain.ainvoke({"jd_text": jd_text[:4000]})  # Limit text length

        if isinstance(result, dict):
            jd_id = _generate_jd_id(jd_text)

            return JobDescription(
                jd_id=jd_id,
                job_title=result.get("job_title", "Unknown"),
                required_skills=result.get("required_skills", []),
                years_of_experience_min=result.get("years_of_experience_min"),
                years_of_experience_max=result.get("years_of_experience_max"),
                company=result.get("company"),
                location=result.get("location"),
                job_type=result.get("job_type"),
                salary_range=result.get("salary_range"),
                description=jd_text[:1000],  # Store truncated description
                metadata={"extraction_method": "llm"}
            )

    except OutputParserException as e:
        logger.warning(f"JSON parsing failed for LLM extraction: {e}")
    except Exception as e:
        logger.warning(f"LLM entity extraction failed: {e}")

    return None


def _extract_entities_rule_based(jd_text: str) -> JobDescription:
    """Fallback rule-based entity extraction from job description."""
    import re

    jd_id = _generate_jd_id(jd_text)
    text_lower = jd_text.lower()

    # Extract job title (simple heuristic - first line or after "position:", "role:", etc.)
    job_title = "Unknown"
    title_patterns = [
        r'(?:position|role|job title|title):\s*([^\n]+)',
        r'^([^\n]+?)(?:\s*-\s*|\s*\|\s*)',  # First line before dash or pipe
    ]

    for pattern in title_patterns:
        match = re.search(pattern, jd_text, re.IGNORECASE | re.MULTILINE)
        if match:
            job_title = match.group(1).strip()
            break

    # Extract years of experience
    years_min = None
    years_max = None
    exp_patterns = [
        r'(\d+)\+?\s*(?:to\s+(\d+))?\s*years?\s+(?:of\s+)?experience',
        r'(\d+)-(\d+)\s*years?\s+(?:of\s+)?experience',
        r'minimum\s+(\d+)\s*years?',
        r'at least\s+(\d+)\s*years?',
    ]

    for pattern in exp_patterns:
        match = re.search(pattern, text_lower)
        if match:
            years_min = int(match.group(1))
            if match.group(2):
                years_max = int(match.group(2))
            break

    # Extract skills using common technical terms
    common_skills = [
        'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue',
        'node.js', 'express', 'django', 'flask', 'spring', 'sql', 'postgresql',
        'mysql', 'mongodb', 'redis', 'docker', 'kubernetes', 'aws', 'azure',
        'gcp', 'git', 'jenkins', 'ci/cd', 'agile', 'scrum', 'rest', 'api',
        'microservices', 'machine learning', 'data science', 'tensorflow',
        'pytorch', 'pandas', 'numpy', 'html', 'css', 'bootstrap', 'sass',
        'webpack', 'babel', 'elasticsearch', 'kafka', 'rabbitmq', 'graphql'
    ]

    found_skills = []
    for skill in common_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.append(skill.title())

    # Extract company name (simple heuristic)
    company = None
    company_patterns = [
        r'(?:company|organization):\s*([^\n]+)',
        r'join\s+([A-Z][a-zA-Z\s&]+?)(?:\s+as|\s+in|\s+to)',
    ]

    for pattern in company_patterns:
        match = re.search(pattern, jd_text, re.IGNORECASE)
        if match:
            company = match.group(1).strip()
            break

    return JobDescription(
        jd_id=jd_id,
        job_title=job_title,
        required_skills=found_skills,
        years_of_experience_min=years_min,
        years_of_experience_max=years_max,
        company=company,
        description=jd_text[:1000],
        metadata={"extraction_method": "rule_based"}
    )


def _get_embedder():
    """Get embedding function for job descriptions."""
    settings = get_settings()
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(settings.embedding_model_name)

        def _embed(texts: List[str]) -> List[List[float]]:
            return [list(map(float, v)) for v in model.encode(texts, normalize_embeddings=True)]

        logger.info(f"Using SentenceTransformer for JD embeddings: {settings.embedding_model_name}")
        return _embed
    except Exception as e:
        logger.warning(f"SentenceTransformer unavailable; using hashing fallback: {e}")

        def _hashing_embed(texts: List[str], dim: int = 256) -> List[List[float]]:
            vecs: List[List[float]] = []
            for t in texts:
                buckets = [0.0] * dim
                for token in t.lower().split():
                    h = int(sha256(token.encode("utf-8")).hexdigest(), 16)
                    idx = h % dim
                    sign = -1.0 if (h // dim) % 2 else 1.0
                    buckets[idx] += sign
                # L2 normalize
                norm = sum(x * x for x in buckets) ** 0.5 or 1.0
                vecs.append([x / norm for x in buckets])
            return vecs

        return _hashing_embed


def _build_embedding_text(jd: JobDescription) -> str:
    """Build text representation for embedding generation."""
    skills_text = ", ".join(jd.required_skills)
    exp_text = ""
    if jd.years_of_experience_min:
        exp_text = f"Experience: {jd.years_of_experience_min}"
        if jd.years_of_experience_max:
            exp_text += f"-{jd.years_of_experience_max}"
        exp_text += " years. "

    return f"{jd.job_title}. Required skills: {skills_text}. {exp_text}Company: {jd.company or 'Unknown'}."


def _upsert_pinecone_jd(vectors: List[Dict[str, Any]], namespace: str = "job_descriptions") -> None:
    """Upsert job description vectors to Pinecone."""
    settings = get_settings()
    api_key = settings.pinecone_api_key
    index_name = settings.pinecone_index

    if not api_key or not index_name:
        logger.info("Pinecone not configured. Skipping JD upsert.")
        return

    # Support both older and newer pinecone clients
    try:
        import pinecone

        pinecone.init(api_key=api_key)
        if index_name not in [i.name if hasattr(i, 'name') else i for i in pinecone.list_indexes()]:
            try:
                pinecone.create_index(index_name, dimension=len(vectors[0]["values"]))
            except Exception:
                pass
        index = pinecone.Index(index_name)
    except Exception:
        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=api_key)
            try:
                index = pc.Index(index_name)
            except Exception:
                pc.create_index(name=index_name, dimension=len(vectors[0]["values"]))
                index = pc.Index(index_name)
        except Exception as e:
            logger.warning(f"Pinecone client not available. Skipping JD upsert: {e}")
            return

    # Batch upsert
    def _batch(iterable: List[Any], size: int = 100):
        for i in range(0, len(iterable), size):
            yield iterable[i : i + size]

    upserted = 0
    for batch in _batch(vectors, size=100):
        try:
            if hasattr(index, "upsert"):
                index.upsert(vectors=batch, namespace=namespace)
            else:
                index.upsert({"vectors": batch, "namespace": namespace})
            upserted += len(batch)
        except Exception as e:
            logger.error(f"JD upsert batch failed: {e}")

    logger.info(f"JD upsert completed: {upserted} vectors in namespace '{namespace}'")


async def run(
    source_path: Optional[str] = None,
    source_type: str = "auto",
    namespace: str = "job_descriptions",
    use_llm: bool = True
) -> None:
    """
    Main function to run job description ingestion pipeline.

    Args:
        source_path: Path to JD source (folder, .jsonl, or .json file)
        source_type: "folder", "jsonl", "json", or "auto" to detect
        namespace: Pinecone namespace for JD vectors
        use_llm: Whether to use LLM for entity extraction (fallback to rule-based)
    """
    settings = get_settings()

    # Determine source path and type
    if not source_path:
        source_path = os.getenv("JD_SOURCE_PATH", os.path.join(os.getcwd(), "job_descriptions"))

    if source_type == "auto":
        if os.path.isdir(source_path):
            source_type = "folder"
        elif source_path.endswith(".jsonl"):
            source_type = "jsonl"
        elif source_path.endswith(".json"):
            source_type = "json"
        else:
            logger.error(f"Cannot determine source type for: {source_path}")
            return

    # Load raw job descriptions
    logger.info(f"Loading job descriptions from {source_path} (type: {source_type})")

    if source_type == "folder":
        raw_jds = _load_jds_from_folder(source_path)
    elif source_type == "jsonl":
        raw_jds = _load_jds_from_jsonl(source_path)
    elif source_type == "json":
        raw_jds = _load_jds_from_json(source_path)
    else:
        logger.error(f"Unsupported source type: {source_type}")
        return

    if not raw_jds:
        logger.info("No job descriptions found to process")
        return

    logger.info(f"Processing {len(raw_jds)} job descriptions")

    # Extract entities from job descriptions
    structured_jds: List[JobDescription] = []

    for i, jd_text in enumerate(raw_jds):
        logger.info(f"Processing JD {i+1}/{len(raw_jds)}")

        extracted_jd = None

        # Try LLM extraction first if enabled
        if use_llm:
            try:
                extracted_jd = await _extract_entities_llm(jd_text)
            except Exception as e:
                logger.warning(f"LLM extraction failed for JD {i+1}: {e}")

        # Fallback to rule-based extraction
        if not extracted_jd:
            extracted_jd = _extract_entities_rule_based(jd_text)

        if extracted_jd:
            structured_jds.append(extracted_jd)

    logger.info(f"Successfully extracted entities from {len(structured_jds)} job descriptions")

    if not structured_jds:
        logger.warning("No structured job descriptions to process")
        return

    # Generate embeddings
    embed = _get_embedder()
    embedding_texts = [_build_embedding_text(jd) for jd in structured_jds]
    vectors_f = embed(embedding_texts)

    # Build Pinecone-ready vectors
    vectors: List[Dict[str, Any]] = []
    for jd, vector in zip(structured_jds, vectors_f):
        metadata = {
            "job_title": jd.job_title,
            "required_skills": jd.required_skills,
            "years_of_experience_min": jd.years_of_experience_min,
            "years_of_experience_max": jd.years_of_experience_max,
            "company": jd.company,
            "location": jd.location,
            "job_type": jd.job_type,
            "salary_range": jd.salary_range,
            **(jd.metadata or {})
        }

        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}

        vectors.append({
            "id": jd.jd_id,
            "values": vector,
            "metadata": metadata
        })

    logger.info(f"Prepared {len(vectors)} JD vectors for upsert (dim: {len(vectors[0]['values'])})")

    # Upsert to Pinecone
    _upsert_pinecone_jd(vectors, namespace=namespace)

    # Save structured data locally for inspection
    output_file = f"structured_jds_{namespace}.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            jd_data = []
            for jd in structured_jds:
                jd_dict = {
                    "jd_id": jd.jd_id,
                    "job_title": jd.job_title,
                    "required_skills": jd.required_skills,
                    "years_of_experience_min": jd.years_of_experience_min,
                    "years_of_experience_max": jd.years_of_experience_max,
                    "company": jd.company,
                    "location": jd.location,
                    "job_type": jd.job_type,
                    "salary_range": jd.salary_range,
                    "description": jd.description,
                    "metadata": jd.metadata
                }
                jd_data.append(jd_dict)

            json.dump(jd_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Structured JD data saved to {output_file}")
    except Exception as e:
        logger.warning(f"Failed to save structured JD data: {e}")


if __name__ == "__main__":
    import asyncio

    # Simple CLI usage with environment variable configuration
    asyncio.run(run())