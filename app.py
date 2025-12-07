
"""Streamlit recruiter dashboard with multi-page workflow."""
from __future__ import annotations

# Bootstrap secrets from environment variables (for cloud deployment)
# This MUST run before any other imports that need credentials
import bootstrap_secrets
bootstrap_secrets.setup()

import os
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq

from matcher import ResumeMatcher, MAX_MATCH_SCORE
from parser import ResumeParserLLM
from internal_talent_pool import InternalTalentPool
from google_scheduler.services import (
    ScheduleProposal,
    ScheduledInterview,
    SchedulerPipeline,
)
from google_scheduler.services.sheets_service import SheetRow
from scheduler import (
    GoogleCalendarCredentials,
    schedule_interview,
)

logger = logging.getLogger(__name__)


THEME_CSS_PATH = Path(__file__).parent / "styles" / "theme.css"
LOGO_PATH = Path(__file__).parent / "Logo-Hacakthon.png"
ITC_LOGO_PATH = Path(__file__).parent / "ITC_Logo.jpg"


def get_embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Styling and configuration helpers
# ---------------------------------------------------------------------------

def configure_page() -> None:
    st.set_page_config(
        page_title="Recruiter Command Center",
        page_icon="💼",
        layout="wide",
    )


@lru_cache(maxsize=1)
def get_logo_base64() -> str:
    """Load and encode the logo as base64 for embedding in HTML."""
    import base64
    try:
        with open(LOGO_PATH, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        logger.warning("Logo not found at %s", LOGO_PATH)
        return ""
    except OSError as exc:
        logger.error("Unable to read logo at %s: %s", LOGO_PATH, exc)
        return ""


@lru_cache(maxsize=1)
def get_itc_logo_base64() -> str:
    """Load and encode the ITC logo as base64 for embedding in HTML."""
    import base64
    try:
        with open(ITC_LOGO_PATH, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        logger.warning("ITC Logo not found at %s", ITC_LOGO_PATH)
        return ""
    except OSError as exc:
        logger.error("Unable to read ITC logo at %s: %s", ITC_LOGO_PATH, exc)
        return ""


@lru_cache(maxsize=1)
def _load_theme_css() -> str:
    try:
        return THEME_CSS_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Theme CSS not found at %s", THEME_CSS_PATH)
    except OSError as exc:
        logger.error("Unable to read theme CSS at %s: %s", THEME_CSS_PATH, exc)
    return ""


def inject_styles() -> None:
    styles = _load_theme_css()
    if not styles:
        return

    st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders and utility dataclasses
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_resume_parser() -> ResumeParserLLM:
    return ResumeParserLLM(compute_embeddings=False)


@lru_cache(maxsize=1)
def load_resume_matcher() -> ResumeMatcher:
    return ResumeMatcher(embedding_model=get_embedding_model())


def compute_match_percentage(match_score: Any) -> float:
    """Return a human-friendly percentage while honoring the capped score scale."""
    try:
        value = float(match_score or 0)
    except (TypeError, ValueError):
        return 0.0

    # Backwards compatibility: legacy rows stored scores in [0, 1]
    if value <= 1.0:
        value *= 100.0

    cap = MAX_MATCH_SCORE if MAX_MATCH_SCORE > 0 else 100.0
    return max(0.0, min(value, cap))


# ---------------------------------------------------------------------------
# Groq LLM Client (replaces Azure OpenAI)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_groq_client() -> Optional[Groq]:
    """Get Groq client for LLM operations (chat completion)."""
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        return None
    
    return Groq(api_key=api_key)


def get_llm_model() -> str:
    """Get the LLM model to use."""
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


@lru_cache(maxsize=1)
def _load_scheduler_pipeline() -> SchedulerPipeline:
    return SchedulerPipeline.from_env()


def get_scheduler_pipeline() -> Optional[SchedulerPipeline]:
    try:
        return _load_scheduler_pipeline()
    except Exception as exc:
        st.error("⚠️ Google scheduler pipeline is not configured correctly. Check your .env values.")
        logger.exception("Failed to initialize SchedulerPipeline: %s", exc)
        return None


def generate_candidate_insight(
    candidate_name: str,
    jd_title: str,
    skills: List[str],
    experience: List[str],
    education: str,
    match_score: float,
    summary: str = "",
) -> str:
    """
    Generate a personalized AI insight explaining why this candidate is a strong fit.
    Uses Groq LLM to create a compelling, human-readable rationale.
    """
    client = get_groq_client()
    if not client:
        return "⚠️ AI insights unavailable (Groq API not configured)"
    
    try:
        
        skills_text = ", ".join(skills[:8]) if skills else "Not specified"
        experience_text = "\n".join(f"• {exp}" for exp in experience[:4]) if experience else "Not specified"
        
        match_pct = compute_match_percentage(match_score)
        
        # Different prompts for good vs poor matches
        if match_pct < 50:
            prompt = f"""You are an expert recruiter analyzing why a candidate may NOT be the best fit for a role.

**Role:** {jd_title}
**Candidate:** {candidate_name}
**Match Score:** {match_pct:.1f}% (LOW MATCH)
**Education:** {education}

**Key Skills:** {skills_text}

**Recent Experience:**
{experience_text}

**Profile Summary:** {summary[:300] if summary else "Not provided"}

Write a brief, constructive 2-3 sentence explanation of WHY this candidate is NOT a strong match for this role. Be specific about what's missing or misaligned (e.g., lacking required skills, insufficient experience in key areas, different domain focus). Be professional and objective—not harsh, but honest. Do NOT sound like a sales pitch."""
        else:
            prompt = f"""You are an expert recruiter analyzing candidate fit for a role.

**Role:** {jd_title}
**Candidate:** {candidate_name}
**Match Score:** {match_pct:.1f}%
**Education:** {education}

**Key Skills:** {skills_text}

**Recent Experience:**
{experience_text}

**Profile Summary:** {summary[:300] if summary else "Not provided"}

Write a brief, compelling 2-3 sentence insight explaining WHY this candidate would be a great fit for this role. Focus on their unique strengths, relevant experience, and potential value-add. Be specific and actionable. Make it sound natural and professional, like advice from a senior recruiter. Do NOT sound like a sales pitch or say things like "excited to have them join your team"."""

        response = client.chat.completions.create(
            model=get_llm_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        
        insight = response.choices[0].message.content.strip()
        return insight
        
    except Exception as exc:
        logger.warning("Failed to generate AI insight: %s", exc)
        return f"⚠️ Unable to generate AI insight at this time"


def enhance_job_description(jd_text: str, role_title: str = "") -> Optional[str]:
    """
    Enhance a job description using LLM to make it more attractive, clear, and comprehensive.
    
    Args:
        jd_text: Original job description text
        role_title: Optional role title for context
        
    Returns:
        Enhanced job description or None if enhancement fails
    """
    client = get_groq_client()
    if not client:
        st.error("⚠️ JD enhancement unavailable (Groq API not configured)")
        return None
    
    try:
        prompt = f"""You are an expert recruiter and HR professional. Enhance the following job description to make it more attractive, clear, and comprehensive while maintaining its core requirements and keeping it concise.

Role Title: {role_title or "Not specified"}

Current Job Description:
{jd_text}

Please enhance this job description by:
1. Making it more engaging and attractive to top talent
2. Ensuring clarity in responsibilities and requirements
3. Adding any missing standard sections (if appropriate)
4. Using professional, inclusive language
5. Keeping it concise (aim for similar length, max 20% longer)
6. Maintaining all technical requirements and key details

Return ONLY the enhanced job description text, no preamble or explanation."""

        response = client.chat.completions.create(
            model=get_llm_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500,
        )
        
        enhanced_jd = response.choices[0].message.content.strip()
        return enhanced_jd
        
    except Exception as exc:
        logger.warning("Failed to enhance JD: %s", exc)
        st.error(f"⚠️ Unable to enhance job description: {exc}")
        return None


def predict_candidate_ctc(
    candidate_name: str,
    skills: List[str],
    experience: List[str],
    education: str,
    years_of_experience: float,
    summary: str = "",
) -> str:
    """
    Predict the candidate's current CTC (salary range) based on their profile.
    Uses Groq LLM to estimate salary based on skills, experience, and education.
    """
    client = get_groq_client()
    if not client:
        return "⚠️ CTC prediction unavailable (Groq API not configured)"
    
    try:
        
        skills_text = ", ".join(skills[:12]) if skills else "Not specified"
        experience_text = "\n".join(f"• {exp}" for exp in experience[:6]) if experience else "Not specified"
        
        prompt = f"""You are a compensation analyst predicting salary ranges for candidates.

**Candidate:** {candidate_name}
**Years of Experience:** {years_of_experience:.1f} years
**Education:** {education}

**Key Skills:** {skills_text}

**Recent Experience:**
{experience_text}

**Profile Summary:** {summary[:400] if summary else "Not provided"}

Based on this profile, predict their CURRENT annual CTC (Cost to Company) in INR. Consider:
- Industry standards for their skill set
- Years of experience
- Education level
- Technical expertise and specializations

Provide:
1. A realistic salary RANGE (e.g., "Rs.80,000 - Rs.110,000")
2. ONE short sentence explaining the key factors driving this estimate

Format: 
Range: Rs.XX,XXX - Rs.XX,XXX
Rationale: [one short sentence]

Be realistic and data-driven. Don't be overly optimistic or pessimistic."""

        response = client.chat.completions.create(
            model=get_llm_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=150,
        )
        
        prediction = response.choices[0].message.content.strip()
        return prediction
        
    except Exception as exc:
        logger.warning("Failed to predict CTC: %s", exc)
        return f"⚠️ Unable to predict CTC at this time"


def generate_comprehensive_candidate_insights(
    candidate_name: str,
    jd_title: str,
    jd_text: str,
    skills: List[str],
    experience: List[str],
    education: str,
    match_score: float,
    years_of_experience: float,
    summary: str = "",
) -> Dict[str, Any]:
    """
    Generate comprehensive candidate insights in ONE LLM call.
    Returns structured JSON with all insights: profile, skills, timeline, themes, JD match, strengths/weaknesses, CTC.
    
    Returns:
        Dictionary with keys: core_profile, skill_clusters, career_timeline, responsibility_themes,
                             jd_match_breakdown, strengths, weaknesses, ctc_prediction, overall_insight
    """
    # Cache insights in session state to avoid regenerating
    cache_key = f"insights_cache_{candidate_name}_{jd_title}_{match_score}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    client = get_groq_client()
    if not client:
        return {"error": "Groq API not configured"}
    
    try:
        skills_text = ", ".join(skills[:20]) if skills else "Not specified"
        experience_text = "\n".join(f"• {exp}" for exp in experience[:10]) if experience else "Not specified"
        match_pct = compute_match_percentage(match_score)
        
        prompt = f"""You are an expert recruiter and HR analyst. Analyze this candidate comprehensively and return a structured JSON response.

**Role:** {jd_title}
**Job Description:**
{jd_text[:1500]}

**Candidate:** {candidate_name}
**Match Score:** {match_pct:.1f}%
**Years of Experience:** {years_of_experience:.1f} years
**Education:** {education}

**Skills:** {skills_text}

**Experience:**
{experience_text}

**Summary:** {summary[:600] if summary else "Not provided"}

Provide a comprehensive analysis in STRICT JSON format with these exact keys:

{{
  "core_profile": {{
    "total_experience": "{years_of_experience} years",
    "primary_domain": "e.g., Software Engineering, Mechanical Engineering",
    "seniority_level": "e.g., Senior, Mid-Level, Manager, Director",
    "current_role": "Current job title",
    "education_summary": "{education}",
    "location": "Extracted if available, else 'Not specified'"
  }},
  "skill_clusters": {{
    "technical_skills": ["skill1", "skill2", "skill3"],
    "domain_skills": ["skill1", "skill2"],
    "management_skills": ["skill1", "skill2"],
    "tools": ["tool1", "tool2"]
  }},
  "career_timeline": [
    {{"period": "2020-Present", "role": "Senior Engineer", "company": "Company A", "seniority": "senior"}},
    {{"period": "2018-2020", "role": "Engineer", "company": "Company B", "seniority": "mid"}}
  ],
  "responsibility_themes": [
    {{"theme": "Product Development", "importance": "high", "frequency": 8}},
    {{"theme": "Team Leadership", "importance": "medium", "frequency": 5}}
  ],
  "jd_match_breakdown": {{
    "overall_score": {match_pct:.0f},
    "skill_match": 75,
    "experience_match": 80,
    "domain_match": 85,
    "leadership_match": 60,
    "missing_critical_skills": ["skill1", "skill2"]
  }},
  "strengths": [
    "Strength point 1",
    "Strength point 2",
    "Strength point 3"
  ],
  "weaknesses": [
    "Weakness/gap point 1",
    "Weakness/gap point 2"
  ],
  "ctc_prediction": {{
    "range": "₹15,00,000 - ₹22,00,000",
    "rationale": "One sentence explaining the estimate"
  }},
  "overall_insight": "2-3 sentence compelling summary of why this candidate stands out or doesn't fit"
}}

IMPORTANT: 
- Return ONLY valid JSON, no markdown, no explanation
- Extract career timeline from experience entries (infer dates/companies)
- Identify responsibility themes from job descriptions
- Be realistic with match scores (not all 100%)
- Skills should be categorized intelligently
- Strengths/weaknesses should be specific and actionable"""

        response = client.chat.completions.create(
            model=get_llm_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=2000,
        )
        
        insights_json = response.choices[0].message.content.strip()
        # Clean up markdown code blocks if present
        if insights_json.startswith("```"):
            insights_json = insights_json.split("```")[1]
            if insights_json.startswith("json"):
                insights_json = insights_json[4:]
        insights = json.loads(insights_json) if insights_json else {}
        
        # Cache the results
        st.session_state[cache_key] = insights
        
        return insights
        
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse insights JSON: %s", exc)
        return {
            "error": "JSON parsing error",
            "overall_insight": "⚠️ Unable to generate comprehensive insights (invalid format)"
        }
    except Exception as exc:
        logger.warning("Failed to generate comprehensive insights: %s", exc)
        return {
            "error": str(exc),
            "overall_insight": "⚠️ Unable to generate comprehensive insights at this time"
        }


# ---------------------------------------------------------------------------
# Visualization helpers for candidate insights dashboard
# ---------------------------------------------------------------------------

def create_skill_cluster_chart(skill_clusters: Dict[str, List[str]]) -> str:
    """Create a horizontal bar chart HTML for skill clusters."""
    if not skill_clusters:
        return "<p style='font-size: 0.7rem; color: #64748b; margin: 0;'>No skill data available</p>"
    
    # Count skills per category
    categories = []
    counts = []
    colors = {
        "technical_skills": "#6366f1",
        "domain_skills": "#10b981",
        "management_skills": "#f59e0b",
        "tools": "#8b5cf6"
    }
    
    for category, skills in skill_clusters.items():
        if skills:
            categories.append(category.replace("_", " ").title())
            counts.append(len(skills))
    
    if not categories:
        return "<p style='font-size: 0.7rem; color: #64748b; margin: 0;'>No skill data available</p>"
    
    max_count = max(counts)
    bars_html = ""
    
    for i, (cat, count) in enumerate(zip(categories, counts)):
        width_pct = (count / max_count * 100) if max_count > 0 else 0
        color = list(colors.values())[i % len(colors)]
        bars_html += f"""
        <div style="margin-bottom: 0.5rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.15rem;">
                <span style="font-size: 0.7rem; font-weight: 500; color: #334155;">{cat}</span>
                <span style="font-size: 0.65rem; color: #64748b;">{count}</span>
            </div>
            <div style="background: #e2e8f0; border-radius: 4px; height: 16px; overflow: hidden;">
                <div style="background: {color}; height: 100%; width: {width_pct}%; border-radius: 4px; transition: width 0.3s ease;"></div>
            </div>
        </div>
        """
    
    return bars_html


def create_career_timeline_html(timeline: List[Dict[str, str]]) -> str:
    """Create a visual timeline HTML for career progression."""
    if not timeline:
        return "<p style='font-size: 0.7rem; color: #64748b; margin: 0;'>No career timeline available</p>"
    
    seniority_colors = {
        "senior": "#10b981",
        "mid": "#6366f1",
        "junior": "#8b5cf6",
        "manager": "#f59e0b",
        "director": "#ef4444"
    }
    
    timeline_html = '<div style="position: relative; padding-left: 1.2rem;">'
    
    for entry in timeline:
        period = entry.get("period", "Unknown")
        role = entry.get("role", "Unknown Role")
        company = entry.get("company", "")
        seniority = entry.get("seniority", "mid").lower()
        color = seniority_colors.get(seniority, "#6366f1")
        
        timeline_html += f"""
        <div style="position: relative; margin-bottom: 0.8rem; padding-left: 1rem; border-left: 2px solid {color};">
            <div style="position: absolute; left: -0.4rem; top: 0.2rem; width: 0.6rem; height: 0.6rem; border-radius: 50%; background: {color};"></div>
            <div style="font-size: 0.65rem; color: #64748b; font-weight: 600; margin-bottom: 0.1rem;">{period}</div>
            <div style="font-size: 0.75rem; color: #0f172a; font-weight: 600; line-height: 1.2;">{role}</div>
            {f'<div style="font-size: 0.7rem; color: #475569;">{company}</div>' if company else ''}
        </div>
        """
    
    timeline_html += '</div>'
    return timeline_html


def create_responsibility_themes_html(themes: List[Dict[str, Any]]) -> str:
    """Create a bubble/tag cluster HTML for responsibility themes."""
    if not themes:
        return "<p>No themes identified</p>"
    
    importance_colors = {
        "high": "#10b981",
        "medium": "#6366f1",
        "low": "#8b5cf6"
    }
    
    themes_html = '<div style="display: flex; flex-wrap: wrap; gap: 0.75rem;">'
    
    for theme_data in themes[:8]:  # Limit to top 8
        theme = theme_data.get("theme", "")
        importance = theme_data.get("importance", "medium").lower()
        frequency = theme_data.get("frequency", 1)
        color = importance_colors.get(importance, "#6366f1")
        
        # Size based on frequency
        size = "0.85rem" if frequency < 3 else "0.95rem" if frequency < 6 else "1.05rem"
        padding = "0.5rem 1rem" if frequency < 3 else "0.6rem 1.2rem" if frequency < 6 else "0.7rem 1.4rem"
        
        themes_html += f"""
        <div style="
            background: {color}15;
            border: 2px solid {color};
            border-radius: 24px;
            padding: {padding};
            font-size: {size};
            font-weight: 600;
            color: {color};
            white-space: nowrap;
        ">
            {theme}
        </div>
        """
    
    themes_html += '</div>'
    return themes_html


def create_jd_match_radar_html(match_breakdown: Dict[str, Any]) -> str:
    """Create a visual representation of JD match scores."""
    if not match_breakdown:
        return "<p style='font-size: 0.7rem; color: #64748b; margin: 0;'>No match data available</p>"
    
    categories = ["Skill Match", "Experience Match", "Domain Match", "Leadership Match"]
    scores = [
        match_breakdown.get("skill_match", 0),
        match_breakdown.get("experience_match", 0),
        match_breakdown.get("domain_match", 0),
        match_breakdown.get("leadership_match", 0),
    ]
    
    match_html = ""
    for cat, score in zip(categories, scores):
        color = "#10b981" if score >= 70 else "#f59e0b" if score >= 50 else "#ef4444"
        match_html += f"""
        <div style="margin-bottom: 0.6rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.15rem;">
                <span style="font-size: 0.7rem; font-weight: 500; color: #334155;">{cat}</span>
                <span style="font-size: 0.7rem; font-weight: 700; color: {color};">{score}%</span>
            </div>
            <div style="background: #e2e8f0; border-radius: 4px; height: 14px; overflow: hidden;">
                <div style="background: {color}; height: 100%; width: {score}%; border-radius: 4px; transition: width 0.3s ease;"></div>
            </div>
        </div>
        """
    
    # Missing skills
    missing = match_breakdown.get("missing_critical_skills", [])
    if missing:
        missing_html = ", ".join(missing[:3])
        match_html += f"""
        <div style="margin-top: 0.6rem; padding: 0.5rem; background: #fef2f2; border-left: 2px solid #ef4444; border-radius: 4px;">
            <div style="font-size: 0.7rem; font-weight: 600; color: #991b1b; margin-bottom: 0.15rem;">Missing:</div>
            <div style="font-size: 0.65rem; color: #dc2626; line-height: 1.2;">{missing_html}</div>
        </div>
        """
    
    return match_html


def create_strengths_weaknesses_html(strengths: List[str], weaknesses: List[str]) -> str:
    """Create a two-column strengths/weaknesses display."""
    strengths_html = ""
    for strength in strengths[:3]:
        strengths_html += f"""
        <div style="display: flex; align-items: start; margin-bottom: 0.4rem;">
            <span style="color: #10b981; font-size: 0.9rem; margin-right: 0.4rem;">✓</span>
            <span style="font-size: 0.7rem; color: #334155; line-height: 1.3;">{strength}</span>
        </div>
        """
    
    weaknesses_html = ""
    for weakness in weaknesses[:3]:
        weaknesses_html += f"""
        <div style="display: flex; align-items: start; margin-bottom: 0.4rem;">
            <span style="color: #f59e0b; font-size: 0.9rem; margin-right: 0.4rem;">⚠</span>
            <span style="font-size: 0.7rem; color: #334155; line-height: 1.3;">{weakness}</span>
        </div>
        """
    
    return strengths_html, weaknesses_html


def create_modern_dashboard_html(
    candidate_name: str,
    email: str,
    phone: str,
    location: str,
    current_role: str,
    match_pct: float,
    experience_years: str,
    seniority_level: str,
    domain: str,
    education: str,
    skill_match: int,
    domain_match: int,
    experience_match: int,
    leadership_match: int,
    ctc_range: str,
    ctc_rationale: str,
    skill_clusters: Dict[str, List[str]],
    timeline: List[Dict[str, str]],
    strengths: List[str],
    missing_skills: List[str],
    ai_summary: str,
) -> str:
    """Generate modern dashboard HTML based on the React design."""
    
    # Build skills sections HTML
    skills_sections = ""
    skill_colors = {
        "technical_skills": {"bar": "#3b82f6 to #4f46e5", "bg": "#dbeafe", "text": "#1e40af", "label": "Technical Skills"},
        "domain_skills": {"bar": "#a855f7 to #ec4899", "bg": "#f3e8ff", "text": "#7e22ce", "label": "Domain Skills"},
        "management_skills": {"bar": "#f59e0b to #f97316", "bg": "#fef3c7", "text": "#92400e", "label": "Management Skills"},
        "tools": {"bar": "#14b8a6 to #06b6d4", "bg": "#ccfbf1", "text": "#115e59", "label": "Tools & Frameworks"},
    }
    
    for category, skills in skill_clusters.items():
        if not skills:
            continue
        config = skill_colors.get(category, skill_colors["technical_skills"])
        skill_count = len(skills)
        width_pct = min(100, (skill_count / 10) * 100)
        
        skills_tags = "".join([
            f'<span style="padding: 0.25rem 0.75rem; background: {config["bg"]}; color: {config["text"]}; border-radius: 9999px; font-size: 0.75rem; font-weight: 500;">{skill}</span>'
            for skill in skills[:8]
        ])
        
        skills_sections += f'''
        <div style="margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 0.875rem; font-weight: 500; color: #334155;">{config["label"]}</span>
                <span style="font-size: 0.875rem; font-weight: 700; color: {config["text"]};">{skill_count} skills</span>
            </div>
            <div style="height: 0.75rem; background: #f1f5f9; border-radius: 9999px; overflow: hidden; margin-bottom: 0.75rem;">
                <div style="height: 100%; background: linear-gradient(to right, {config["bar"]}); border-radius: 9999px; width: {width_pct}%;"></div>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                {skills_tags}
            </div>
        </div>
        '''
    
    # Build timeline HTML
    timeline_items = ""
    timeline_colors = ["#2563eb", "#a855f7", "#14b8a6"]
    for i, entry in enumerate(timeline[:3]):
        period = entry.get("period", "Unknown")
        role = entry.get("role", "Unknown Role")
        company = entry.get("company", "")
        color = timeline_colors[i % len(timeline_colors)]
        is_current = i == 0
        
        badge = f'<span style="font-size: 0.75rem; background: {color}; color: white; padding: 0.25rem 0.5rem; border-radius: 0.25rem;">Current</span>' if is_current else ''
        bg_color = "#eff6ff" if is_current else "#f8fafc"
        border_color = "#bfdbfe" if is_current else "#e2e8f0"
        
        timeline_items += f'''
        <div style="position: relative; padding-bottom: 2rem;">
            <div style="display: flex; gap: 1rem;">
                <div style="position: relative; z-index: 10;">
                    <div style="width: 2rem; height: 2rem; background: {color}; border-radius: 9999px; display: flex; align-items: center; justify-content: center;">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                            <rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect>
                            <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path>
                        </svg>
                    </div>
                </div>
                <div style="flex: 1;">
                    <div style="background: {bg_color}; border-radius: 0.5rem; padding: 1rem; border: 1px solid {border_color};">
                        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.25rem;">
                            <h4 style="font-weight: 600; color: #1e293b; margin: 0;">{role}</h4>
                            {badge}
                        </div>
                        <p style="font-size: 0.875rem; color: {color}; font-weight: 500; margin: 0.25rem 0;">{company}</p>
                        <p style="font-size: 0.75rem; color: #64748b; margin: 0;">{period}</p>
                    </div>
                </div>
            </div>
        </div>
        '''
    
    # Build strengths HTML
    strengths_items = "".join([
        f'''<li style="font-size: 0.875rem; color: #334155; display: flex; align-items: start; gap: 0.5rem;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" style="margin-top: 0.125rem; flex-shrink: 0;">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                <polyline points="22 4 12 14.01 9 11.01"></polyline>
            </svg>
            <span>{strength}</span>
        </li>'''
        for strength in strengths[:3]
    ])
    
    # Build missing skills HTML
    missing_items = "".join([
        f'''<li style="font-size: 0.875rem; color: #334155; display: flex; align-items: start; gap: 0.5rem;">
            <span style="color: #f59e0b; margin-top: 0.125rem;">•</span>
            <span>{skill}</span>
        </li>'''
        for skill in missing_skills[:3]
    ])
    
    # Calculate circle progress for match percentage
    circle_circumference = 351.86
    circle_offset = circle_circumference * (1 - match_pct / 100)
    
    dashboard_html = f'''
    <div style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 1rem; padding: 2rem; margin: 1rem 0;">
        <!-- Profile Hero Card -->
        <div style="background: linear-gradient(to right, #2563eb, #4f46e5); border-radius: 1rem; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1); padding: 2rem; margin-bottom: 2rem; color: white;">
            <div style="display: flex; align-items: start; justify-content: space-between;">
                <div style="display: flex; gap: 1.5rem;">
                    <div style="width: 6rem; height: 6rem; background: rgba(255,255,255,0.2); backdrop-filter: blur(10px); border-radius: 0.75rem; display: flex; align-items: center; justify-content: center;">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                            <circle cx="12" cy="7" r="4"></circle>
                        </svg>
                    </div>
                    <div>
                        <h2 style="font-size: 1.875rem; font-weight: 700; margin: 0 0 0.5rem 0;">{candidate_name}</h2>
                        <p style="color: #bfdbfe; font-size: 1.125rem; margin: 0 0 1rem 0;">{current_role}</p>
                        <div style="display: flex; flex-wrap: wrap; gap: 1rem; font-size: 0.875rem;">
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>
                                    <circle cx="12" cy="10" r="3"></circle>
                                </svg>
                                <span>{location}</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path>
                                    <polyline points="22,6 12,13 2,6"></polyline>
                                </svg>
                                <span>{email}</span>
                            </div>
                            {f'<div style="display: flex; align-items: center; gap: 0.5rem;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path></svg><span>{phone}</span></div>' if phone else ''}
                        </div>
                    </div>
                </div>
                <div style="text-align: center;">
                    <div style="position: relative; width: 8rem; height: 8rem;">
                        <svg style="width: 8rem; height: 8rem; transform: rotate(-90deg);">
                            <circle cx="64" cy="64" r="56" stroke="rgba(255,255,255,0.2)" stroke-width="8" fill="none" />
                            <circle cx="64" cy="64" r="56" stroke="white" stroke-width="8" fill="none" 
                                stroke-dasharray="{circle_circumference}" 
                                stroke-dashoffset="{circle_offset}" 
                                stroke-linecap="round" />
                        </svg>
                        <div style="position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                            <span style="font-size: 2.5rem; font-weight: 700;">{int(match_pct)}%</span>
                            <span style="font-size: 0.875rem; color: #bfdbfe;">Match</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Quick Stats -->
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; margin-bottom: 2rem;">
            <div style="background: white; border-radius: 0.75rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); padding: 1.5rem; border: 1px solid #e2e8f0;">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                    <div style="width: 2.5rem; height: 2.5rem; background: #dbeafe; border-radius: 0.5rem; display: flex; align-items: center; justify-content: center;">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2">
                            <rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect>
                            <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path>
                        </svg>
                    </div>
                    <span style="color: #64748b; font-size: 0.875rem;">Experience</span>
                </div>
                <p style="font-size: 1.5rem; font-weight: 700; color: #1e293b; margin: 0;">{experience_years}</p>
            </div>
            <div style="background: white; border-radius: 0.75rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); padding: 1.5rem; border: 1px solid #e2e8f0;">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                    <div style="width: 2.5rem; height: 2.5rem; background: #f3e8ff; border-radius: 0.5rem; display: flex; align-items: center; justify-content: center;">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#a855f7" stroke-width="2">
                            <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>
                            <polyline points="17 6 23 6 23 12"></polyline>
                        </svg>
                    </div>
                    <span style="color: #64748b; font-size: 0.875rem;">Level</span>
                </div>
                <p style="font-size: 1.5rem; font-weight: 700; color: #1e293b; margin: 0;">{seniority_level}</p>
            </div>
            <div style="background: white; border-radius: 0.75rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); padding: 1.5rem; border: 1px solid #e2e8f0;">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                    <div style="width: 2.5rem; height: 2.5rem; background: #d1fae5; border-radius: 0.5rem; display: flex; align-items: center; justify-content: center;">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2">
                            <path d="M12 2L2 7l10 5 10-5-10-5z"></path>
                            <path d="M2 17l10 5 10-5M2 12l10 5 10-5"></path>
                        </svg>
                    </div>
                    <span style="color: #64748b; font-size: 0.875rem;">Domain</span>
                </div>
                <p style="font-size: 1.5rem; font-weight: 700; color: #1e293b; margin: 0;">{domain}</p>
            </div>
            <div style="background: white; border-radius: 0.75rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); padding: 1.5rem; border: 1px solid #e2e8f0;">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                    <div style="width: 2.5rem; height: 2.5rem; background: #fef3c7; border-radius: 0.5rem; display: flex; align-items: center; justify-content: center;">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2">
                            <path d="M22 10v6M2 10l10-5 10 5-10 5z"></path>
                            <path d="M6 12v5c3 3 9 3 12 0v-5"></path>
                        </svg>
                    </div>
                    <span style="color: #64748b; font-size: 0.875rem;">Education</span>
                </div>
                <p style="font-size: 1.5rem; font-weight: 700; color: #1e293b; margin: 0;">{education}</p>
            </div>
        </div>

        <!-- Main Content Grid -->
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem;">
            <!-- Left Column - Matching & CTC -->
            <div style="display: flex; flex-direction: column; gap: 1.5rem;">
                <!-- Fit Analysis -->
                <div style="background: white; border-radius: 0.75rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); padding: 1.5rem; border: 1px solid #e2e8f0;">
                    <h3 style="font-size: 1.125rem; font-weight: 600; color: #1e293b; margin: 0 0 1rem 0;">Fit Analysis</h3>
                    <div style="display: flex; flex-direction: column; gap: 1rem;">
                        <div>
                            <div style="display: flex; justify-between; align-items: center; margin-bottom: 0.5rem;">
                                <span style="font-size: 0.875rem; font-weight: 500; color: #334155;">Skill Match</span>
                                <span style="font-size: 0.875rem; font-weight: 700; color: #10b981;">{skill_match}%</span>
                            </div>
                            <div style="height: 0.5rem; background: #f1f5f9; border-radius: 9999px; overflow: hidden;">
                                <div style="height: 100%; background: linear-gradient(to right, #10b981, #059669); border-radius: 9999px; width: {skill_match}%;"></div>
                            </div>
                        </div>
                        <div>
                            <div style="display: flex; justify-between; align-items: center; margin-bottom: 0.5rem;">
                                <span style="font-size: 0.875rem; font-weight: 500; color: #334155;">Domain Match</span>
                                <span style="font-size: 0.875rem; font-weight: 700; color: #10b981;">{domain_match}%</span>
                            </div>
                            <div style="height: 0.5rem; background: #f1f5f9; border-radius: 9999px; overflow: hidden;">
                                <div style="height: 100%; background: linear-gradient(to right, #10b981, #059669); border-radius: 9999px; width: {domain_match}%;"></div>
                            </div>
                        </div>
                        <div>
                            <div style="display: flex; justify-between; align-items: center; margin-bottom: 0.5rem;">
                                <span style="font-size: 0.875rem; font-weight: 500; color: #334155;">Experience Match</span>
                                <span style="font-size: 0.875rem; font-weight: 700; color: #eab308;">{experience_match}%</span>
                            </div>
                            <div style="height: 0.5rem; background: #f1f5f9; border-radius: 9999px; overflow: hidden;">
                                <div style="height: 100%; background: linear-gradient(to right, #eab308, #ca8a04); border-radius: 9999px; width: {experience_match}%;"></div>
                            </div>
                        </div>
                        <div>
                            <div style="display: flex; justify-between; align-items: center; margin-bottom: 0.5rem;">
                                <span style="font-size: 0.875rem; font-weight: 500; color: #334155;">Leadership Match</span>
                                <span style="font-size: 0.875rem; font-weight: 700; color: #f97316;">{leadership_match}%</span>
                            </div>
                            <div style="height: 0.5rem; background: #f1f5f9; border-radius: 9999px; overflow: hidden;">
                                <div style="height: 100%; background: linear-gradient(to right, #f97316, #ea580c); border-radius: 9999px; width: {leadership_match}%;"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- CTC Prediction -->
                <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border-radius: 0.75rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); padding: 1.5rem; border: 1px solid #6ee7b7;">
                    <h3 style="font-size: 1.125rem; font-weight: 600; color: #1e293b; margin: 0 0 0.25rem 0;">Predicted CTC</h3>
                    <p style="font-size: 0.875rem; color: #64748b; margin: 0 0 1rem 0;">Based on skills and experience</p>
                    <div style="font-size: 1.875rem; font-weight: 700; color: #047857; margin-bottom: 1rem;">{ctc_range}</div>
                    <p style="font-size: 0.75rem; color: #64748b; line-height: 1.4;">{ctc_rationale[:120]}...</p>
                </div>

                <!-- Missing Skills -->
                <div style="background: #fef3c7; border-radius: 0.75rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); padding: 1.5rem; border: 1px solid #fde68a;">
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2">
                            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                            <line x1="12" y1="9" x2="12" y2="13"></line>
                            <line x1="12" y1="17" x2="12.01" y2="17"></line>
                        </svg>
                        <h3 style="font-size: 1.125rem; font-weight: 600; color: #1e293b; margin: 0;">Development Areas</h3>
                    </div>
                    <ul style="list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 0.5rem;">
                        {missing_items}
                    </ul>
                </div>
            </div>

            <!-- Center Column - Skills -->
            <div style="display: flex; flex-direction: column; gap: 1.5rem;">
                <div style="background: white; border-radius: 0.75rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); padding: 1.5rem; border: 1px solid #e2e8f0;">
                    <h3 style="font-size: 1.125rem; font-weight: 600; color: #1e293b; margin: 0 0 1rem 0;">Skills Overview</h3>
                    {skills_sections}
                </div>

                <!-- Strengths -->
                <div style="background: #d1fae5; border-radius: 0.75rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); padding: 1.5rem; border: 1px solid #6ee7b7;">
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2">
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                            <polyline points="22 4 12 14.01 9 11.01"></polyline>
                        </svg>
                        <h3 style="font-size: 1.125rem; font-weight: 600; color: #1e293b; margin: 0;">Key Strengths</h3>
                    </div>
                    <ul style="list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 0.5rem;">
                        {strengths_items}
                    </ul>
                </div>
            </div>

            <!-- Right Column - Timeline & Summary -->
            <div style="display: flex; flex-direction: column; gap: 1.5rem;">
                <div style="background: white; border-radius: 0.75rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); padding: 1.5rem; border: 1px solid #e2e8f0;">
                    <h3 style="font-size: 1.125rem; font-weight: 600; color: #1e293b; margin: 0 0 1.5rem 0;">Experience Timeline</h3>
                    <div style="position: relative;">
                        <div style="position: absolute; left: 1rem; top: 0; bottom: 0; width: 0.125rem; background: #e2e8f0;"></div>
                        {timeline_items}
                    </div>
                </div>

                <!-- AI Summary -->
                <div style="background: linear-gradient(135deg, #e0e7ff 0%, #ddd6fe 100%); border-radius: 0.75rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); padding: 1.5rem; border: 1px solid #c7d2fe;">
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#4f46e5" stroke-width="2">
                            <path d="M12 2L2 7l10 5 10-5-10-5z"></path>
                            <path d="M2 17l10 5 10-5M2 12l10 5 10-5"></path>
                        </svg>
                        <h3 style="font-size: 1.125rem; font-weight: 600; color: #1e293b; margin: 0;">AI Summary</h3>
                    </div>
                    <p style="font-size: 0.875rem; color: #334155; line-height: 1.6; margin: 0;">{ai_summary}</p>
                </div>
            </div>
        </div>
    </div>
    '''
    
    return dashboard_html


@dataclass
class DashboardConfig:
    form_id: Optional[str]
    google_form_link: Optional[str]
    interviewer_email: str
    location: str
    subject_template: str
    description: str


SESSION_DEFAULTS = {
    "jd_library": [],
    "selected_jd_id": None,
    "parsed_resumes": None,
    "reference_resumes": None,  # For previously hired candidates
    "ranked_results": {},
    "filtered_ranked_results": {},
    "forms_df": None,
    "selected_slots": [],
    # Interview scheduling state
    "interview_responses_df": None,
    "interview_schedule_df": None,
    "new_responses_count": 0,
    "skipped_responses_count": 0,
    "google_form_rows": [],
    "schedule_proposals": [],
    "schedule_proposals_df": None,
    "scheduled_results": [],
    "confirmed_schedule_df": None,
    # Internal talent pool state
    "talent_pool_manager": None,
    "internal_matches": None,
    "internal_search_performed": False,
}


def ensure_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            if isinstance(value, dict):
                st.session_state[key] = value.copy()
            elif isinstance(value, list):
                st.session_state[key] = list(value)
            else:
                st.session_state[key] = value


def get_jd_library() -> List[Dict[str, str]]:
    return st.session_state["jd_library"]


def set_selected_jd(jd_id: Optional[str]) -> None:
    st.session_state["selected_jd_id"] = jd_id


def get_selected_jd() -> Optional[Dict[str, str]]:
    jd_id = st.session_state.get("selected_jd_id")
    for jd in get_jd_library():
        if jd["id"] == jd_id:
            return jd
    return None


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

SAMPLE_AVAILABILITY = pd.DataFrame(
    [
        {
            "Email": "candidate1@example.com",
            "Preferred Start": "2025-11-13T14:00:00",
            "Preferred End": "2025-11-13T15:00:00",
            "Submitted": "2025-11-10T09:45:00Z",
        },
        {
            "Email": "candidate2@example.com",
            "Preferred Start": "2025-11-14T16:00:00",
            "Preferred End": "2025-11-14T17:00:00",
            "Submitted": "2025-11-10T11:10:00Z",
        },
        {
            "Email": "candidate3@example.com",
            "Preferred Start": "2025-11-15T10:00:00",
            "Preferred End": "2025-11-15T11:00:00",
            "Submitted": "2025-11-10T13:20:00Z",
        },
    ]
)

CURRENT_YEAR = datetime.now().year
EDUCATION_TIERS = [
    "Any",
    "High School / GED",
    "Associate",
    "Bachelor",
    "Master",
    "Doctorate",
]
EDUCATION_SCORE = {level: idx for idx, level in enumerate(EDUCATION_TIERS)}
EDUCATION_KEYWORDS = {
    "doctor": "Doctorate",
    "phd": "Doctorate",
    "master": "Master",
    "mba": "Master",
    "bachelor": "Bachelor",
    "bs ": "Bachelor",
    "ba ": "Bachelor",
    "associate": "Associate",
    "high school": "High School / GED",
    "ged": "High School / GED",
    "diploma": "High School / GED",
}
YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")
DURATION_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s+years?", re.IGNORECASE)


def ensure_iterable(value: Optional[Any]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


def estimate_years_of_experience(experiences: Iterable[str]) -> float:
    entries = [entry for entry in experiences if isinstance(entry, str)]
    years = []
    durations = []
    for entry in entries:
        years.extend([int(match) for match in YEAR_PATTERN.findall(entry) if match])
        duration_match = DURATION_PATTERN.search(entry)
        if duration_match:
            try:
                durations.append(float(duration_match.group(1)))
            except ValueError:
                continue
    if len(years) >= 2:
        return float(max(years) - min(years))
    if durations:
        return sum(durations)
    if len(years) == 1:
        return float(CURRENT_YEAR - years[0])
    return 0.0


def normalize_education_level(education_entries: Iterable[str]) -> Tuple[str, int]:
    best_level = "Any"
    best_score = EDUCATION_SCORE[best_level]
    for entry in education_entries:
        lower = entry.lower()
        for keyword, level in EDUCATION_KEYWORDS.items():
            if keyword in lower and EDUCATION_SCORE.get(level, 0) > best_score:
                best_level = level
                best_score = EDUCATION_SCORE.get(level, 0)
    return best_level, best_score


def filter_ranked_candidates(
    ranked_df: pd.DataFrame,
    min_years: float,
    max_years: float,
    min_education: str,
) -> pd.DataFrame:
    """Return a normalized dataframe with eligibility columns and compact prioritization."""
    df = ranked_df.copy()
    experience_column = df["experience"] if "experience" in df else pd.Series([[]] * len(df), index=df.index)
    df["experience_list"] = experience_column.apply(ensure_iterable)
    df["computed_years"] = df["experience_list"].apply(estimate_years_of_experience)

    education_column = df["education"] if "education" in df else pd.Series([[]] * len(df), index=df.index)
    df["education_list"] = education_column.apply(ensure_iterable)
    education_levels = df["education_list"].apply(normalize_education_level)
    df["education_label"] = education_levels.apply(lambda pair: pair[0])
    df["education_score"] = education_levels.apply(lambda pair: pair[1])

    # Range-based eligibility checks
    df["meets_experience"] = df["computed_years"].apply(
        lambda y: float(min_years) <= float(y or 0) <= float(max_years)
    )
    required_edu_score = EDUCATION_SCORE.get(min_education, 0)
    df["meets_education"] = df["education_score"] >= required_edu_score
    df["meets_all_criteria"] = df["meets_experience"] & df["meets_education"]

    df["match_score"] = pd.to_numeric(df.get("match_score"), errors="coerce").fillna(0.0)

    filtered = df[df["meets_all_criteria"]].copy()
    if filtered.empty:
        filtered = df.copy()
    filtered = filtered.sort_values(by=["meets_all_criteria", "match_score"], ascending=[False, False])
    filtered["meets_all_criteria"] = filtered["meets_all_criteria"].fillna(False)
    return filtered

def parse_form_responses(payload: Dict[str, Any]) -> pd.DataFrame:
    """Normalize Microsoft Forms response payload into a DataFrame."""
    rows: List[Dict[str, Any]] = []
    for response in payload.get("value", []):
        row: Dict[str, Any] = {
            "Response ID": response.get("id"),
            "Submitted": response.get("submittedDateTime"),
        }
        for answer in response.get("answers", []):
            label = answer.get("display") or answer.get("questionId") or "response"
            value = (
                answer.get("text")
                or answer.get("answer")
                or answer.get("value")
                or answer.get("choice")
            )
            row[label] = value
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def fetch_form_responses(form_id: str, credentials) -> pd.DataFrame:
    """
    Fetch Microsoft Forms responses via Graph API.
    
    DEPRECATED: This function is no longer used as Microsoft Graph API has been removed.
    Kept for reference only.
    """
    raise NotImplementedError(
        "Microsoft Forms integration via Graph API has been removed. "
        "Please use Google Forms or another alternative."
    )
    
    # Old implementation - commented out
    # from scheduler import get_graph_access_token
    #
    # token = get_graph_access_token(credentials)
    # url = f"https://graph.microsoft.com/beta/forms/{form_id}/responses"
    # headers = {
    #     "Authorization": f"Bearer {token}",
    #     "Accept": "application/json",
    # }
    # response = requests.get(url, headers=headers, timeout=15)
    # if response.status_code >= 400:
    #     try:
    #         details = response.json()
    #     except ValueError:
    #         details = response.text
    #     raise RuntimeError(f"Graph API error: {details}")
    # payload = response.json()
    # return parse_form_responses(payload)


# ---------------------------------------------------------------------------
# Google scheduler helpers
# ---------------------------------------------------------------------------


def _rows_to_dataframe(rows: List[SheetRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    records: List[Dict[str, Any]] = []
    for row in rows:
        record = dict(row.data)
        record["Row #"] = row.row_number
        records.append(record)
    return pd.DataFrame(records)


def handle_google_form_fetch(pipeline: SchedulerPipeline) -> pd.DataFrame:
    with st.spinner("Fetching Google Form responses from Sheets..."):
        rows = pipeline.fetch_form_responses()

    st.session_state["google_form_rows"] = rows
    df = _rows_to_dataframe(rows)
    st.session_state["interview_responses_df"] = df
    st.session_state["new_responses_count"] = len(df)
    st.session_state["skipped_responses_count"] = 0
    st.session_state["schedule_proposals"] = []
    st.session_state["schedule_proposals_df"] = None
    st.session_state["scheduled_results"] = []
    st.session_state["confirmed_schedule_df"] = None

    if df.empty:
        st.info("No new responses found in the configured Google Sheet.")
    else:
        st.success(f"Fetched {len(df)} response(s) from Google Sheets.")
    return df


def _proposals_to_dataframe(proposals: List[ScheduleProposal]) -> pd.DataFrame:
    if not proposals:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    def _fmt(dt_value: Optional[datetime]) -> str:
        """Format datetime as human-readable string (e.g., '24-Nov-2025 10:00 AM')"""
        if not isinstance(dt_value, datetime):
            return ""
        # Format: DD-Mon-YYYY HH:MM AM/PM
        return dt_value.strftime("%d-%b-%Y %I:%M %p")

    for idx, proposal in enumerate(proposals):
        candidate = proposal.candidate
        rows.append(
            {
                "proposal_id": idx,
                "Candidate": candidate.name,
                "Email": candidate.email,
                "Preferred Date": candidate.preferred_date.strftime("%Y-%m-%d"),
                "Preferred Slot": candidate.preferred_slot_label,
                "Suggested Start": _fmt(proposal.suggested_start),
                "Suggested End": _fmt(proposal.suggested_end),
                "Status": proposal.status,
                "Note": proposal.note,
            }
        )
    return pd.DataFrame(rows)


def _parse_datetime_value(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    try:
        parsed = pd.to_datetime(value)
        return parsed.to_pydatetime() if hasattr(parsed, "to_pydatetime") else parsed
    except Exception:
        return None


def _apply_proposal_edits(
    proposals: List[ScheduleProposal], editor_df: Optional[pd.DataFrame]
) -> List[ScheduleProposal]:
    if not isinstance(editor_df, pd.DataFrame) or editor_df.empty or "proposal_id" not in editor_df:
        return proposals

    lookup = editor_df.set_index("proposal_id")
    updated: List[ScheduleProposal] = []
    for idx, proposal in enumerate(proposals):
        if idx not in lookup.index:
            updated.append(proposal)
            continue
        row = lookup.loc[idx]
        updated.append(
            proposal.with_updates(
                start=_parse_datetime_value(row.get("Suggested Start")),
                end=_parse_datetime_value(row.get("Suggested End")),
                status=str(row.get("Status")) if row.get("Status") else proposal.status,
                note=str(row.get("Note")) if row.get("Note") else proposal.note,
            )
        )
    return updated


def handle_google_schedule_plan(
    pipeline: SchedulerPipeline,
    rows: Optional[List[SheetRow]] = None,
) -> List[ScheduleProposal]:
    base_rows = rows if rows is not None else st.session_state.get("google_form_rows")
    if isinstance(base_rows, list) and not base_rows:
        base_rows = None
    with st.spinner("Planning interview schedule..."):
        proposals = pipeline.plan_schedule(base_rows)

    st.session_state["schedule_proposals"] = proposals
    proposals_df = _proposals_to_dataframe(proposals)
    st.session_state["schedule_proposals_df"] = proposals_df
    st.session_state["interview_schedule_df"] = proposals_df
    st.session_state["confirmed_schedule_df"] = None

    ready = sum(1 for proposal in proposals if proposal.status.lower() == "ready")
    st.session_state["ready_count"] = ready

    if proposals:
        st.success(f"Generated {len(proposals)} proposal(s) ({ready} ready).")
    else:
        st.info("No pending candidates require scheduling.")
    return proposals


def _scheduled_to_dataframe(scheduled: List[ScheduledInterview]) -> pd.DataFrame:
    if not scheduled:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for record in scheduled:
        rows.append(
            {
                "Candidate": record.candidate.name,
                "Email": record.candidate.email,
                "Start": record.start.isoformat(),
                "End": record.end.isoformat(),
                "Meet Link": record.calendar_event.hangout_link or "",
                "Event ID": record.calendar_event.event_id,
                "Status": "Scheduled",
            }
        )
    return pd.DataFrame(rows)


def handle_google_schedule_finalize(
    pipeline: SchedulerPipeline,
    config: DashboardConfig,
) -> List[ScheduledInterview]:
    proposals = st.session_state.get("schedule_proposals") or []
    if not proposals:
        st.warning("Generate schedule proposals before finalizing.")
        return []

    editor_df = st.session_state.get("schedule_proposals_df")
    updated = _apply_proposal_edits(proposals, editor_df)

    interviewer_email = config.interviewer_email or pipeline.settings.default_interviewer_email
    if not interviewer_email:
        st.error("Set an interviewer email in the configuration panel.")
        return []

    with st.spinner("Creating calendar events and sending confirmations via Gmail..."):
        scheduled = pipeline.finalize_schedule(updated, interviewer_email=interviewer_email)

    if scheduled:
        st.success(f"Scheduled {len(scheduled)} interview(s).")
        scheduled_df = _scheduled_to_dataframe(scheduled)
        st.session_state["scheduled_results"] = scheduled
        st.session_state["interview_schedule_df"] = scheduled_df
        st.session_state["confirmed_schedule_df"] = scheduled_df
        st.session_state["schedule_proposals"] = []
        st.session_state["schedule_proposals_df"] = None
        st.session_state["ready_count"] = 0
    else:
        st.info("No proposals were ready to schedule. Adjust statuses or time slots and try again.")
    return scheduled


def format_skills(column: Iterable[str]) -> str:
    if not column:
        return "—"
    return ", ".join(sorted({skill.title() for skill in column}))


def build_slot_options(
    df: pd.DataFrame,
    email_column: str,
    start_column: str,
    end_column: str,
) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    labels: List[str] = []
    mapping: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        email = str(row[email_column]).strip()
        start = str(row[start_column]).strip()
        end = str(row[end_column]).strip()
        if not email or not start or not end:
            continue
        label = f"{email}  •  {start} → {end}"
        labels.append(label)
        mapping[label] = {"email": email, "start": start, "end": end}
    return labels, mapping


def handle_jd_submission(title: str, text: str, reference_files: Optional[List[Any]] = None) -> None:
    text = text.strip()
    if not text:
        st.error("Please provide job description text before saving.")
        return
    
    # Process reference resumes if provided
    reference_filenames = []
    reference_df = None
    if reference_files:
        if len(reference_files) > 2:
            st.warning("⚠️ More than 2 reference resumes uploaded. Using first 2 only.")
            reference_files = reference_files[:2]
        
        # Parse reference resumes
        parser = ResumeParserLLM(compute_embeddings=True)
        with st.spinner("Processing reference resumes..."):
            with TemporaryDirectory() as temp_dir:
                for uploaded in reference_files:
                    target_path = os.path.join(temp_dir, uploaded.name)
                    with open(target_path, "wb") as target:
                        target.write(uploaded.getbuffer())
                    reference_filenames.append(uploaded.name)
                reference_df = parser.parse_folder(temp_dir)
    
    library = get_jd_library()
    jd_id = str(uuid4())
    title = title.strip() or text.splitlines()[0][:60] or f"Job {len(library) + 1}"
    
    jd_entry = {
        "id": jd_id,
        "title": title,
        "text": text,
        "reference_filenames": reference_filenames,
        "reference_data": reference_df.to_dict() if reference_df is not None else None,
    }
    
    library.insert(0, jd_entry)
    set_selected_jd(jd_id)
    
    success_msg = "✅ Job description saved!"
    if reference_filenames:
        success_msg += f" ({len(reference_filenames)} reference resume(s) included)"
    st.success(success_msg)


# ---------------------------------------------------------------------------
# Streamlit interaction handlers
# ---------------------------------------------------------------------------


def handle_resumes_parse(resume_files: List[Any], compute_embeddings: bool) -> Optional[pd.DataFrame]:
    if not resume_files:
        st.error("Upload at least one resume PDF.")
        return None
    parser = ResumeParserLLM(compute_embeddings=compute_embeddings)
    with TemporaryDirectory() as temp_dir:
        for uploaded in resume_files:
            target_path = os.path.join(temp_dir, uploaded.name)
            with open(target_path, "wb") as target:
                target.write(uploaded.getbuffer())
        parsed_df = parser.parse_folder(temp_dir)
    st.session_state["parsed_resumes"] = parsed_df
    st.success(f"Parsed {len(parsed_df)} resumes.")
    return parsed_df


def handle_ranking(selected_jd: Dict[str, str]) -> Optional[pd.DataFrame]:
    parsed_df: Optional[pd.DataFrame] = st.session_state.get("parsed_resumes")
    if parsed_df is None or parsed_df.empty:
        st.error("Parse resumes first before ranking candidates.")
        return None
    
    # Get reference resumes from the selected JD (if available)
    reference_df: Optional[pd.DataFrame] = None
    reference_data = selected_jd.get("reference_data")
    if reference_data is not None:
        try:
            reference_df = pd.DataFrame.from_dict(reference_data)
        except Exception as exc:
            logger.warning(f"Could not load reference resumes: {exc}")
    
    try:
        matcher = load_resume_matcher()
    except RuntimeError as exc:
        st.error(f"Unable to load embedding model: {exc}")
        logger.warning("Failed to initialize embedding model: %s", exc)
        return None
    
    try:
        # Pass reference resumes to the matcher if available
        if reference_df is not None and not reference_df.empty:
            ref_filenames = selected_jd.get("reference_filenames", [])
            ref_names = ", ".join(ref_filenames) if ref_filenames else f"{len(reference_df)} reference(s)"
            st.info(f"🎯 Using reference resumes: {ref_names}")
        ranked = matcher.rank(
            selected_jd["text"], 
            parsed_df, 
            top_k=len(parsed_df),
            reference_resumes_df=reference_df
        )
    except RuntimeError as exc:
        st.error(f"Unable to compute embeddings: {exc}")
        logger.warning("Embedding computation failed: %s", exc)
        return None
    except Exception as exc:
        st.error(f"Unable to rank candidates right now: {exc}")
        logger.exception("Unexpected error during resume ranking")
        return None
    
    if "skills" in ranked.columns:
        ranked["skills"] = ranked["skills"].apply(format_skills)
    st.session_state.setdefault("ranked_results", {})[selected_jd["id"]] = ranked
    st.session_state.setdefault("filtered_ranked_results", {})[selected_jd["id"]] = None
    
    return ranked





def handle_scheduling(
    selections: List[str],
    slot_mapping: Dict[str, Dict[str, str]],
    config: DashboardConfig,
) -> List[Dict[str, Any]]:
    if not selections:
        st.warning("Select at least one availability slot.")
        return []
    if not config.interviewer_email:
        st.error("Provide an interviewer email in the pipeline configuration panel.")
        return []

    try:
        credentials = GoogleCalendarCredentials.from_env()
    except RuntimeError as exc:
        st.error(f"Google Calendar credentials missing: {exc}")
        return []

    confirmations: List[Dict[str, Any]] = []
    for label in selections:
        slot = slot_mapping.get(label)
        if not slot:
            continue
        subject = config.subject_template.format(candidate=slot["email"])
        try:
            confirmation = schedule_interview(
                candidate_email=slot["email"],
                interviewer_email=config.interviewer_email,
                time_slot=(slot["start"], slot["end"]),
                subject=subject,
                description=config.description,
                location=config.location,
                config=credentials,
                send_email=True,
            )
            confirmations.append(confirmation)
        except Exception as exc:  # pragma: no cover - network dependent
            st.error(f"Failed to schedule {slot['email']}: {exc}")
    if confirmations:
        st.success(f"Scheduled {len(confirmations)} interview(s).")
    return confirmations





def handle_send_forms(
    selected_candidates: List[dict],
    job_title: str,
    config: DashboardConfig,
    forms_link: Optional[str] = None,
) -> List[dict]:
    """Send Google Form invitations via Gmail."""
    if not selected_candidates:
        st.warning("Select at least one candidate before sending forms.")
        return []

    pipeline = get_scheduler_pipeline()
    if not pipeline:
        return []

    link = forms_link or config.google_form_link or pipeline.settings.google_form_link
    if not link:
        st.error("Google Form link is not configured. Set GOOGLE_FORM_LINK in your .env file or the config panel.")
        return []

    with st.spinner("Sending Google Form invitations via Gmail..."):
        try:
            results = pipeline.send_form_invitations(
                candidates=selected_candidates,
                job_title=job_title,
                forms_link=link,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            st.error(f"Error sending invitations: {exc}")
            logger.exception("Failed to send forms to candidates")
            return []

    successes = [res for res in results if res.get("status") == "sent"]
    failures = [res for res in results if res.get("status") == "failed"]
    skipped = [res for res in results if res.get("status") == "skipped"]

    if successes:
        st.success(f"✅ Successfully emailed {len(successes)} candidate(s)")
    if failures:
        st.error(f"❌ Failed to send to {len(failures)} candidate(s)")
        for result in failures:
            st.error(
                f"  • {result.get('name', 'Unknown')} ({result.get('email', 'N/A')}): {result.get('error', 'Unknown error')}"
            )
    if skipped:
        st.warning(f"⚠️ Skipped {len(skipped)} candidate(s) due to missing email address")

    return results


# ---------------------------------------------------------------------------
# UI Sections
# ---------------------------------------------------------------------------


def render_header() -> None:
    jd_count = len(st.session_state.get("jd_library", []))
    parsed_df = st.session_state.get("parsed_resumes")
    resume_count = int(parsed_df.shape[0]) if isinstance(parsed_df, pd.DataFrame) else 0
    new_responses = int(st.session_state.get("new_responses_count", 0))
    
    # Get ready count from session state (set when plan interviews is clicked)
    ready_count = int(st.session_state.get("ready_count", 0))
    
    # If not set, try to count from schedule dataframe as fallback
    if ready_count == 0:
        schedule_df = st.session_state.get("interview_schedule_df")
        if isinstance(schedule_df, pd.DataFrame):
            status_series = schedule_df.get("Status")
            if status_series is not None:
                ready_count = int((status_series == "Ready").sum())
            else:
                ready_count = int(schedule_df.shape[0])

    jd_count_display = f"{jd_count:,}"
    resume_count_display = f"{resume_count:,}"
    new_responses_display = f"{new_responses:,}"
    ready_count_display = f"{ready_count:,}"

    st.markdown(
        f"""
        <div class="hero-card">
            <img src="data:image/jpeg;base64,{get_itc_logo_base64()}" alt="ITC Infotech" class="itc-logo-corner" />
            <div class="hero-brand-container">
                <img src="data:image/png;base64,{get_logo_base64()}" alt="Logo" class="hero-logo" />
                <span class="hero-brand">TalentFlow AI</span>
            </div>
            <div class="hero-grid">
                <div>
                    <h1 class="hero-title">Build <strong>brilliant</strong> hiring journeys</h1>
                    <p class="hero-subtitle">
                        Curate roles, rank talent, and schedule interviews—all in one place. Your one-stop solution for smarter, faster hiring.
                    </p>
                    <div class="hero-actions">
                        <div class="hero-step"><span>1</span>Curate roles</div>
                        <div class="hero-step"><span>2</span>Rank resumes</div>
                        <div class="hero-step"><span>3</span>Book interviews</div>
                    </div>
                </div>
                <div class="hero-badges">
                    <div class="hero-badge">
                        <h4>Session snapshot</h4>
                        <p><strong>{jd_count_display}</strong> job description(s) saved this session.</p>
                        <p><strong>{resume_count_display}</strong> resume(s) parsed and ready to compare.</p>
                    </div>
                    <div class="hero-badge hero-badge--accent">
                        <h4>Next best actions</h4>
                        <ul>
                            <li>Review <strong>{new_responses_display}</strong> new availability response(s).</li>
                            <li>Confirm <strong>{ready_count_display}</strong> interview invite(s) awaiting send.</li>
                            <li>Use filters to spotlight top candidates instantly.</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="stats-row">
            <div class="stat-card">
                <span class="stat-label">Roles curated</span>
                <span class="stat-value">{jd_count_display}</span>
                <span class="stat-footnote">Saved to this workspace</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Resumes parsed</span>
                <span class="stat-value">{resume_count_display}</span>
                <span class="stat-footnote">Ready for ranking</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">New form responses</span>
                <span class="stat-value">{new_responses_display}</span>
                <span class="stat-footnote">Awaiting slot assignment</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Interviews ready</span>
                <span class="stat-value">{ready_count_display}</span>
                <span class="stat-footnote">Can be sent right now</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_config_panel() -> DashboardConfig:
    with st.expander("⚙️ Pipeline configuration", expanded=False):
        st.caption(
            "Adjust these settings to control how invitations are composed and where availability is fetched from."
        )
        model_name = get_embedding_model()
        st.info(
            "Using sentence-transformers embedding model: "
            f"`{model_name}`. Override via the EMBEDDING_MODEL environment variable."
        )
        col1, col2 = st.columns(2)
        with col1:
            form_id = st.text_input(
                "OneDrive Form ID (legacy)",
                value=os.getenv("MS_FORM_ID", ""),
                key="config_form_id",
                help="Only used for the legacy Microsoft/OneDrive workflow.",
            )
            google_form_link = st.text_input(
                "Google Form link",
                value=os.getenv("GOOGLE_FORM_LINK", ""),
                key="config_google_form_link",
                placeholder="https://forms.gle/...",
                help="Used when emailing availability requests via Gmail.",
            )
            interviewer_email = st.text_input(
                "Interviewer Email",
                value=os.getenv("DEFAULT_INTERVIEWER_EMAIL", ""),
                key="config_interviewer_email",
            )
            location = st.text_input(
                "Meeting location",
                value=os.getenv("INTERVIEW_LOCATION", "Microsoft Teams Meeting"),
                key="config_location",
            )
        with col2:
            subject_template = st.text_input(
                "Subject template",
                value=os.getenv("INTERVIEW_SUBJECT", "Interview with {candidate}"),
                key="config_subject_template",
                help="Use {candidate} as a placeholder for the candidate email or name.",
            )
            description = st.text_area(
                "Invitation message",
                value=os.getenv(
                    "INTERVIEW_DESCRIPTION",
                    "Looking forward to discussing the opportunity with you!",
                ),
                key="config_description",
                height=140,
            )
            st.caption(
                "We'll pull environment defaults if present, but you can override them here for this session."
            )
    return DashboardConfig(
        form_id=form_id or None,
        google_form_link=google_form_link or None,
        interviewer_email=interviewer_email,
        location=location,
        subject_template=subject_template,
        description=description,
    )


def render_jd_page() -> None:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("🧾 Step 1 · Curate job descriptions")
    
    # Initialize form key and enhanced JD storage if not exists
    if "jd_form_key" not in st.session_state:
        st.session_state["jd_form_key"] = 0
    if "enhanced_jd_text" not in st.session_state:
        st.session_state["enhanced_jd_text"] = ""
    if "jd_title_text" not in st.session_state:
        st.session_state["jd_title_text"] = ""
    
    # Title field (outside form for consistency)
    title = st.text_input("Title", placeholder="AI Engineer", value=st.session_state["jd_title_text"], key=f"title_{st.session_state['jd_form_key']}")
    st.session_state["jd_title_text"] = title
    
    # Job description with enhance button
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("**Job description**")
    with col2:
        enhance_clicked = st.button("✨ Enhance", key=f"enhance_{st.session_state['jd_form_key']}")
    
    # Handle enhance button click BEFORE rendering text area
    if enhance_clicked:
        # Get current text from the text area
        current_text = st.session_state.get(f"jd_text_{st.session_state['jd_form_key']}", "")
        if not current_text.strip():
            st.warning("⚠️ Please enter a job description first before enhancing.")
        else:
            with st.spinner("✨ Enhancing job description with AI..."):
                enhanced = enhance_job_description(current_text, title)
                if enhanced:
                    # Update the session state so text area shows enhanced version
                    st.session_state["enhanced_jd_text"] = enhanced
                    # Increment form key to force text area to refresh with new value
                    st.session_state["jd_form_key"] += 1
                    st.success("✅ Job description enhanced! The text has been updated above.")
                    st.rerun()
    
    # Render text area with current enhanced text or empty
    jd_text = st.text_area(
        "Job description",
        placeholder="Paste or type the job description here...",
        value=st.session_state.get("enhanced_jd_text", ""),
        height=200,
        key=f"jd_text_{st.session_state['jd_form_key']}",
        label_visibility="collapsed",
    )
    
    # Update session state with any manual edits
    if jd_text != st.session_state.get("enhanced_jd_text", ""):
        st.session_state["enhanced_jd_text"] = jd_text
    
    with st.form(f"jd_form_{st.session_state['jd_form_key']}"):
        st.markdown("---")
        st.markdown("**⭐ Reference resumes (optional)**")
        st.caption("Upload 1-2 resumes from previously hired candidates for this role. These will be used to improve ranking.")
        
        reference_files = st.file_uploader(
            "Reference resumes from successful hires",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload resumes of people who excelled in similar roles. Max 2 recommended.",
            key=f"reference_upload_{st.session_state['jd_form_key']}",
        )
        
        submitted = st.form_submit_button("💾 Save job description", use_container_width=True)
        if submitted:
            handle_jd_submission(title, jd_text, list(reference_files or []))
            # Reset form state
            st.session_state["jd_form_key"] += 1
            st.session_state["enhanced_jd_text"] = ""
            st.session_state["jd_title_text"] = ""
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

    library = get_jd_library()
    if not library:
        st.info("Saved job descriptions will appear here. Add one to get started.")
        return

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📚 Saved job descriptions")
    remove_ids: List[str] = []
    selected = st.session_state.get("selected_jd_id")
    for jd in library:
        with st.expander(jd["title"], expanded=jd["id"] == selected):
            st.write(jd["text"])
            
            # Show reference resumes if available
            reference_filenames = jd.get("reference_filenames", [])
            if reference_filenames:
                st.markdown("---")
                st.caption(f"**📎 Reference resumes:** {', '.join(reference_filenames)}")
            
            col_select, col_delete = st.columns(2)
            if col_select.button("Use this JD", key=f"select_{jd['id']}"):
                set_selected_jd(jd["id"])
            if col_delete.button("Remove", key=f"delete_{jd['id']}"):
                remove_ids.append(jd["id"])
    if remove_ids:
        st.session_state["jd_library"] = [jd for jd in library if jd["id"] not in remove_ids]
        if selected in remove_ids:
            remaining = st.session_state["jd_library"]
            set_selected_jd(remaining[0]["id"] if remaining else None)
    st.markdown('</div>', unsafe_allow_html=True)


def render_ranking_page() -> None:
    library = get_jd_library()
    if not library:
        st.warning("Add at least one job description in the JD page before ranking resumes.")
        return

    selected = get_selected_jd() or library[0]
    options = {jd["title"]: jd for jd in library}
    chosen_title = st.selectbox(
        "Job description",
        options=list(options.keys()),
        index=list(options.keys()).index(selected["title"]),
    )
    selected = options[chosen_title]
    set_selected_jd(selected["id"])

    parsed_df = st.session_state.get("parsed_resumes")
    resume_count = int(parsed_df.shape[0]) if isinstance(parsed_df, pd.DataFrame) else 0

    min_years_key = f"min_years_filter_{selected['id']}"
    max_years_key = f"max_years_filter_{selected['id']}"
    years_range_key = f"years_range_filter_{selected['id']}"
    min_edu_key = f"min_education_filter_{selected['id']}"
    st.session_state.setdefault(min_years_key, 3.0)
    st.session_state.setdefault(max_years_key, 15.0)
    st.session_state.setdefault(years_range_key, (3.0, 15.0))
    st.session_state.setdefault(min_edu_key, EDUCATION_TIERS[0])
    st.session_state.setdefault("filtered_ranked_results", {})

    st.markdown("## 🚀 Resume Intelligence Board")
    
    # Unified control panel - parsing + filters in one compact row
    st.markdown("#### Controls & filters")
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1, 0.8])
    
    with ctrl_col1:
        st.markdown("**Role context**")
        st.text_area(
            "Selected job description",
            selected["text"],
            height=140,
            disabled=True,
            label_visibility="collapsed",
        )
        if selected.get("reference_filenames"):
            st.caption("References: " + ", ".join(selected.get("reference_filenames", [])))
    
    with ctrl_col2:
        st.markdown("**Resume intake**")
        resume_files = st.file_uploader(
            "Upload PDF resumes",
            type=["pdf"],
            accept_multiple_files=True,
            help="Drop multiple resumes.",
            key="resume_upload",
        )
        compute_embeddings = st.toggle("Cache embeddings", value=False)
        
        if isinstance(parsed_df, pd.DataFrame) and not parsed_df.empty:
            st.caption(f"📥 {resume_count} resume(s) ready")
        elif resume_count == 0:
            st.caption("No resumes parsed yet")
        else:
            st.caption(f"📥 {resume_count} resume(s) ready")
    
    with ctrl_col3:
        st.markdown("**Filters**")
        years_range = st.slider(
            "Experience (yrs)",
            min_value=0.0,
            max_value=40.0,
            value=st.session_state[years_range_key],
            step=0.5,
            key=years_range_key,
        )
        min_years, max_years = years_range
        st.session_state[min_years_key] = min_years
        st.session_state[max_years_key] = max_years
        
        default_edu = st.session_state[min_edu_key]
        edu_index = EDUCATION_TIERS.index(default_edu) if default_edu in EDUCATION_TIERS else 0
        min_education = st.selectbox(
            "Min education",
            options=EDUCATION_TIERS,
            index=edu_index,
            key=min_edu_key,
        )
    
    # Action buttons row - aligned horizontally
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        parse_clicked = st.button("📄 Parse resumes", use_container_width=True)
        if parse_clicked:
            handle_resumes_parse(list(resume_files or []), compute_embeddings)
            parsed_df = st.session_state.get("parsed_resumes")
            resume_count = int(parsed_df.shape[0]) if isinstance(parsed_df, pd.DataFrame) else 0
    with btn_col2:
        rank_clicked = st.button("⚡ Rank candidates", type="primary", use_container_width=True)
        if rank_clicked:
            handle_ranking(selected)

    ranked_map = st.session_state.get("ranked_results", {})
    ranked_df = ranked_map.get(selected["id"])

    if isinstance(ranked_df, pd.DataFrame) and not ranked_df.empty:
        filtered_df = filter_ranked_candidates(ranked_df, min_years, max_years, min_education)
        st.session_state["filtered_ranked_results"][selected["id"]] = filtered_df

        eligible_total = int(filtered_df["meets_all_criteria"].sum())
        raw_avg = (
            filtered_df.loc[filtered_df["meets_all_criteria"], "match_score"].mean()
            if eligible_total
            else filtered_df["match_score"].head(5).mean()
        )
        avg_score = 0.0
        if raw_avg is not None and not pd.isna(raw_avg):
            avg_score = float(raw_avg)
        avg_pct = compute_match_percentage(avg_score)
        top_match = filtered_df.iloc[0] if not filtered_df.empty else None

        chip_col1, chip_col2, chip_col3 = st.columns(3)
        chip_col1.metric("Total candidates", len(filtered_df))
        chip_col2.metric("Avg match score", f"{avg_pct:.1f}%")
        chip_col3.metric(
            "Top resume",
            top_match.get("name", "—") if top_match is not None else "—",
            help="Highest ranked profile for the current filters",
        )

        table_df = filtered_df.copy()
        table_df["Match Score (%)"] = (
            table_df["match_score"].apply(compute_match_percentage).round(1)
        )
        table_df["Years of Experience"] = table_df["computed_years"].round(1)
        table_df["Education"] = table_df["education_label"]
        display_columns = [
            "name",
            "email",
            "Match Score (%)",
            "Years of Experience",
            "Education",
            "skills",
        ]

        st.markdown("#### Ranked candidates")
        st.caption(f"Filters applied · {min_years}–{max_years} yrs · {min_education}")
        
        # Full-width table first
        st.dataframe(
            table_df[display_columns].head(15),
            use_container_width=True,
            height=280,
        )
        
        dl_col1, dl_col2 = st.columns([1, 3])
        with dl_col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                data=csv,
                file_name=f"ranked_{selected['title'].replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        def _normalize_list(value: Any) -> List[str]:
            if value is None:
                return []
            if isinstance(value, str):
                return [item.strip() for item in value.split(",") if item.strip() and item.strip() != "—"]
            if isinstance(value, Iterable):
                return [str(item).strip() for item in value if str(item).strip()]
            return []

        def render_candidate_details(row: pd.Series) -> None:
            """Render modern dashboard with improved visual design using Streamlit components."""
            skills_list = _normalize_list(row.get("skills"))
            experience_list = _normalize_list(row.get("experience_list"))
            candidate_name = row.get("name", "Candidate")
            email = row.get("email", "")
            phone = row.get("phone", "")
            summary = row.get("summary") or row.get("profile_summary") or ""
            match_score = row.get("match_score") or 0
            experience_years = row.get("computed_years") or 0
            education_label = row.get("education_label") or "—"

            jd_text = selected.get("description", selected.get("text", ""))
            cache_key = f"insights_cache_{candidate_name}_{selected.get('title', '')}_{match_score}"
            if cache_key in st.session_state:
                insights = st.session_state[cache_key]
            else:
                with st.spinner("🔍 Generating comprehensive insights..."):
                    insights = generate_comprehensive_candidate_insights(
                        candidate_name=candidate_name,
                        jd_title=selected.get("title", "this role"),
                        jd_text=jd_text,
                        skills=skills_list,
                        experience=experience_list,
                        education=education_label,
                        match_score=match_score,
                        years_of_experience=experience_years,
                        summary=summary,
                    )
                st.session_state[cache_key] = insights

            if insights.get("error"):
                st.error(f"⚠️ {insights.get('overall_insight', 'Unable to generate insights')}")
                return

            core_profile = insights.get("core_profile", {})
            match_pct = compute_match_percentage(match_score)
            insight_text = insights.get("overall_insight", "No narrative available.")
            ctc_data = insights.get("ctc_prediction", {})
            skill_clusters = insights.get("skill_clusters", {})
            timeline = insights.get("career_timeline", [])
            strengths = insights.get("strengths", [])
            weaknesses = insights.get("weaknesses", [])
            match_breakdown = insights.get("jd_match_breakdown", {})

            # Profile Hero Card
            circle_circumference = 351.86
            circle_offset = circle_circumference * (1 - match_pct / 100)
            
            st.markdown(f"""
            <div style="background: linear-gradient(to right, #2563eb, #4f46e5); border-radius: 1rem; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1); padding: 2rem; margin-bottom: 2rem; color: white;">
                <div style="display: flex; align-items: start; justify-content: space-between; flex-wrap: wrap; gap: 1.5rem;">
                    <div style="flex: 1; min-width: 300px;">
                        <h2 style="font-size: 1.875rem; font-weight: 700; margin: 0 0 0.5rem 0; color: white;">{candidate_name}</h2>
                        <p style="color: #bfdbfe; font-size: 1.125rem; margin: 0 0 1rem 0;">{core_profile.get('current_role', 'N/A')}</p>
                        <div style="display: flex; flex-wrap: wrap; gap: 1rem; font-size: 0.875rem;">
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <span>📍</span>
                                <span>{core_profile.get('location', 'N/A')}</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <span>✉️</span>
                                <span>{email or 'Not provided'}</span>
                            </div>
                            {f'<div style="display: flex; align-items: center; gap: 0.5rem;"><span>📞</span><span>{phone}</span></div>' if phone else ''}
                        </div>
                    </div>
                    <div style="text-align: center;">
                        <div style="position: relative; width: 8rem; height: 8rem;">
                            <svg style="width: 8rem; height: 8rem; transform: rotate(-90deg);">
                                <circle cx="64" cy="64" r="56" stroke="rgba(255,255,255,0.2)" stroke-width="8" fill="none" />
                                <circle cx="64" cy="64" r="56" stroke="white" stroke-width="8" fill="none" 
                                    stroke-dasharray="{circle_circumference}" 
                                    stroke-dashoffset="{circle_offset}" 
                                    stroke-linecap="round" />
                            </svg>
                            <div style="position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                                <span style="font-size: 2.5rem; font-weight: 700; color: white;">{int(match_pct)}%</span>
                                <span style="font-size: 0.875rem; color: #bfdbfe;">Match</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Quick Stats using Streamlit columns
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.markdown(f"""
                <div style="background: white; border-radius: 0.75rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); padding: 1.5rem; border: 1px solid #e2e8f0;">
                    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                        <div style="width: 2.5rem; height: 2.5rem; background: #dbeafe; border-radius: 0.5rem; display: flex; align-items: center; justify-content: center;">
                            <span style="font-size: 1.25rem;">💼</span>
                        </div>
                        <span style="color: #64748b; font-size: 0.875rem;">Experience</span>
                    </div>
                    <p style="font-size: 1.5rem; font-weight: 700; color: #1e293b; margin: 0;">{core_profile.get('total_experience', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with stat_col2:
                st.markdown(f"""
                <div style="background: white; border-radius: 0.75rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); padding: 1.5rem; border: 1px solid #e2e8f0;">
                    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                        <div style="width: 2.5rem; height: 2.5rem; background: #f3e8ff; border-radius: 0.5rem; display: flex; align-items: center; justify-content: center;">
                            <span style="font-size: 1.25rem;">📈</span>
                        </div>
                        <span style="color: #64748b; font-size: 0.875rem;">Level</span>
                    </div>
                    <p style="font-size: 1.5rem; font-weight: 700; color: #1e293b; margin: 0;">{core_profile.get('seniority_level', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with stat_col3:
                st.markdown(f"""
                <div style="background: white; border-radius: 0.75rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); padding: 1.5rem; border: 1px solid #e2e8f0;">
                    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                        <div style="width: 2.5rem; height: 2.5rem; background: #d1fae5; border-radius: 0.5rem; display: flex; align-items: center; justify-content: center;">
                            <span style="font-size: 1.25rem;">🧠</span>
                        </div>
                        <span style="color: #64748b; font-size: 0.875rem;">Domain</span>
                    </div>
                    <p style="font-size: 1.5rem; font-weight: 700; color: #1e293b; margin: 0;">{core_profile.get('primary_domain', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with stat_col4:
                st.markdown(f"""
                <div style="background: white; border-radius: 0.75rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); padding: 1.5rem; border: 1px solid #e2e8f0;">
                    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                        <div style="width: 2.5rem; height: 2.5rem; background: #fef3c7; border-radius: 0.5rem; display: flex; align-items: center; justify-content: center;">
                            <span style="font-size: 1.25rem;">🎓</span>
                        </div>
                        <span style="color: #64748b; font-size: 0.875rem;">Education</span>
                    </div>
                    <p style="font-size: 1.5rem; font-weight: 700; color: #1e293b; margin: 0;">{core_profile.get('education_summary', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Main Content - Three columns
            main_col1, main_col2, main_col3 = st.columns(3)
            
            # LEFT COLUMN - Fit Analysis & CTC
            with main_col1:
                # Fit Analysis
                skill_match = match_breakdown.get("skill_match", 0)
                domain_match = match_breakdown.get("domain_match", 0)
                exp_match = match_breakdown.get("experience_match", 0)
                lead_match = match_breakdown.get("leadership_match", 0)
                
                st.markdown(f"""
                <div style="background: white; border-radius: 0.75rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); padding: 1.5rem; border: 1px solid #e2e8f0; margin-bottom: 1.5rem;">
                    <h3 style="font-size: 1.125rem; font-weight: 600; color: #1e293b; margin: 0 0 1rem 0;">Fit Analysis</h3>
                    <div style="margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span style="font-size: 0.875rem; font-weight: 500; color: #334155;">Skill Match</span>
                            <span style="font-size: 0.875rem; font-weight: 700; color: #10b981;">{skill_match}%</span>
                        </div>
                        <div style="height: 0.5rem; background: #f1f5f9; border-radius: 9999px; overflow: hidden;">
                            <div style="height: 100%; background: linear-gradient(to right, #10b981, #059669); border-radius: 9999px; width: {skill_match}%;"></div>
                        </div>
                    </div>
                    <div style="margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span style="font-size: 0.875rem; font-weight: 500; color: #334155;">Domain Match</span>
                            <span style="font-size: 0.875rem; font-weight: 700; color: #10b981;">{domain_match}%</span>
                        </div>
                        <div style="height: 0.5rem; background: #f1f5f9; border-radius: 9999px; overflow: hidden;">
                            <div style="height: 100%; background: linear-gradient(to right, #10b981, #059669); border-radius: 9999px; width: {domain_match}%;"></div>
                        </div>
                    </div>
                    <div style="margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span style="font-size: 0.875rem; font-weight: 500; color: #334155;">Experience Match</span>
                            <span style="font-size: 0.875rem; font-weight: 700; color: #eab308;">{exp_match}%</span>
                        </div>
                        <div style="height: 0.5rem; background: #f1f5f9; border-radius: 9999px; overflow: hidden;">
                            <div style="height: 100%; background: linear-gradient(to right, #eab308, #ca8a04); border-radius: 9999px; width: {exp_match}%;"></div>
                        </div>
                    </div>
                    <div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span style="font-size: 0.875rem; font-weight: 500; color: #334155;">Leadership Match</span>
                            <span style="font-size: 0.875rem; font-weight: 700; color: #f97316;">{lead_match}%</span>
                        </div>
                        <div style="height: 0.5rem; background: #f1f5f9; border-radius: 9999px; overflow: hidden;">
                            <div style="height: 100%; background: linear-gradient(to right, #f97316, #ea580c); border-radius: 9999px; width: {lead_match}%;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # CTC Card
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border-radius: 0.75rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); padding: 1.5rem; border: 1px solid #6ee7b7; margin-bottom: 1.5rem;">
                    <h3 style="font-size: 1.125rem; font-weight: 600; color: #1e293b; margin: 0 0 0.25rem 0;">💰 Predicted CTC</h3>
                    <p style="font-size: 0.875rem; color: #64748b; margin: 0 0 1rem 0;">Based on skills and experience</p>
                    <div style="font-size: 1.875rem; font-weight: 700; color: #047857; margin-bottom: 1rem;">{ctc_data.get('range', 'N/A')}</div>
                    <p style="font-size: 0.75rem; color: #64748b; line-height: 1.4; margin: 0;">{ctc_data.get('rationale', '')[:150]}...</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Missing Skills
                missing_skills = match_breakdown.get("missing_critical_skills", [])
                missing_items = "".join([
                    f'<li style="font-size: 0.875rem; color: #334155; margin-bottom: 0.5rem;">• {skill}</li>'
                    for skill in missing_skills[:3]
                ])
                
                st.markdown(f"""
                <div style="background: #fef3c7; border-radius: 0.75rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); padding: 1.5rem; border: 1px solid #fde68a;">
                    <h3 style="font-size: 1.125rem; font-weight: 600; color: #1e293b; margin: 0 0 0.75rem 0;">⚠️ Development Areas</h3>
                    <ul style="list-style: none; padding: 0; margin: 0;">
                        {missing_items if missing_items else '<li style="font-size: 0.875rem; color: #64748b;">No critical gaps identified</li>'}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # CENTER COLUMN - Skills
            with main_col2:
                # Build complete skills HTML as one block
                skill_configs = {
                    "technical_skills": {"label": "Technical Skills", "bg": "#dbeafe", "color": "#1e40af", "bar": "#3b82f6"},
                    "domain_skills": {"label": "Domain Skills", "bg": "#f3e8ff", "color": "#7e22ce", "bar": "#a855f7"},
                    "management_skills": {"label": "Management Skills", "bg": "#fef3c7", "color": "#92400e", "bar": "#f59e0b"},
                    "tools": {"label": "Tools & Frameworks", "bg": "#ccfbf1", "color": "#115e59", "bar": "#14b8a6"},
                }
                
                skills_content = ""
                for category, skills in skill_clusters.items():
                    if not skills:
                        continue
                    config = skill_configs.get(category, skill_configs["technical_skills"])
                    skill_count = len(skills)
                    width_pct = min(100, (skill_count / 10) * 100)
                    
                    # Build skill tags as plain text without nested divs
                    skill_tags = " ".join([
                        f'<span style="padding:0.25rem 0.75rem;background:{config["bg"]};color:{config["color"]};border-radius:9999px;font-size:0.75rem;font-weight:500;display:inline-block;margin:0.25rem;">{skill}</span>'
                        for skill in skills[:6]
                    ])
                    
                    skills_content += f'''
                    <div style="margin-bottom:1.5rem;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;">
                            <span style="font-size:0.875rem;font-weight:500;color:#334155;">{config["label"]}</span>
                            <span style="font-size:0.875rem;font-weight:700;color:{config["color"]};">{skill_count}</span>
                        </div>
                        <div style="height:0.75rem;background:#f1f5f9;border-radius:9999px;overflow:hidden;margin-bottom:0.75rem;">
                            <div style="height:100%;background:{config["bar"]};border-radius:9999px;width:{width_pct}%;"></div>
                        </div>
                        <div style="margin-top:0.5rem;">
                            {skill_tags}
                        </div>
                    </div>
                    '''
                
                if skills_content:
                    st.markdown(f'''
                    <div style="background:white;border-radius:0.75rem;box-shadow:0 1px 3px 0 rgba(0,0,0,0.1);padding:1.5rem;border:1px solid #e2e8f0;margin-bottom:1.5rem;">
                        <h3 style="font-size:1.125rem;font-weight:600;color:#1e293b;margin:0 0 1rem 0;">Skills Overview</h3>
                        {skills_content}
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown('''
                    <div style="background:white;border-radius:0.75rem;box-shadow:0 1px 3px 0 rgba(0,0,0,0.1);padding:1.5rem;border:1px solid #e2e8f0;margin-bottom:1.5rem;">
                        <h3 style="font-size:1.125rem;font-weight:600;color:#1e293b;margin:0 0 1rem 0;">Skills Overview</h3>
                        <p style="color:#64748b;font-size:0.875rem;">No skill data available</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Strengths
                strengths_items = "".join([
                    f'<li style="font-size: 0.875rem; color: #334155; margin-bottom: 0.5rem; display: flex; align-items: start; gap: 0.5rem;"><span style="color: #10b981;">✓</span><span>{strength}</span></li>'
                    for strength in strengths[:3]
                ])
                
                st.markdown(f"""
                <div style="background: #d1fae5; border-radius: 0.75rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); padding: 1.5rem; border: 1px solid #6ee7b7;">
                    <h3 style="font-size: 1.125rem; font-weight: 600; color: #1e293b; margin: 0 0 0.75rem 0;">💪 Key Strengths</h3>
                    <ul style="list-style: none; padding: 0; margin: 0;">
                        {strengths_items if strengths_items else '<li style="font-size: 0.875rem; color: #64748b;">Analyzing strengths...</li>'}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # RIGHT COLUMN - Timeline & AI Summary
            with main_col3:
                # Build timeline HTML as one block
                timeline_colors = ["#2563eb", "#a855f7", "#14b8a6"]
                timeline_items = []
                
                for i, entry in enumerate(timeline[:3]):
                    period = entry.get("period", "Unknown")
                    role = entry.get("role", "Unknown Role")
                    company = entry.get("company", "")
                    color = timeline_colors[i % len(timeline_colors)]
                    is_current = i == 0
                    
                    badge_html = f'<span style="font-size:0.75rem;background:{color};color:white;padding:0.25rem 0.5rem;border-radius:0.25rem;margin-left:0.5rem;">Current</span>' if is_current else ''
                    bg_color = "#eff6ff" if is_current else "#f8fafc"
                    
                    # Build each timeline item without nested f-strings
                    item_html = (
                        f'<div style="background:{bg_color};border-radius:0.5rem;padding:1rem;border:1px solid #e2e8f0;'
                        f'border-left:4px solid {color};margin-bottom:1rem;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:0.25rem;">'
                        f'<h4 style="font-weight:600;color:#1e293b;margin:0;font-size:0.9rem;">{role}</h4>'
                        f'{badge_html}'
                        f'</div>'
                        f'<p style="font-size:0.875rem;color:{color};font-weight:500;margin:0.25rem 0;">{company}</p>'
                        f'<p style="font-size:0.75rem;color:#64748b;margin:0;">{period}</p>'
                        f'</div>'
                    )
                    timeline_items.append(item_html)
                
                if timeline_items:
                    timeline_content = "".join(timeline_items)
                    full_html = (
                        '<div style="background:white;border-radius:0.75rem;box-shadow:0 1px 3px 0 rgba(0,0,0,0.1);'
                        'padding:1.5rem;border:1px solid #e2e8f0;margin-bottom:1.5rem;">'
                        '<h3 style="font-size:1.125rem;font-weight:600;color:#1e293b;margin:0 0 1rem 0;">📅 Experience Timeline</h3>'
                        f'{timeline_content}'
                        '</div>'
                    )
                    st.markdown(full_html, unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div style="background:white;border-radius:0.75rem;box-shadow:0 1px 3px 0 rgba(0,0,0,0.1);'
                        'padding:1.5rem;border:1px solid #e2e8f0;margin-bottom:1.5rem;">'
                        '<h3 style="font-size:1.125rem;font-weight:600;color:#1e293b;margin:0 0 1rem 0;">📅 Experience Timeline</h3>'
                        '<p style="color:#64748b;font-size:0.875rem;">No timeline available</p>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                
                # AI Summary
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #e0e7ff 0%, #ddd6fe 100%); border-radius: 0.75rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); padding: 1.5rem; border: 1px solid #c7d2fe;">
                    <h3 style="font-size: 1.125rem; font-weight: 600; color: #1e293b; margin: 0 0 0.75rem 0;">🤖 AI Summary</h3>
                    <p style="font-size: 0.875rem; color: #334155; line-height: 1.6; margin: 0;">{insight_text}</p>
                </div>
                """, unsafe_allow_html=True)

        # Candidate insights below the table
        insights_source = filtered_df.head(8).reset_index(drop=True)
        if not insights_source.empty:
            st.markdown("---")
            st.markdown("### 🎯 Candidate insights")
            inspect_key = f"candidate_inspect_{selected['id']}"
            option_labels = []
            for _, option_row in insights_source.iterrows():
                name = option_row.get("name") or "Candidate"
                try:
                    pct = int(round(
                        compute_match_percentage(option_row.get("match_score"))
                    ))
                except (TypeError, ValueError):
                    pct = 0
                option_labels.append(f"{name} · {pct}% match")

            selection = st.selectbox(
                "Select candidate to inspect",
                list(range(len(option_labels))),
                format_func=lambda i: option_labels[i],
                key=inspect_key,
            )
            render_candidate_details(insights_source.loc[selection])

    elif resume_count == 0:
        st.info("Parse resumes to unlock ranking.")
    else:
        st.info('Upload resumes and click "Rank candidates" to populate this board.')


def render_scheduling_page(config: DashboardConfig) -> None:
    pipeline = get_scheduler_pipeline()
    ranked_map = st.session_state.get("ranked_results", {})
    selected = get_selected_jd()
    
    # Step 3a: Show ranked candidates with checkboxes for selection
    filtered_results = st.session_state.get("filtered_ranked_results", {})
    if ranked_map and selected and selected["id"] in ranked_map:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📮 Step 3 · Select candidates to send forms")

        filtered_df = filtered_results.get(selected["id"])
        base_df = ranked_map[selected["id"]]
        candidate_df = filtered_df if (filtered_df is not None and not filtered_df.empty) else base_df
        if filtered_df is not None and filtered_df.empty:
            st.warning("Filters cleared all candidates; falling back to the original ranking.")
        candidate_df = candidate_df.head(20)

        # Resolve Google Form link
        forms_link = (
            config.google_form_link
            or os.getenv("GOOGLE_FORM_LINK", "")
            or (pipeline.settings.google_form_link if pipeline else "")
        )

        if not forms_link:
            st.warning(
                "⚠️ GOOGLE_FORM_LINK is not configured. Set it in your .env file or the configuration panel to email availability forms."
            )

        # Initialize selected candidates in session state
        if "selected_candidates" not in st.session_state:
            st.session_state["selected_candidates"] = []

        view_label = "filtered" if filtered_df is not None and not filtered_df.empty else "sorted by match score"
        st.write(f"**Top {len(candidate_df)} candidates** ({view_label})")

        # Create checkbox for each candidate
        selected_candidates = []
        for idx, row in candidate_df.iterrows():
            # Build display label
            name = row.get("name", "Unknown")
            email = row.get("email", "No email")
            score = row.get("match_score", 0)
            file_name = row.get("file_name", "")

            score_pct = int(round(compute_match_percentage(score))) if score else 0
            label = f"{name} — {score_pct}% match — {email}"

            # Create unique key for checkbox
            checkbox_key = f"candidate_select_{idx}"

            # Show checkbox
            checked = st.checkbox(label, key=checkbox_key, value=False)

            if checked:
                selected_candidates.append({
                    "name": name,
                    "email": email,
                    "file_name": file_name,
                    "match_score": score,
                })

        st.write(f"**Selected: {len(selected_candidates)} candidate(s)**")

        # Send forms button
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("📧 Send Google Form Link to Selected", width="stretch", type="primary"):
                if selected_candidates:
                    job_title = selected.get("title", "this position")
                    handle_send_forms(selected_candidates, job_title, config, forms_link=forms_link)
                else:
                    st.warning("Please select at least one candidate.")

        with col2:
            if st.button("Clear Selection", width="stretch"):
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
    elif not ranked_map:
        st.info("Rank candidates first to see which resumes align best with your job.")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📬 Step 4 · Fetch availability responses")
    

    st.markdown("### 📅 Automated Interview Scheduling (Google Workspace)")
    st.info(
    "📌 Use Google Forms → Sheets → Calendar + Gmail to collect availability, plan slots, "
    "and send confirmations without leaving this page."
    )

    pipeline_ready = pipeline is not None
    fetch_disabled = not pipeline_ready
    if st.button(
    "🔄 Fetch Google Form responses",
    width="stretch",
    type="primary",
    disabled=fetch_disabled,
    ) and pipeline:
        handle_google_form_fetch(pipeline)

    responses_df = st.session_state.get("interview_responses_df")
    if isinstance(responses_df, pd.DataFrame) and not responses_df.empty:
        new_count = st.session_state.get("new_responses_count", 0)
        st.success(f"✅ Loaded {new_count} response(s) from Google Sheets")

        with st.expander("📋 View responses", expanded=False):
            st.dataframe(responses_df, use_container_width=True, height=320)

    st.divider()
    st.markdown("### �️ Generate schedule proposals")
    if st.button(
        "�️ Plan interview schedule",
        width="stretch",
        disabled=fetch_disabled,
    ) and pipeline:
        handle_google_schedule_plan(pipeline)

    proposals_df = st.session_state.get("schedule_proposals_df")
    if isinstance(proposals_df, pd.DataFrame) and not proposals_df.empty:
        st.markdown("### ✏️ Review & edit proposals")
        editor_df = st.data_editor(
            proposals_df,
            key="proposal_editor",
            use_container_width=True,
            height=360,
            num_rows="dynamic",
            column_config={
                "Suggested Start": st.column_config.TextColumn(
                    "Suggested Start",
                    help="Format: DD-Mon-YYYY HH:MM AM/PM (e.g., 24-Nov-2025 10:00 AM)",
                    width="medium",
                ),
                "Suggested End": st.column_config.TextColumn(
                    "Suggested End",
                    help="Format: DD-Mon-YYYY HH:MM AM/PM (e.g., 24-Nov-2025 10:30 AM)",
                    width="medium",
                ),
                "Preferred Slot": st.column_config.TextColumn(
                    "Preferred Slot",
                    help="The time slot requested by the candidate",
                    width="medium",
                ),
                "Status": st.column_config.SelectboxColumn(
                    "Status",
                    options=["Ready", "Waiting", "Error", "Scheduled"],
                ),
            },
        )
        st.session_state["schedule_proposals_df"] = editor_df

        ready_series = editor_df.get("Status")
        ready_count = (
            int(ready_series.astype(str).str.lower().eq("ready").sum())
            if isinstance(ready_series, pd.Series)
            else 0
        )
        st.session_state["ready_count"] = ready_count

        col_a, col_b = st.columns(2)
        col_a.metric("Ready proposals", ready_count)
        col_b.metric("Total proposals", len(editor_df))

        if st.button(
            "✅ Finalize & send confirmations",
            width="stretch",
            type="primary",
            disabled=(ready_count == 0) or fetch_disabled,
        ) and pipeline:
            handle_google_schedule_finalize(pipeline, config)

    confirmed_df = st.session_state.get("confirmed_schedule_df")
    if isinstance(confirmed_df, pd.DataFrame) and not confirmed_df.empty:
        st.divider()
        st.markdown("### 📬 Confirmed interviews")
        st.dataframe(confirmed_df, use_container_width=True, height=320)
    elif pipeline_ready:
        st.info("👆 Click 'Fetch Google Form responses' to load the latest submissions.")
    else:
        st.error("Configure GOOGLE_* settings to enable the automated Google workflow.")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_internal_talent_page() -> None:
    """Compact internal talent search experience."""
    st.markdown("## 🏢 Fill Role Internally")
    st.caption("Search your private FAISS talent pool for internal movers in seconds.")

    store_dir = Path("internal_talent_store")
    index_path = store_dir / "faiss_index.bin"
    metadata_path = store_dir / "metadata.pkl"

    if not index_path.exists() or not metadata_path.exists():
        st.warning("⚠️ Internal talent pool not found.")
        st.info(
            "Add resumes to `data/`, then run `python build_faiss_from_cache.py` to generate the index."
        )
        return

    if "faiss_index" not in st.session_state or "faiss_metadata" not in st.session_state:
        try:
            import faiss
            import pickle

            st.session_state["faiss_index"] = faiss.read_index(str(index_path))
            with open(metadata_path, "rb") as f:
                st.session_state["faiss_metadata"] = pickle.load(f)
        except Exception as exc:  # pragma: no cover - IO heavy
            st.error(f"❌ Failed to load FAISS store: {exc}")
            logger.exception("Internal talent store load error")
            return

    index = st.session_state["faiss_index"]
    metadata = st.session_state["faiss_metadata"]
    total_indexed = index.ntotal

    library = get_jd_library()
    if not library:
        st.warning("⚠️ No job descriptions available. Please create one first.")
        return

    jd_options = {jd["id"]: jd for jd in library}
    col_select, col_meta = st.columns([1.4, 0.6])
    with col_select:
        selected_jd_id = st.selectbox(
            "Job description",
            options=list(jd_options.keys()),
            format_func=lambda x: jd_options[x]["title"],
            key="internal_jd_selector",
        )
        selected_jd = jd_options[selected_jd_id]
        st.text_area(
            "Internal JD",
            selected_jd["text"],
            height=150,
            disabled=True,
            label_visibility="collapsed",
        )

    with col_meta:
        st.metric("Indexed resumes", total_indexed)
        st.metric("Similarity threshold", "28%", help="Scores ≥28% behave as strong internal matches")
        search_clicked = st.button("🔍 Find internal matches", type="primary", use_container_width=True)

    threshold = 0.28
    if search_clicked:
        with st.spinner("Scanning internal profiles..."):
            try:
                matcher = load_resume_matcher()
                client = matcher._ensure_client()
                jd_embedding = matcher._embed_text(client, selected_jd["text"])
                if jd_embedding is None or len(jd_embedding) == 0:
                    st.error("❌ Failed to generate JD embedding")
                    return
                jd_embedding = jd_embedding.astype(np.float32)
                norm = np.linalg.norm(jd_embedding)
                if norm > 0:
                    jd_embedding = jd_embedding / norm
                similarities, indices = index.search(jd_embedding.reshape(1, -1), min(40, total_indexed))

                matches = []
                for similarity, idx in zip(similarities[0], indices[0]):
                    if idx < 0 or idx >= len(metadata):
                        continue
                    if similarity >= threshold:
                        candidate = metadata[idx]
                        matches.append(
                            {
                                "Name": candidate.get("name", "N/A"),
                                "Email": candidate.get("email", "N/A"),
                                "Phone": candidate.get("phone", "N/A"),
                                "Skills": candidate.get("skills", []),
                                "Education": candidate.get("education", []),
                                "Experience": candidate.get("experience", []),
                                "Summary": candidate.get("summary", ""),
                                "Match Score": float(similarity),
                                "File Path": candidate.get("file_path", ""),
                                "File Name": candidate.get("file_name", "internal_resume.pdf"),
                            }
                        )

                st.session_state["internal_matches"] = pd.DataFrame(matches)
                st.session_state["internal_search_performed"] = True
            except Exception as exc:  # pragma: no cover - FAISS heavy
                st.error(f"❌ Search failed: {exc}")
                logger.exception("Internal talent search error")

    if st.session_state.get("internal_search_performed"):
        matches_df = st.session_state.get("internal_matches")
        st.markdown("### 🎯 Best internal matches")

        if matches_df is None or matches_df.empty:
            st.warning("No strong matches found for this JD.")
            st.caption("Try lowering the bar or expanding your talent pool.")
            return

        st.success(f"Found {len(matches_df)} relevant internal candidates.")
        table_df = matches_df.copy()
        table_df["Match Score (%)"] = (table_df["Match Score"] * 100).round(1)
        st.dataframe(
            table_df[["Name", "Match Score (%)", "Email", "Phone"]].head(12),
            use_container_width=True,
            height=240,
        )

        explore_df = matches_df.head(8).reset_index(drop=True)
        option_labels = [
            f"{row['Name'] or 'Candidate'} · {row['Match Score'] * 100:.1f}%"
            for _, row in explore_df.iterrows()
        ]
        selected_idx = st.selectbox(
            "Inspect employee profile",
            list(range(len(option_labels))),
            format_func=lambda i: option_labels[i],
            key="internal_match_selector",
        )
        profile = explore_df.loc[selected_idx]

        left, right = st.columns([1.1, 0.9])
        with left:
            st.markdown(f"**Email:** {profile.get('Email', 'N/A')}")
            st.markdown(f"**Phone:** {profile.get('Phone', 'N/A')}")
            education = profile.get("Education", [])
            if isinstance(education, list) and education:
                st.markdown(f"**Education:** {education[0]}")
            skills = profile.get("Skills", [])
            if isinstance(skills, list) and skills:
                st.markdown(
                    "**Key skills:** " + ", ".join(skills[:12])
                )
            summary = profile.get("Summary")
            if summary:
                st.write(summary)

        with right:
            experience = profile.get("Experience", [])
            if isinstance(experience, list) and experience:
                st.markdown("**Experience highlights**")
                for bullet in experience[:4]:
                    st.markdown(f"- {bullet}")
            file_path = profile.get("File Path")
            file_name = profile.get("File Name", "resume.pdf")
            if file_path and Path(file_path).exists():
                with open(file_path, "rb") as f:
                    st.download_button(
                        label=f"📄 Download {file_name}",
                        data=f.read(),
                        file_name=file_name,
                        mime="application/pdf",
                        use_container_width=True,
                    )

        csv = matches_df.to_csv(index=False)
        st.download_button(
            label="📥 Download matches CSV",
            data=csv,
            file_name=f"internal_matches_{selected_jd['title'].replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def main() -> None:
    ensure_session_state()
    configure_page()
    inject_styles()
    render_header()
    config = render_config_panel()

    nav_tabs = st.tabs(["Job Descriptions", "Resume Ranking", "Scheduling", "Fill Role Internally"])
    with nav_tabs[0]:
        render_jd_page()
    with nav_tabs[1]:
        render_ranking_page()
    with nav_tabs[2]:
        render_scheduling_page(config)
    with nav_tabs[3]:
        render_internal_talent_page()


if __name__ == "__main__":
    main()
