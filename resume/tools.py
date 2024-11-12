from typing import Optional, Literal
from langchain_core.tools import tool

from modules.job import Job
from modules.resume import Resume

def process_job() -> Job:
    """Process job data."""
    job = Job.mock()
    return job

def process_resume() -> Resume:
    """Process resume data."""
    resume = Resume.mock()
    return resume

@tool
def get_job(field: Optional[Literal['title', 'company', 'location', 'salary', 'description', 'responsibilities', 'benefits', 'employment_type', 'posted_date']] = None) -> str:
    """Get job data."""
    job = process_job()
    print("job :", job)
    if field:
        return getattr(job, field)
    return job.dict()


@tool
def get_resume(field: Optional[Literal['name', 'professional_summary', 'work_experience', 'education', 'skills']] = None) -> str:
    """Get resume data."""
    resume = process_resume()
    print("resume :", resume)
    if field:
        return getattr(resume, field)
    return resume.dict()