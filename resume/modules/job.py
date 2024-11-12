from typing import List, Optional
from pydantic import BaseModel, Field

class Job(BaseModel):
    title: str = Field(description="Job title or position.")
    company: str = Field(description="The company name.")
    location: Optional[str] = Field(description="Location of the job.")
    salary: Optional[str] = Field(description="Salary range for the job.")
    description: str = Field(description="Detailed job description.")
    responsibilities: List[str] = Field(description="List of job responsibilities.")
    benefits: Optional[List[str]] = Field(description="List of job benefits.")
    employment_type: Optional[str] = Field(description="Type of employment (e.g., full-time, part-time).")
    posted_date: Optional[str] = Field(description="Date when the job was posted.")

    @classmethod
    def mock(cls):
        return cls(
            title='Software Engineer',
            company='Tech Corp',
            location='San Francisco, CA',
            salary='$100,000 - $120,000',
            description='We are looking for a skilled Software Engineer to join our team.',
            requirements=[
                'Bachelor\'s degree in Computer Science or related field',
                '3+ years of experience in software development',
                'Proficiency in Python and JavaScript',
                'Experience with Django and React',
                'Strong problem-solving skills'
            ],
             responsibilities=[
                'Develop and maintain web applications',
                'Collaborate with cross-functional teams',
                'Write clean, scalable, and efficient code',
                'Participate in code reviews',
                'Troubleshoot and debug applications'
            ],
            benefits=[
                'Health insurance',
                '401(k) matching',
                'Paid time off',
                'Flexible working hours'
            ],
            employment_type='Full-time',
            posted_date='2024-10-01'
        )