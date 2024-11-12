from typing import List, Optional
from pydantic import BaseModel, Field, validator

class WorkExperience(BaseModel):
    job_title: str = Field(description="Job title or position.")
    company: str = Field(description="The company name.")
    experience: int = Field(description="Years of experience in the job.")
    responsibilities: List[str] = Field(description="List of responsibilities in the job.")

class Education(BaseModel):
    degree: str = Field(description="Degree obtained.")
    school: str = Field(description="The university name.")
    major: str = Field(description="Major subject.")
    year: Optional[int] = Field(description="Year of graduation.")

    @validator('year', pre=True, always=True)
    def set_year(cls, v):
        if v is None:
            return 0
        return v

class Resume(BaseModel):
    """Structured resume data."""

    name: str = Field(description="Name of the person")
    professional_summary: str = Field(description="Professional summary of the person.")
    work_experience: List[WorkExperience] = Field(description="List of work experiences held by the person.")
    education: List[Education] = Field(description="List of educational qualifications of the person.")
    skills: List[str] = Field(description="List of skills relevant to the jobs.")

    @classmethod
    def mock(cls):
        return cls(
            name='Jeff',
            professional_summary='Innovative software engineer with 8+ years of experience in the tech industry. Senior Developer at Company X, Freelance Software Architect, and Junior Developer at Company Y. Proficient in developing scalable applications, optimizing system performance, and leading cross-functional teams. Fluent in English and Spanish.',
            work_experience=[
                WorkExperience(
                    job_title='Senior Developer',
                    company='Company X',
                    experience=5,
                    responsibilities=[
                        'Led the development of scalable web applications',
                        'Optimized system performance and reduced server costs',
                        'Mentored junior developers and conducted code reviews',
                        'Collaborated with product managers to define project requirements',
                        'Implemented CI/CD pipelines to streamline deployments',
                        'Developed RESTful APIs for mobile and web applications',
                        'Ensured application security and compliance with industry standards'
                    ]
                ),
                 WorkExperience(
                    job_title='Freelance Software Architect',
                    company='Independent Consultant',
                    experience=2,
                    responsibilities=[
                        'Designed software architecture for various clients',
                        'Provided technical consultancy and project management',
                        'Developed custom software solutions to meet client needs',
                        'Conducted system analysis and performance tuning',
                        'Integrated third-party services and APIs',
                        'Created technical documentation and user manuals'
                    ]
                ),
                WorkExperience(
                    job_title='Junior Developer',
                    company='Company Y',
                    experience=1,
                    responsibilities=[
                        'Assisted in the development of web applications',
                        'Performed bug fixes and code maintenance',
                        'Collaborated with senior developers on project tasks',
                        'Participated in daily stand-ups and sprint planning',
                        'Wrote unit tests to ensure code quality',
                        'Contributed to open-source projects'
                    ]
                )
            ],
             education=[
                Education(
                    degree='B.Sc. Computer Science',
                    school='X University',
                    major='Computer Science',
                    year=1999
                )
            ],
            skills=[
                'Software Architecture',
                'System Optimization',
                'Team Mentorship',
                'Project Management',
                'API Development',
                'Continuous Integration/Continuous Deployment',
                'Bilingual'
            ]
        )