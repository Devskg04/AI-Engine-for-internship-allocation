import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
from typing import Dict, List, Tuple, Optional
import logging
import time
import random
import re
import PyPDF2
import pdfplumber
from pathlib import Path
import hashlib
import shutil
import warnings
import traceback

# ML/AI Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import sqlite3

warnings.filterwarnings('ignore')

# filepath: d:/SIH/admin.py
# Add this function to update your schema

def add_source_column_to_allocations():
    conn = sqlite3.connect('internship_allocation.db')  # Update with your actual DB path
    cursor = conn.cursor()
    # Check if 'source' column exists
    cursor.execute("PRAGMA table_info(allocations)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'source' not in columns:
        cursor.execute("ALTER TABLE allocations ADD COLUMN source TEXT")
        print("Added 'source' column to allocations table.")
    else:
        print("'source' column already exists.")
    conn.commit()
    conn.close()

# Call this function before any allocation logic
add_source_column_to_allocations()
def convert_numpy_types(obj):
    """Convert numpy types to regular Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

# Field categories for organizing resumes
FIELD_CATEGORIES = {
    "ACCOUNTANT": ["Accounting", "Finance", "Commerce", "Business"],
    "ADVOCATE": ["Law", "Legal", "LLB", "Corporate Law"],
    "AGRICULTURE": ["Agriculture", "Farming", "Agronomy", "Horticulture"],
    "APPAREL": ["Fashion", "Textile", "Design", "Merchandising"],
    "Arts": ["Literature", "History", "Psychology", "Sociology"],
    "AUTOMOBILE": ["Automotive", "Mechanical", "Vehicle Engineering"],
    "AVIATION": ["Aeronautics", "Aviation", "Aerospace", "Pilot"],
    "BANKING": ["Banking", "Finance", "Economics", "Commerce"],
    "BPO": ["Customer Service", "Operations", "Process Management"],
    "BUSINESS-DEVELOPMENT": ["Business", "Sales", "Marketing", "Strategy"],
    "CHEF": ["Culinary", "Hospitality", "Food Science"],
    "Commerce": ["Commerce", "Business", "Accounting", "Finance"],
    "CONSTRUCTION": ["Civil Engineering", "Architecture", "Construction"],
    "CONSULTANT": ["Consulting", "Advisory", "Strategy", "Management"],
    "DESIGNER": ["Design", "Creative", "Graphics", "UI/UX"],
    "DIGITAL-MEDIA": ["Digital Marketing", "Social Media", "Content"],
    "Engineering": ["Engineering", "Technology", "Computer Science"],
    "FINANCE": ["Finance", "Investment", "Banking", "Accounting"],
    "FITNESS": ["Sports Science", "Physical Education", "Health"],
    "HEALTHCARE": ["Medicine", "Nursing", "Healthcare", "Pharmacy"],
    "HR": ["Human Resources", "Psychology", "Management"],
    "INFORMATION-TECHNOLOGY": ["IT", "Computer Science", "Software"],
    "Law": ["Law", "Legal Studies", "LLB", "Judiciary"],
    "Management": ["Management", "MBA", "Business Administration"],
    "Medical": ["Medicine", "MBBS", "Nursing", "Healthcare"],
    "PUBLIC-RELATIONS": ["PR", "Communications", "Media", "Journalism"],
    "SALES": ["Sales", "Marketing", "Business Development"],
    "Science": ["Science", "Research", "Physics", "Chemistry", "Biology"]
}

INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", 
    "Ahmedabad", "Jaipur", "Surat", "Lucknow", "Kanpur", "Nagpur", "Indore", 
    "Thane", "Bhopal", "Visakhapatnam", "Patna", "Vadodara", "Ghaziabad", 
    "Ludhiana", "Coimbatore", "Agra", "Madurai", "Nashik", "Faridabad",
    "Meerut", "Rajkot", "Varanasi", "Aurangabad", "Amritsar", "Allahabad", 
    "Ranchi", "Gwalior", "Jabalpur", "Vijayawada", "Jodhpur", "Raipur",
    "Gurugram", "Gurgaon", "Noida"
]

def normalize_city_name(city: str) -> str:
    """Normalize city names for consistent matching"""
    if not city or pd.isna(city):
        return ""
    
    city_aliases = {
        'bombay': 'Mumbai', 'madras': 'Chennai', 'calcutta': 'Kolkata',
        'bengaluru': 'Bangalore', 'gurgaon': 'Gurugram'
    }
    
    city = str(city).strip().title()
    city_lower = city.lower()
    
    if city_lower in city_aliases:
        return city_aliases[city_lower]
    
    for indian_city in INDIAN_CITIES:
        if city_lower == indian_city.lower():
            return indian_city
    
    return city

class MLResumeParser:
    """AI/ML Enhanced resume parser with training capabilities"""
    
    def __init__(self):
        print("Initializing AI/ML resume parser with training capabilities...")
        self.skill_vocabulary = [
            "Python", "Java", "JavaScript", "C++", "C#", "PHP", "Ruby", "Go", "HTML", "CSS",
            "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Spring",
            "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Jenkins", "Git", "Linux",
            "Machine Learning", "Deep Learning", "Data Science", "AI", "NLP", "Computer Vision",
            "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy", "Matplotlib",
            "Project Management", "Agile", "Scrum", "Leadership", "Strategy", "Marketing",
            "Sales", "Business Development", "Finance", "Accounting", "Operations",
            "Consulting", "Analytics", "Research", "Communication", "Teamwork",
            "UI/UX", "Photoshop", "Illustrator", "Figma", "Sketch", "InDesign",
            "DevOps", "CI/CD", "Microservices", "API", "REST", "GraphQL", "Blockchain",
            "Cybersecurity", "Networking", "Cloud Computing", "Big Data", "IoT",
            "Content Writing", "SEO", "Digital Marketing", "Social Media", "Excel",
            "PowerBI", "Tableau", "R", "MATLAB", "Statistics", "Mathematics"
        ]
        
        # Initialize ML models
        self.tfidf_vectorizer = None
        self.skill_model = None
        self.experience_model = None
        self.cgpa_model = None
        
        # Training data storage
        self.training_features = []
        self.training_labels = {}
        
        # Load or train models
        self.load_or_train_models()
    
    def load_or_train_models(self):
        """Load existing models or train new ones from data folder"""
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            # Try to load existing models
            self.tfidf_vectorizer = joblib.load(f"{model_dir}/tfidf_vectorizer.pkl")
            self.skill_model = joblib.load(f"{model_dir}/skill_model.pkl")
            self.experience_model = joblib.load(f"{model_dir}/experience_model.pkl")
            self.cgpa_model = joblib.load(f"{model_dir}/cgpa_model.pkl")
            print("Loaded existing ML models successfully!")
        except:
            print("Training new ML models from data folder...")
            self.train_models_from_data()
    
    def train_models_from_data(self):
        """Train ML models using PDFs from data/{field} folders"""
        training_data = []
        
        # Collect training data from data folder
        data_folder = "data"
        if os.path.exists(data_folder):
            for field in FIELD_CATEGORIES.keys():
                field_folder = os.path.join(data_folder, field)
                if os.path.exists(field_folder):
                    print(f"Processing training data for {field}...")
                    for pdf_file in os.listdir(field_folder):
                        if pdf_file.lower().endswith('.pdf'):
                            pdf_path = os.path.join(field_folder, pdf_file)
                            try:
                                text = self.extract_text_from_pdf(pdf_path)
                                if text:
                                    # Generate training labels based on field
                                    training_sample = {
                                        'text': text,
                                        'field': field,
                                        'skills': self.extract_skills_basic(text),
                                        'experience': self.extract_experience_basic(text),
                                        'cgpa': self.generate_training_cgpa(field),
                                        'education_level': self.extract_education_level_basic(text)
                                    }
                                    training_data.append(training_sample)
                            except Exception as e:
                                print(f"Error processing {pdf_file}: {e}")
        
        # If no training data found, use default models
        if not training_data:
            print("No training data found in data folder. Using default models...")
            self.create_default_models()
            return
        
        print(f"Training models on {len(training_data)} samples...")
        
        # Prepare training features
        texts = [sample['text'] for sample in training_data]
        
        # Train TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        text_features = self.tfidf_vectorizer.fit_transform(texts)
        
        # Prepare labels for different models
        skill_labels = [len(sample['skills']) for sample in training_data]
        experience_labels = [sample['experience'] for sample in training_data]
        cgpa_labels = [sample['cgpa'] for sample in training_data]
        
        # Train models
        self.skill_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.skill_model.fit(text_features.toarray(), skill_labels)
        
        self.experience_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.experience_model.fit(text_features.toarray(), experience_labels)
        
        self.cgpa_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.cgpa_model.fit(text_features.toarray(), cgpa_labels)
        
        # Save models
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.tfidf_vectorizer, f"{model_dir}/tfidf_vectorizer.pkl")
        joblib.dump(self.skill_model, f"{model_dir}/skill_model.pkl")
        joblib.dump(self.experience_model, f"{model_dir}/experience_model.pkl")
        joblib.dump(self.cgpa_model, f"{model_dir}/cgpa_model.pkl")
        
        print("ML models trained and saved successfully!")
    
    def create_default_models(self):
        """Create default models when no training data available"""
        # Create simple default models
        sample_texts = ["sample resume text", "another sample"]
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100)
        self.tfidf_vectorizer.fit(sample_texts)
        
        # Simple default models
        from sklearn.dummy import DummyRegressor
        self.skill_model = DummyRegressor(strategy='constant', constant=5)
        self.experience_model = DummyRegressor(strategy='constant', constant=6)
        self.cgpa_model = DummyRegressor(strategy='constant', constant=7.0)
        
        # Fit with dummy data
        dummy_features = self.tfidf_vectorizer.transform(sample_texts)
        self.skill_model.fit(dummy_features, [5, 5])
        self.experience_model.fit(dummy_features, [6, 6])
        self.cgpa_model.fit(dummy_features, [7.0, 7.0])
    
    def extract_skills_basic(self, text: str) -> List[str]:
        """Enhanced skill extraction with comprehensive pattern matching"""
        if not text:
            return ["Communication", "Teamwork"]
        
        text_lower = text.lower()
        found_skills = set()
        
        # Enhanced skill vocabulary with variations
        enhanced_skill_vocabulary = {
            # Only Programming Languages
            'Python': ['python', 'python3', 'py'],
            'Java': ['java', 'core java'],
            'JavaScript': ['javascript', 'js'],
            'C++': ['c++', 'cpp'],
            'C#': ['c#', 'csharp'],
            'PHP': ['php'],
            'Ruby': ['ruby'],
            'Go': ['golang', 'go'],
            'Swift': ['swift'],
            'Kotlin': ['kotlin'],
            'TypeScript': ['typescript'],
            'Scala': ['scala'],
            'Rust': ['rust'],
            'Perl': ['perl'],
            'C': [' c ', 'c programming'],
            'R': [' r ', 'r programming'],
            'MATLAB': ['matlab'],
            'SQL': ['sql'],
            'HTML': ['html'],
            'CSS': ['css'],
        }
        
        # Search for programming languages only
        for language, variations in enhanced_skill_vocabulary.items():
            for variation in variations:
                if variation in text_lower:
                    found_skills.add(language)
                    break
        
        # Convert to list
        skills_list = list(found_skills)
        
        # Add basic soft skills if no coding languages found
        if not skills_list:
            skills_list = ["Communication", "Teamwork"]
        
        return skills_list
    
    def extract_experience_basic(self, text: str) -> int:
        """Enhanced experience extraction for training"""
        if not text:
            return 0
        
        text_lower = text.lower()
        experience_months = 0
        
        # Look for explicit year/month patterns
        year_patterns = [
            r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp|work)',
            r'(?:experience|exp|work)\s*(?:of\s*)?\s*(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\s*(?:years?|yrs?)\s*(?:in|at|with)',
            r'(?:total|overall)\s*(?:experience|exp)\s*[:-]?\s*(\d+)\s*(?:years?|yrs?)'
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    years = int(matches[0])
                    experience_months = max(experience_months, years * 12)
                except:
                    continue
        
        # Look for month patterns
        month_patterns = [
            r'(\d+)\s*months?\s*(?:of\s*)?(?:experience|exp|work)',
            r'(?:experience|exp|work)\s*(?:of\s*)?\s*(\d+)\s*months?',
            r'(\d+)\s*months?\s*(?:in|at|with)'
        ]
        
        for pattern in month_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    months = int(matches[0])
                    experience_months = max(experience_months, months)
                except:
                    continue
        
        # Look for year ranges (e.g., "2020-2023")
        date_range_patterns = [
            r'(20\d{2})\s*[-–—]\s*(20\d{2})',
            r'(\d{4})\s*[-–—]\s*(\d{4})',
            r'(\d{1,2})[/\\](\d{4})\s*[-–—]\s*(\d{1,2})[/\\](\d{4})'
        ]
        
        for pattern in date_range_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    if len(match) == 2:  # Year range
                        start_year, end_year = int(match[0]), int(match[1])
                        if 2000 <= start_year <= 2024 and 2000 <= end_year <= 2024:
                            duration_months = (end_year - start_year) * 12
                            experience_months = max(experience_months, duration_months)
                except:
                    continue
        
        # Look for work-related keywords and estimate
        work_keywords = ['internship', 'job', 'position', 'role', 'work', 'employed', 'freelance', 'consultant']
        work_mentions = sum(1 for keyword in work_keywords if keyword in text_lower)
        
        if work_mentions >= 3 and experience_months == 0:
            experience_months = 6  # Default estimate for multiple work mentions
        
        # Look for specific duration formats (e.g., "6 months", "1.5 years")
        duration_patterns = [
            r'(\d+\.\d+)\s*(?:years?|yrs?)',
            r'(\d+)\s*to\s*(\d+)\s*(?:months?|years?)',
            r'(?:duration|period)\s*[:-]?\s*(\d+)\s*(?:months?|years?)'
        ]
        
        for pattern in duration_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    if isinstance(matches[0], tuple):
                        # Range pattern
                        start, end = matches[0]
                        avg_duration = (int(start) + int(end)) / 2
                        if 'year' in pattern:
                            experience_months = max(experience_months, int(avg_duration * 12))
                        else:
                            experience_months = max(experience_months, int(avg_duration))
                    else:
                        # Single value
                        value = float(matches[0])
                        if 'year' in pattern:
                            experience_months = max(experience_months, int(value * 12))
                        else:
                            experience_months = max(experience_months, int(value))
                except:
                    continue
        
        return min(experience_months, 120)  # Cap at 10 years
    
    def extract_education_level_basic(self, text: str) -> str:
        """Basic education level extraction"""
        if not text:
            return "Bachelor's"
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['phd', 'doctorate']):
            return "PhD"
        elif any(word in text_lower for word in ['master', 'mtech', 'mba']):
            return "Master's"
        else:
            return "Bachelor's"
    
    def generate_training_cgpa(self, field: str) -> float:
        """Generate realistic CGPA for training based on field"""
        field_cgpa_ranges = {
            "Engineering": (7.0, 9.0),
            "Medical": (7.5, 9.5),
            "Law": (6.5, 8.5),
            "Management": (7.0, 8.5),
            "Arts": (6.0, 8.0),
            "Commerce": (6.5, 8.0)
        }
        
        range_min, range_max = field_cgpa_ranges.get(field, (6.5, 8.0))
        return round(random.uniform(range_min, range_max), 1)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e2:
                return ""
        
        return text.strip()
    
    def extract_skills(self, text: str) -> List[str]:
        """Enhanced skill extraction with comprehensive pattern matching"""
        if not text:
            return ["Communication", "Teamwork", "Problem Solving"]
        
        # Use the enhanced basic extraction method
        return self.extract_skills_basic(text)
    
    def extract_experience(self, text: str) -> int:
        """Enhanced experience extraction with comprehensive pattern matching"""
        if not text:
            return 0
        
        # Use the enhanced basic extraction method
        return self.extract_experience_basic(text)
    
    def extract_cgpa(self, text: str, education_level: str) -> float:
        """Simple and accurate CGPA extraction - no rounding, exact values"""
        if not text:
            return 7.0
        
        # Direct CGPA patterns - extract exact values
        cgpa_patterns = [
            r'(?:cgpa|gpa)\s*[:-]?\s*([0-9]+\.?[0-9]*)',
            r'([0-9]+\.?[0-9]*)\s*/\s*10',
            r'([0-9]+\.?[0-9]*)\s*/\s*4',
            r'(?:percentage|%)\s*[:-]?\s*([0-9]+\.?[0-9]*)',
            r'([0-9]+\.?[0-9]*)\s*%',
        ]
        
        text_lower = text.lower()
        
        for pattern in cgpa_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    value = float(matches[0])
                    
                    # Convert based on scale - NO ROUNDING
                    if value <= 4.0:  # 4.0 scale
                        return (value / 4.0) * 10
                    elif value <= 10.0:  # 10.0 scale - return as is
                        return value
                    elif value <= 100.0:  # Percentage - convert to 10 scale
                        return value / 10.0
                        
                except (ValueError, TypeError):
                    continue
        
        # Simple fallback based on education level
        defaults = {"PhD": 8.2, "Master's": 7.8, "Bachelor's": 7.0}
        return defaults.get(education_level, 7.0)
    
    def extract_education_level(self, text: str) -> str:
        """Extract education level"""
        if not text:
            return "Bachelor's"
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['phd', 'doctorate', 'doctoral']):
            return "PhD"
        elif any(word in text_lower for word in ['master', 'mtech', 'mba', 'msc', 'mcom', 'pg']):
            return "Master's"
        else:
            return "Bachelor's"
    
    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information"""
        email = None
        phone = None
        name = None
        
        if text:
            # Extract email
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, text)
            if emails:
                email = emails[0]
            
            # Extract phone
            phone_pattern = r'[\+]?[1-9][0-9 .\-\(\)]{8,}[0-9]'
            phones = re.findall(phone_pattern, text)
            if phones:
                phone = phones[0]
            
            # Extract name (first non-empty line that looks like a name)
            lines = text.split('\n')[:5]
            for line in lines:
                line = line.strip()
                if (3 < len(line) < 50 and 
                    re.match(r'^[A-Za-z\s.]+$', line) and
                    (not email or email not in line)):
                    name = line
                    break
        
        return {'email': email, 'phone': phone, 'name': name}
    
    def parse_resume(self, pdf_path: str, field: str, username: str) -> Dict:
        """Parse resume and extract information using ML"""
        try:
            print(f"ML-parsing resume: {pdf_path}")
            text = self.extract_text_from_pdf(pdf_path)
            
            if not text:
                print("No text extracted from PDF")
                return self.get_default_resume_data(username, field)
            
            skills = self.extract_skills(text)
            contact_info = self.extract_contact_info(text)
            education_level = self.extract_education_level(text)
            experience = self.extract_experience(text)
            cgpa = self.extract_cgpa(text, education_level)
            
            print(f"ML extracted: {len(skills)} skills, {experience} months exp, CGPA {cgpa}")
            
            return {
                'id': f"CAND_{hashlib.md5(username.encode()).hexdigest()[:8].upper()}",
                'username': username,
                'field': field,
                'name': contact_info['name'] or username,
                'email': contact_info['email'] or f"{username}@email.com",
                'phone': contact_info['phone'] or '',
                'skills': skills,
                'education_level': education_level,
                'cgpa': cgpa,
                'experience_months': experience,
                'resume_text': text[:1000],
                'pdf_path': pdf_path
            }
        except Exception as e:
            print(f"Error in ML parsing: {e}")
            return self.get_default_resume_data(username, field)
    
    def get_default_resume_data(self, username: str, field: str) -> Dict:
        """Generate default resume data"""
        return {
            'id': f"CAND_{hashlib.md5(username.encode()).hexdigest()[:8].upper()}",
            'username': username,
            'field': field,
            'name': username,
            'email': f"{username}@email.com",
            'phone': '',
            'skills': ["Communication", "Teamwork", "Problem Solving"],
            'education_level': "Bachelor's",
            'cgpa': 7.0,
            'experience_months': 0,
            'resume_text': "",
            'pdf_path': None
        }

class InternshipRecommendationEngine:
    """Advanced recommendation engine for internships with ML integration"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.ml_allocation_engine = None  # Will be set by main system
        
        # Skill similarity mapping for enhanced recommendations
        self.skill_groups = {
            'Programming': ['Python', 'Java', 'JavaScript', 'C++', 'C#', 'PHP', 'Ruby', 'Go'],
            'Web Development': ['HTML', 'CSS', 'React', 'Angular', 'Vue.js', 'Node.js'],
            'Data Science': ['Machine Learning', 'Data Science', 'Python', 'R', 'Statistics', 'Pandas', 'NumPy'],
            'Cloud & DevOps': ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Jenkins', 'DevOps'],
            'Database': ['SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Redis'],
            'Business': ['Project Management', 'Leadership', 'Strategy', 'Marketing', 'Sales'],
            'Design': ['UI/UX', 'Photoshop', 'Illustrator', 'Figma', 'Sketch']
        }
        
        # Field to sector mapping
        self.field_sector_mapping = {
            'Engineering': ['Technology', 'Automobile', 'Construction'],
            'INFORMATION-TECHNOLOGY': ['Technology'],
            'FINANCE': ['Finance', 'Banking'],
            'BANKING': ['Finance', 'Banking'],
            'HEALTHCARE': ['Healthcare', 'Medical'],
            'Medical': ['Healthcare', 'Medical'],
            'Management': ['Consulting', 'Business'],
            'BUSINESS-DEVELOPMENT': ['Business', 'Consulting'],
            'MARKETING': ['Business', 'Technology'],
            'Law': ['Legal', 'Consulting'],
            'Commerce': ['Finance', 'Business']
        }
    
    def set_ml_engine(self, ml_engine):
        """Set the ML allocation engine for enhanced scoring"""
        self.ml_allocation_engine = ml_engine
    
    def get_skill_similarity_score(self, candidate_skills: List[str], required_skills: List[str]) -> float:
        """Calculate advanced skill similarity including skill groups"""
        if not candidate_skills or not required_skills:
            return 0.0
        
        candidate_skills_lower = [s.lower() for s in candidate_skills]
        required_skills_lower = [s.lower() for s in required_skills]
        
        # Direct skill matches
        direct_matches = len(set(candidate_skills_lower) & set(required_skills_lower))
        direct_score = (direct_matches / len(required_skills_lower)) * 100
        
        # Skill group matches
        group_score = 0
        for group_name, skills in self.skill_groups.items():
            candidate_group_skills = [s for s in candidate_skills if s in skills]
            required_group_skills = [s for s in required_skills if s in skills]
            
            if candidate_group_skills and required_group_skills:
                group_match_ratio = min(len(candidate_group_skills) / len(required_group_skills), 1.0)
                group_score += group_match_ratio * 20  # Max 20 points per group
        
        # Combine scores
        total_score = min(direct_score + (group_score * 0.5), 100)
        return total_score
    
    def get_recommendations(self, username: str, top_n: int = 5) -> List[Dict]:
        """Get personalized internship recommendations for a student"""
        try:
            # Get student data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            candidate_id = f"CAND_{hashlib.md5(username.encode()).hexdigest()[:8].upper()}"
            cursor.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,))
            candidate_row = cursor.fetchone()
            
            if not candidate_row:
                print(f"Student {username} not found")
                conn.close()
                return []
            
            candidate_data = json.loads(candidate_row[3])
            
            # Get all available internships (with capacity > 0)
            cursor.execute("SELECT * FROM internships WHERE current_capacity > 0")
            internship_rows = cursor.fetchall()
            conn.close()
            
            if not internship_rows:
                print("No internships available")
                return []
            
            recommendations = []
            
            for row in internship_rows:
                try:
                    internship = json.loads(row[2])
                    
                    # Calculate comprehensive recommendation score
                    rec_score = self.calculate_recommendation_score(candidate_data, internship)
                    
                    if rec_score['total_score'] > 0:
                        recommendation = {
                            'internship': internship,
                            'scores': rec_score,
                            'recommendation_reasons': self.generate_recommendation_reasons(candidate_data, internship, rec_score)
                        }
                        recommendations.append(recommendation)
                        
                except Exception as e:
                    print(f"Error processing internship: {e}")
                    continue
            
            # Sort by total score and return top N
            recommendations.sort(key=lambda x: x['scores']['total_score'], reverse=True)
            return recommendations[:top_n]
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return []
    
    def calculate_recommendation_score(self, candidate: Dict, internship: Dict) -> Dict:
        """Calculate comprehensive recommendation score with ML enhancement"""
        scores = {
            'field_match': 0,
            'skill_match': 0,
            'cgpa_match': 0,
            'location_match': 0,
            'experience_match': 0,
            'growth_potential': 0,
            'ml_enhancement': 0,
            'total_score': 0
        }
        
        # Traditional scoring components
        # Field/Sector matching (25 points)
        candidate_field = candidate.get('field', '').upper()
        internship_sector = internship.get('sector', '').upper()
        
        mapped_sectors = self.field_sector_mapping.get(candidate_field, [])
        if any(sector.upper() in internship_sector for sector in mapped_sectors):
            scores['field_match'] = 25
        elif candidate_field in internship_sector or internship_sector in candidate_field:
            scores['field_match'] = 20
        else:
            scores['field_match'] = 5
        
        # Skill matching (25 points)
        candidate_skills = candidate.get('skills', [])
        required_skills = internship.get('required_skills', [])
        scores['skill_match'] = self.get_skill_similarity_score(candidate_skills, required_skills)
        scores['skill_match'] = min(scores['skill_match'], 25)
        
        # CGPA matching (20 points)
        candidate_cgpa = float(candidate.get('cgpa', 7.0))
        min_cgpa = float(internship.get('min_cgpa', 6.0))
        if candidate_cgpa >= min_cgpa + 1.5:
            scores['cgpa_match'] = 20
        elif candidate_cgpa >= min_cgpa + 0.5:
            scores['cgpa_match'] = 15
        elif candidate_cgpa >= min_cgpa:
            scores['cgpa_match'] = 10
        else:
            scores['cgpa_match'] = 2
        
        # Location matching (15 points)
        preferred_locations = candidate.get('preferred_locations', [])
        internship_location = internship.get('location', '')
        if preferred_locations:
            for pref_loc in preferred_locations:
                if pref_loc.lower() in internship_location.lower():
                    scores['location_match'] = 15
                    break
            else:
                scores['location_match'] = 3
        else:
            scores['location_match'] = 8  # Neutral if no preference
        
        # Experience matching (10 points)
        candidate_experience = int(candidate.get('experience_months', 0))
        if candidate_experience >= 24:
            scores['experience_match'] = 10
        elif candidate_experience >= 12:
            scores['experience_match'] = 8
        elif candidate_experience >= 6:
            scores['experience_match'] = 6
        elif candidate_experience > 0:
            scores['experience_match'] = 4
        else:
            scores['experience_match'] = 2
        
        # Growth potential (5 points)
        education_level = candidate.get('education_level', "Bachelor's")
        if education_level == "PhD":
            scores['growth_potential'] = 5
        elif education_level == "Master's":
            scores['growth_potential'] = 4
        else:
            scores['growth_potential'] = 3
        
        # Calculate traditional total
        traditional_total = (
            scores['field_match'] + 
            scores['skill_match'] + 
            scores['cgpa_match'] + 
            scores['location_match'] + 
            scores['experience_match'] + 
            scores['growth_potential']
        )
        
        # ML Enhancement (up to 10 bonus points)
        if self.ml_allocation_engine and self.ml_allocation_engine.is_trained:
            try:
                ml_score = self.ml_allocation_engine.predict_compatibility(candidate, internship)
                if ml_score is not None:
                    # ML enhancement based on how much ML score exceeds traditional score
                    ml_boost = max(0, min(10, (ml_score - traditional_total) * 0.1))
                    scores['ml_enhancement'] = ml_boost
                    scores['total_score'] = traditional_total + ml_boost
                else:
                    scores['total_score'] = traditional_total
            except Exception as e:
                scores['total_score'] = traditional_total
        else:
            scores['total_score'] = traditional_total
        
        return scores
    
    def generate_recommendation_reasons(self, candidate: Dict, internship: Dict, scores: Dict) -> List[str]:
        """Generate human-readable reasons for the recommendation"""
        reasons = []
        
        # Field match reasons
        if scores['field_match'] >= 20:
            reasons.append(f"Perfect field alignment: Your {candidate.get('field', '')} background matches {internship.get('sector', '')} sector")
        elif scores['field_match'] >= 10:
            reasons.append(f"Good field compatibility with {internship.get('sector', '')} sector")
        
        # Skill match reasons
        candidate_skills = set(s.lower() for s in candidate.get('skills', []))
        required_skills = set(s.lower() for s in internship.get('required_skills', []))
        matching_skills = candidate_skills & required_skills
        
        if matching_skills:
            if len(matching_skills) >= 3:
                reasons.append(f"Strong skill match: You have {len(matching_skills)} relevant skills including {', '.join(list(matching_skills)[:3])}")
            else:
                reasons.append(f"Good skill match: You have relevant skills in {', '.join(matching_skills)}")
        
        # CGPA reasons
        candidate_cgpa = float(candidate.get('cgpa', 7.0))
        min_cgpa = float(internship.get('min_cgpa', 6.0))
        if candidate_cgpa > min_cgpa + 1:
            reasons.append(f"Excellent academic performance: Your CGPA ({candidate_cgpa}) exceeds requirements ({min_cgpa})")
        elif candidate_cgpa >= min_cgpa:
            reasons.append(f"Meets academic requirements: CGPA {candidate_cgpa} (required: {min_cgpa})")
        
        # Location reasons
        preferred_locations = candidate.get('preferred_locations', [])
        internship_location = internship.get('location', '')
        if preferred_locations:
            for pref_loc in preferred_locations:
                if pref_loc.lower() in internship_location.lower():
                    reasons.append(f"Perfect location match: {internship_location} is in your preferred locations")
                    break
        
        # Experience reasons
        experience = int(candidate.get('experience_months', 0))
        if experience >= 12:
            reasons.append(f"Valuable experience: {experience} months of relevant experience")
        elif experience > 0:
            reasons.append(f"Some relevant experience: {experience} months")
        
        # Growth potential
        education_level = candidate.get('education_level', "Bachelor's")
        if education_level in ["Master's", "PhD"]:
            reasons.append(f"High growth potential with {education_level} background")
        
        # Stipend attractiveness
        stipend = internship.get('stipend', 0)
        if stipend >= 30000:
            reasons.append(f"Competitive stipend: Rs.{stipend}/month")
        elif stipend >= 20000:
            reasons.append(f"Good stipend: Rs.{stipend}/month")
        
        return reasons[:5]  # Return top 5 reasons
    
    def get_skill_based_recommendations(self, username: str, skill_focus: str = None) -> List[Dict]:
        """Get recommendations based on specific skill focus"""
        try:
            candidate_id = f"CAND_{hashlib.md5(username.encode()).hexdigest()[:8].upper()}"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,))
            candidate_row = cursor.fetchone()
            
            if not candidate_row:
                conn.close()
                return []
            
            candidate_data = json.loads(candidate_row[3])
            candidate_skills = [s.lower() for s in candidate_data.get('skills', [])]
            
            # Filter for specific skill group if specified
            if skill_focus and skill_focus in self.skill_groups:
                focus_skills = [s.lower() for s in self.skill_groups[skill_focus]]
                relevant_skills = [s for s in candidate_skills if s in focus_skills]
                if not relevant_skills:
                    print(f"No skills found in {skill_focus} category")
                    conn.close()
                    return []
            
            # Get internships with skill requirements matching the focus (with capacity > 0)
            cursor.execute("SELECT * FROM internships WHERE current_capacity > 0")
            internship_rows = cursor.fetchall()
            conn.close()
            
            skill_recommendations = []
            
            for row in internship_rows:
                try:
                    internship = json.loads(row[2])
                    required_skills = [s.lower() for s in internship.get('required_skills', [])]
                    
                    if skill_focus:
                        focus_skills = [s.lower() for s in self.skill_groups[skill_focus]]
                        # Only include if internship requires skills in the focus area
                        if not any(skill in required_skills for skill in focus_skills):
                            continue
                    
                    skill_score = self.get_skill_similarity_score(candidate_skills, required_skills)
                    
                    if skill_score > 10:  # Minimum threshold
                        recommendation = {
                            'internship': internship,
                            'skill_score': skill_score,
                            'matching_skills': list(set(candidate_skills) & set(required_skills)),
                            'skill_gap': [skill for skill in required_skills if skill not in candidate_skills]
                        }
                        skill_recommendations.append(recommendation)
                        
                except Exception as e:
                    continue
            
            # Sort by skill score
            skill_recommendations.sort(key=lambda x: x['skill_score'], reverse=True)
            return skill_recommendations[:5]
            
        except Exception as e:
            print(f"Error generating skill-based recommendations: {e}")
            return []

class MLAllocationEngine:
    """Machine Learning based allocation engine for intelligent student-internship matching"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.model = None
        self.feature_scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
        
        # Advanced feature engineering parameters
        self.skill_vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        self.skill_similarity_threshold = 0.3
        
        # Model performance tracking
        self.training_accuracy = 0.0
        self.validation_accuracy = 0.0
        
        # Load or train model
        self._load_or_train_model()
    
    def _load_or_train_model(self):
        """Load existing ML model or train new one"""
        model_path = "models/allocation_model.pkl"
        scaler_path = "models/allocation_scaler.pkl"
        encoders_path = "models/allocation_encoders.pkl"
        
        try:
            # Try loading existing model
            self.model = joblib.load(model_path)
            self.feature_scaler = joblib.load(scaler_path)
            self.label_encoders = joblib.load(encoders_path)
            self.is_trained = True
            print("✅ Loaded existing ML allocation model")
        except (FileNotFoundError, Exception) as e:
            print("🔄 Training new ML allocation model...")
            self._train_allocation_model()
    
    def _train_allocation_model(self):
        """Train ML model using historical allocation data and synthetic data"""
        try:
            # Collect training data from multiple sources
            training_data = self._collect_training_data()
            
            if len(training_data) < 50:  # Need minimum data for ML
                print("⚠️  Insufficient training data. Generating synthetic data...")
                training_data.extend(self._generate_synthetic_training_data(200))
            
            if len(training_data) < 20:
                print("❌ Cannot train ML model with insufficient data. Using rule-based fallback.")
                return
            
            print(f"📊 Training ML model with {len(training_data)} samples...")
            
            # Prepare features and labels
            X, y = self._prepare_features_and_labels(training_data)
            
            if X.shape[0] == 0:
                print("❌ No valid features extracted. Using rule-based fallback.")
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Train ensemble model
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            self.training_accuracy = train_score
            self.validation_accuracy = test_score
            
            print(f"✅ Model trained successfully!")
            print(f"   Training accuracy: {train_score:.3f}")
            print(f"   Validation accuracy: {test_score:.3f}")
            
            # Save model
            os.makedirs("models", exist_ok=True)
            joblib.dump(self.model, "models/allocation_model.pkl")
            joblib.dump(self.feature_scaler, "models/allocation_scaler.pkl")
            joblib.dump(self.label_encoders, "models/allocation_encoders.pkl")
            
            self.is_trained = True
            
        except Exception as e:
            print(f"❌ Error training ML model: {e}")
            self.is_trained = False
    
    def _collect_training_data(self) -> List[Dict]:
        """Collect training data from various sources"""
        training_data = []
        
        try:
            # Get historical allocations if available
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get successful allocations with feedback
            cursor.execute("""
                SELECT c.data, i.data, a.match_score, 
                       COALESCE(f.satisfaction_rating, 4.0) as satisfaction
                FROM allocations a
                JOIN candidates c ON a.candidate_id = c.id
                JOIN internships i ON a.internship_id = i.id
                LEFT JOIN feedback f ON a.id = f.allocation_id
                WHERE a.status = 'Allocated'
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            for row in rows:
                try:
                    candidate = json.loads(row[0])
                    internship = json.loads(row[1])
                    match_score = float(row[2])
                    satisfaction = float(row[3])
                    
                    # Convert satisfaction to success score (0-100)
                    success_score = min(100, (satisfaction / 5.0) * 100)
                    
                    training_data.append({
                        'candidate': candidate,
                        'internship': internship,
                        'success_score': success_score,
                        'source': 'historical'
                    })
                except Exception as e:
                    continue
            
            print(f"📈 Collected {len(training_data)} historical training samples")
            
        except Exception as e:
            print(f"⚠️  Could not collect historical data: {e}")
        
        return training_data
    
    def _generate_synthetic_training_data(self, count: int) -> List[Dict]:
        """Generate synthetic training data for ML model"""
        synthetic_data = []
        
        # Get available internships
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM internships")
            internship_rows = cursor.fetchall()
            conn.close()
            
            internships = [json.loads(row[0]) for row in internship_rows]
        except:
            print("⚠️  Could not load internships for synthetic data")
            return []
        
        if not internships:
            return []
        
        print(f"🤖 Generating {count} synthetic training samples...")
        
        # Generate synthetic candidates and calculate success scores
        for i in range(count):
            # Create synthetic candidate
            candidate = self._generate_synthetic_candidate()
            
            # Pick random internship
            internship = random.choice(internships)
            
            # Calculate realistic success score based on compatibility
            success_score = self._calculate_synthetic_success_score(candidate, internship)
            
            synthetic_data.append({
                'candidate': candidate,
                'internship': internship,
                'success_score': success_score,
                'source': 'synthetic'
            })
        
        return synthetic_data
    
    def _generate_synthetic_candidate(self) -> Dict:
        """Generate a realistic synthetic candidate"""
        fields = list(FIELD_CATEGORIES.keys())
        field = random.choice(fields)
        
        # Generate realistic skills based on field
        field_skills = FIELD_CATEGORIES.get(field, [])
        base_skills = ["Communication", "Teamwork", "Problem Solving", "Leadership"]
        
        # Select 3-8 skills with field bias
        num_skills = random.randint(3, 8)
        skills = random.sample(field_skills, min(len(field_skills), num_skills // 2))
        skills.extend(random.sample(base_skills, min(len(base_skills), num_skills - len(skills))))
        
        # Add some random technical skills
        tech_skills = ["Python", "Java", "SQL", "Excel", "Analytics", "Project Management"]
        skills.extend(random.sample(tech_skills, random.randint(0, 2)))
        
        # Generate synthetic ID
        synth_id = f"SYNTH_{random.randint(1000, 9999)}"
        
        return {
            'id': synth_id,
            'field_of_study': field,
            'skills': skills,
            'cgpa': round(random.uniform(6.0, 9.5), 1),
            'experience_months': random.randint(0, 36),
            'education_level': random.choice(["Bachelor's", "Master's", "PhD"]),
            'preferred_locations': random.sample(INDIAN_CITIES, random.randint(1, 3)),
            'district_type': random.choice(["Urban", "Rural", "Aspirational District"]),
            'social_category': random.choice(["General", "OBC", "SC", "ST"]),
            'gender': random.choice(["Male", "Female"]),
            'past_participation': random.choice([True, False])
        }
    
    def _calculate_synthetic_success_score(self, candidate: Dict, internship: Dict) -> float:
        """Calculate realistic success score for synthetic data"""
        score = 50  # Base score
        
        # Field alignment impact (±30 points)
        candidate_field = candidate.get('field_of_study', '').lower()
        internship_sector = internship.get('sector', '').lower()
        
        if candidate_field in internship_sector or internship_sector in candidate_field:
            score += 25
        elif any(keyword in internship_sector for keyword in ['technology', 'it'] if 'engineering' in candidate_field):
            score += 15
        else:
            score -= 15
        
        # CGPA impact (±20 points)
        candidate_cgpa = float(candidate.get('cgpa', 7.0))
        min_cgpa = float(internship.get('min_cgpa', 6.0))
        
        cgpa_diff = candidate_cgpa - min_cgpa
        if cgpa_diff >= 1.5:
            score += 20
        elif cgpa_diff >= 0.5:
            score += 10
        elif cgpa_diff >= 0:
            score += 5
        else:
            score -= 20
        
        # Skills matching impact (±25 points)
        candidate_skills = set(s.lower() for s in candidate.get('skills', []))
        required_skills = set(s.lower() for s in internship.get('required_skills', []))
        
        if required_skills:
            skill_overlap = len(candidate_skills & required_skills) / len(required_skills)
            score += skill_overlap * 25
        
        # Location preference impact (±15 points)
        preferred_locations = [loc.lower() for loc in candidate.get('preferred_locations', [])]
        internship_location = internship.get('location', '').lower()
        
        if any(pref in internship_location for pref in preferred_locations):
            score += 15
        elif preferred_locations:  # Has preferences but no match
            score -= 10
        
        # Experience impact (±10 points)
        experience = int(candidate.get('experience_months', 0))
        if experience >= 24:
            score += 10
        elif experience >= 12:
            score += 5
        elif experience == 0:
            score -= 5
        
        # Add some randomness (±10 points) to simulate real-world variance
        score += random.uniform(-10, 10)
        
        # Ensure score is in valid range
        return max(0, min(100, score))
    
    def _prepare_features_and_labels(self, training_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from training data"""
        features = []
        labels = []
        
        for data_point in training_data:
            try:
                candidate = data_point['candidate']
                internship = data_point['internship']
                success_score = data_point['success_score']
                
                feature_vector = self._extract_features(candidate, internship)
                if feature_vector is not None:
                    features.append(feature_vector)
                    labels.append(success_score)
                    
            except Exception as e:
                continue
        
        if not features:
            return np.array([]), np.array([])
        
        return np.array(features), np.array(labels)
    
    def _extract_features(self, candidate: Dict, internship: Dict) -> np.ndarray:
        """Extract numerical features for ML model"""
        try:
            features = []
            
            # Candidate features
            features.append(float(candidate.get('cgpa', 7.0)))
            features.append(float(candidate.get('experience_months', 0)))
            
            # Education level encoding
            education_level = candidate.get('education_level', "Bachelor's")
            edu_encoding = {"Bachelor's": 1, "Master's": 2, "PhD": 3}
            features.append(edu_encoding.get(education_level, 1))
            
            # Field matching (binary)
            candidate_field = candidate.get('field_of_study', '').lower()
            internship_sector = internship.get('sector', '').lower()
            field_match = 1 if (candidate_field in internship_sector or internship_sector in candidate_field) else 0
            features.append(field_match)
            
            # CGPA vs requirement
            min_cgpa = float(internship.get('min_cgpa', 6.0))
            cgpa_diff = float(candidate.get('cgpa', 7.0)) - min_cgpa
            features.append(cgpa_diff)
            
            # Skills similarity
            candidate_skills = set(s.lower() for s in candidate.get('skills', []))
            required_skills = set(s.lower() for s in internship.get('required_skills', []))
            
            if required_skills:
                skill_jaccard = len(candidate_skills & required_skills) / len(candidate_skills | required_skills)
                skill_overlap = len(candidate_skills & required_skills) / len(required_skills)
            else:
                skill_jaccard = 0
                skill_overlap = 0
            
            features.append(skill_jaccard)
            features.append(skill_overlap)
            
            # Location preference match (binary)
            preferred_locations = [loc.lower() for loc in candidate.get('preferred_locations', [])]
            internship_location = internship.get('location', '').lower()
            location_match = 1 if any(pref in internship_location for pref in preferred_locations) else 0
            features.append(location_match)
            
            # Diversity features
            district_encoding = {"Urban": 1, "Rural": 2, "Aspirational District": 3}
            features.append(district_encoding.get(candidate.get('district_type', 'Urban'), 1))
            
            social_encoding = {"General": 1, "OBC": 2, "SC": 3, "ST": 4}
            features.append(social_encoding.get(candidate.get('social_category', 'General'), 1))
            
            gender_encoding = {"Male": 1, "Female": 2, "Other": 3}
            features.append(gender_encoding.get(candidate.get('gender', 'Male'), 1))
            
            # Past participation (binary)
            features.append(1 if candidate.get('past_participation', False) else 0)
            
            # Internship features
            features.append(float(internship.get('stipend', 20000)))
            features.append(float(internship.get('duration_months', 6)))
            
            # Number of required skills
            features.append(len(internship.get('required_skills', [])))
            
            # Number of candidate skills
            features.append(len(candidate.get('skills', [])))
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def predict_compatibility(self, candidate: Dict, internship: Dict) -> float:
        """Predict compatibility score using ML model"""
        if not self.is_trained or self.model is None:
            return None
        
        try:
            # Extract features
            features = self._extract_features(candidate, internship)
            if features is None:
                return None
            
            # Scale features
            features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
            
            # Predict
            predicted_score = self.model.predict(features_scaled)[0]
            
            # Ensure score is in valid range
            return max(0, min(100, predicted_score))
            
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            return None
    
    def get_model_info(self) -> Dict:
        """Get information about the ML model"""
        return {
            'is_trained': self.is_trained,
            'model_type': type(self.model).__name__ if self.model else 'None',
            'training_accuracy': self.training_accuracy,
            'validation_accuracy': self.validation_accuracy,
            'feature_count': len(self._extract_features(
                {'cgpa': 7.0, 'skills': [], 'field_of_study': 'test'}, 
                {'required_skills': [], 'sector': 'test', 'stipend': 20000}
            )) if self.is_trained else 0
        }


class InternshipAllocationSystem:
    """Complete Internship Allocation System with ML Training and Recommendations"""
    
    def __init__(self):
        print("Initializing Enhanced ML Internship Allocation System with Company Selection...")
        self.db_path = "internship_allocation.db"
        self.student_csv = "registered_students.csv"  # For students from student_resume folder
        self.allocation_csv = "student_allocations.csv"  # New allocation CSV for student_resume data
        self.feedback_csv = "feedback.csv"
        self.TRAINING_DATA_FOLDER = "data"  # Only for ML training
        self.STUDENT_RESUME_FOLDER = "student_resume"  # For actual student resumes
        
        self.resume_parser = MLResumeParser()
        self.recommendation_engine = InternshipRecommendationEngine(self.db_path)
        
        # Initialize ML allocation engine
        print("🤖 Initializing ML-based allocation engine...")
        self.ml_allocation_engine = MLAllocationEngine(self.db_path)
        
        # Connect ML engine to recommendation engine
        self.recommendation_engine.set_ml_engine(self.ml_allocation_engine)
        
        self.init_database()
        self.init_csv_files()
        self.init_internships()
        print("Enhanced system with ML allocation and company selection initialized successfully!")
    
    def init_database(self):
        """Initialize database with proper capacity tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS candidates (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE,
                field TEXT,
                data TEXT,
                created_at TEXT,
                source TEXT DEFAULT 'student_resume'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS internships (
                id TEXT PRIMARY KEY,
                company_name TEXT,
                data TEXT,
                current_capacity INTEGER DEFAULT 0,
                original_capacity INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS allocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT,
                internship_id TEXT,
                match_score REAL,
                allocated_date TEXT,
                status TEXT,
                source TEXT DEFAULT 'student_resume',
                UNIQUE(candidate_id, internship_id)
            )
        ''')
        
        # Add original_capacity column if it doesn't exist
        try:
            cursor.execute("ALTER TABLE internships ADD COLUMN original_capacity INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        conn.commit()
        conn.close()
    
    def init_csv_files(self):
        """Initialize CSV files"""
        # Registered students CSV (from student_resume folder)
        student_columns = [
            'candidate_id', 'username', 'name', 'email', 'phone', 'age',
            'field_of_study', 'education_level', 'cgpa', 'skills',
            'experience_months', 'resume_path', 'registration_date',
            'preferred_locations', 'district_type', 'social_category',
            'gender', 'past_participation', 'source'
        ]
        
        if not os.path.exists(self.student_csv):
            df = pd.DataFrame(columns=student_columns)
            df.to_csv(self.student_csv, index=False)
        
        # Student allocations CSV (separate from training data)
        allocation_columns = [
            'candidate_id', 'username', 'company_name', 'position', 'location',
            'sector', 'stipend', 'match_score', 'allocated_date', 'status',
            'location_match_score', 'diversity_bonus', 'internship_id', 'source'
        ]
        
        if not os.path.exists(self.allocation_csv):
            df = pd.DataFrame(columns=allocation_columns)
            df.to_csv(self.allocation_csv, index=False)
        
        # Feedback CSV
        feedback_columns = [
            'allocation_id', 'candidate_id', 'internship_id', 'feedback_score',
            'completion_status', 'satisfaction_rating', 'feedback_date', 'source'
        ]
        
        if not os.path.exists(self.feedback_csv):
            df = pd.DataFrame(columns=feedback_columns)
            df.to_csv(self.feedback_csv, index=False)
    
    def init_internships(self):
        """Initialize internships with proper capacity tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM internships")
            count = cursor.fetchone()[0]
            
            if count > 0:
                # Update original_capacity for existing internships if not set
                cursor.execute("SELECT id, current_capacity FROM internships WHERE original_capacity = 0")
                internships_to_update = cursor.fetchall()
                for int_id, current_cap in internships_to_update:
                    cursor.execute("UPDATE internships SET original_capacity = ? WHERE id = ?", (current_cap, int_id))
                conn.commit()
                conn.close()
                return
                
            conn.close()
        except:
            pass
        
        companies = [
            {"name": "TCS", "sector": "Technology", "location": "Mumbai", "skills": ["Python", "Java", "SQL"], "capacity": 5},
            {"name": "Infosys", "sector": "Technology", "location": "Bangalore", "skills": ["JavaScript", "React", "Node.js"], "capacity": 7},
            {"name": "Wipro", "sector": "Technology", "location": "Pune", "skills": ["C++", "Java", "Machine Learning"], "capacity": 4},
            {"name": "HDFC Bank", "sector": "Finance", "location": "Mumbai", "skills": ["Finance", "Excel", "Analytics"], "capacity": 6},
            {"name": "ICICI Bank", "sector": "Finance", "location": "Delhi", "skills": ["Banking", "Customer Service", "Sales"], "capacity": 3},
            {"name": "Apollo Hospitals", "sector": "Healthcare", "location": "Chennai", "skills": ["Healthcare", "Communication", "Research"], "capacity": 4},
            {"name": "Deloitte", "sector": "Consulting", "location": "Gurugram", "skills": ["Strategy", "Analytics", "Leadership"], "capacity": 5},
            {"name": "Amazon", "sector": "Technology", "location": "Hyderabad", "skills": ["AWS", "Python", "DevOps"], "capacity": 8},
            {"name": "Reliance Industries", "sector": "Business", "location": "Mumbai", "skills": ["Business Development", "Strategy", "Operations"], "capacity": 6},
            {"name": "Tata Motors", "sector": "Automobile", "location": "Pune", "skills": ["Mechanical", "Design", "Manufacturing"], "capacity": 4},
            {"name": "Flipkart", "sector": "Technology", "location": "Bangalore", "skills": ["Java", "Python", "Analytics"], "capacity": 7},
            {"name": "Microsoft", "sector": "Technology", "location": "Hyderabad", "skills": ["C#", "Azure", "AI"], "capacity": 5},
            {"name": "Google", "sector": "Technology", "location": "Bangalore", "skills": ["Python", "Machine Learning", "Cloud Computing"], "capacity": 6},
            {"name": "IBM", "sector": "Technology", "location": "Pune", "skills": ["AI", "Blockchain", "Cloud Computing"], "capacity": 4},
            {"name": "Accenture", "sector": "Consulting", "location": "Chennai", "skills": ["Consulting", "Digital Marketing", "Analytics"], "capacity": 5}
        ]
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM internships")
            
            count = 0
            for company in companies:
                capacity = company['capacity']
                internship = {
                    'id': f"INT_{count+1:06d}",
                    'company_name': company['name'],
                    'title': f"{company['sector']} Intern",
                    'sector': company['sector'],
                    'location': company['location'],
                    'required_skills': company['skills'] + random.sample(['Communication', 'Teamwork', 'Problem Solving'], 2),
                    'min_cgpa': round(random.uniform(6.0, 7.5), 1),
                    'stipend': random.randint(15000, 45000),
                    'capacity': capacity,
                    'duration_months': random.choice([3, 6, 12]),
                    'description': f"Internship opportunity at {company['name']} in {company['sector']} domain",
                    'requirements': f"Looking for candidates with skills in {', '.join(company['skills'][:3])}"
                }
                
                cursor.execute(
                    "INSERT INTO internships (id, company_name, data, current_capacity, original_capacity) VALUES (?, ?, ?, ?, ?)",
                    (internship['id'], internship['company_name'], 
                     json.dumps(internship), capacity, capacity)
                )
                count += 1
            
            conn.commit()
            conn.close()
            print(f"Initialized {count} internships with proper capacity tracking!")
        except Exception as e:
            print(f"Error initializing internships: {e}")
    
    def check_username_exists(self, username: str) -> bool:
        """Check if username exists in registered students"""
        try:
            df = pd.read_csv(self.student_csv)
            return username in df['username'].values
        except:
            return False
    
    def check_internship_capacity(self, internship_id: str) -> bool:
        """Check if internship has available capacity"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT current_capacity FROM internships WHERE id = ?", (internship_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0] > 0:
                return True
            return False
        except Exception as e:
            print(f"Error checking capacity: {e}")
            return False
    
    def update_internship_capacity(self, internship_id: str, decrement: bool = True) -> bool:
        """Update internship capacity - decrement when allocating, increment when deallocating"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if decrement:
                # Decrement capacity (allocation)
                cursor.execute(
                    "UPDATE internships SET current_capacity = current_capacity - 1 WHERE id = ? AND current_capacity > 0",
                    (internship_id,)
                )
            else:
                # Increment capacity (deallocation) - but don't exceed original capacity
                cursor.execute(
                    "UPDATE internships SET current_capacity = min(current_capacity + 1, original_capacity) WHERE id = ?",
                    (internship_id,)
                )
            
            success = cursor.rowcount > 0
            conn.commit()
            conn.close()
            return success
        except Exception as e:
            print(f"Error updating capacity: {e}")
            return False
    
    def calculate_match_score(self, candidate: Dict, internship: Dict) -> float:
        """ML-enhanced matching score between candidate and internship"""
        # Use ML-based allocation if available
        ml_score = self.ml_allocation_engine.predict_compatibility(candidate, internship)
        if ml_score is not None:
            return ml_score
        
        # Fallback to traditional scoring if ML not available
        return self._calculate_traditional_score(candidate, internship)
    
    def _calculate_traditional_score(self, candidate: Dict, internship: Dict) -> float:
        """Traditional rule-based scoring as fallback"""
        score = 0
        
        # Field matching (30 points)
        candidate_field = candidate.get('field_of_study', '').lower()
        internship_sector = internship.get('sector', '').lower()
        if candidate_field in internship_sector or internship_sector in candidate_field:
            score += 30
        elif any(keyword in internship_sector for keyword in ['technology', 'it'] if 'engineering' in candidate_field):
            score += 25
        else:
            score += 10
        
        # CGPA matching (25 points)
        candidate_cgpa = float(candidate.get('cgpa', 7.0))
        min_cgpa = float(internship.get('min_cgpa', 6.0))
        if candidate_cgpa >= min_cgpa + 1:
            score += 25
        elif candidate_cgpa >= min_cgpa:
            score += 20
        else:
            score += 5
        
        # Location matching (20 points)
        preferred_locations = candidate.get('preferred_locations', [])
        internship_location = internship.get('location', '')
        if preferred_locations:
            for pref_loc in preferred_locations:
                if pref_loc.lower() in internship_location.lower():
                    score += 20
                    break
            else:
                score += 5
        else:
            score += 10
        
        # Experience bonus (15 points)
        experience = int(candidate.get('experience_months', 0))
        if experience >= 12:
            score += 15
        elif experience >= 6:
            score += 10
        elif experience > 0:
            score += 5
        
        # Skills matching (10 points)
        candidate_skills = [s.lower() for s in candidate.get('skills', [])]
        required_skills = [s.lower() for s in internship.get('required_skills', [])]
        if required_skills and candidate_skills:
            matches = len(set(candidate_skills) & set(required_skills))
            if matches > 0:
                score += min(10, matches * 3)
        
        return min(score, 100)
    
    def select_company_from_recommendations(self, username: str) -> bool:
        """Let student select from personalized recommendations and allocate"""
        print("\n" + "="*60)
        print("SELECT YOUR PREFERRED COMPANY FROM RECOMMENDATIONS")
        print("="*60)
        
        # Ensure student exists in database by loading from CSV first
        candidate_id = f"CAND_{hashlib.md5(username.encode()).hexdigest()[:8].upper()}"
        
        # Check if student exists in registered students CSV
        try:
            student_df = pd.read_csv(self.student_csv)
            student = student_df[student_df['candidate_id'] == candidate_id]
            if student.empty:
                print(f"Student {username} not found in registered students. Please register first.")
                return False
        except Exception as e:
            print(f"Error loading student data: {e}")
            return False
        
        # Ensure student exists in database for recommendations
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,))
            existing = cursor.fetchone()
            
            if not existing:
                # Add student to database from CSV data
                print("Adding student to database for recommendations...")
                student_data = student.iloc[0]
                
                # Parse student data
                preferred_locations = []
                if pd.notna(student_data.get('preferred_locations')):
                    preferred_locations = [loc.strip() for loc in str(student_data['preferred_locations']).split(',')]
                
                skills = []
                if pd.notna(student_data.get('skills')):
                    skills = [skill.strip() for skill in str(student_data['skills']).split(',')]
                
                candidate_for_db = {
                    'id': candidate_id,
                    'username': username,
                    'field': str(student_data.get('field_of_study', '')),
                    'name': str(student_data.get('name', username)),
                    'email': str(student_data.get('email', f"{username}@email.com")),
                    'phone': str(student_data.get('phone', '')),
                    'skills': skills,
                    'education_level': str(student_data.get('education_level', "Bachelor's")),
                    'cgpa': float(student_data.get('cgpa', 7.0)),
                    'experience_months': int(student_data.get('experience_months', 0)),
                    'preferred_locations': preferred_locations,
                    'district_type': str(student_data.get('district_type', 'Urban')),
                    'social_category': str(student_data.get('social_category', 'General')),
                    'gender': str(student_data.get('gender', 'Male')),
                    'past_participation': bool(student_data.get('past_participation', False)),
                    'source': 'student_resume'
                }
                
                cursor.execute('''
                    INSERT INTO candidates (id, username, field, data, created_at, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    candidate_id, username, candidate_for_db['field'],
                    json.dumps(candidate_for_db), datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'student_resume'
                ))
                conn.commit()
            
            conn.close()
        except Exception as e:
            print(f"Error ensuring student in database: {e}")
            return False
        
        # Get recommendations
        print("Generating your personalized recommendations...")
        recommendations = self.recommendation_engine.get_recommendations(username, top_n=10)
        
        if not recommendations:
            print(f"No recommendations available for {username}")
            return False
        
        print(f"\nAVAILABLE INTERNSHIP OPTIONS FOR {username.upper()}:")
        print("="*60)
        
        # Display recommendations with selection numbers
        valid_choices = []
        for i, rec in enumerate(recommendations, 1):
            internship = rec['internship']
            scores = rec['scores']
            reasons = rec['recommendation_reasons']
            
            # Check current capacity
            capacity_info = "FULL"
            available = False
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT current_capacity FROM internships WHERE id = ?", (internship['id'],))
                result = cursor.fetchone()
                conn.close()
                if result and result[0] > 0:
                    capacity_info = f"{result[0]} slots available"
                    available = True
                    valid_choices.append(i)
            except:
                pass
            
            status_color = "✅" if available else "❌"
            
            print(f"\n{i:2d}. {status_color} {internship['company_name']} - {internship['title']}")
            print(f"     Location: {internship['location']}")
            print(f"     Sector: {internship['sector']}")
            print(f"     Stipend: Rs.{internship['stipend']:,}/month")
            print(f"     Duration: {internship['duration_months']} months")
            print(f"     Match Score: {scores['total_score']:.1f}/100")
            print(f"     Status: {capacity_info}")
            
            # Show top 2 reasons
            if reasons:
                print(f"     Why recommended:")
                for reason in reasons[:2]:
                    print(f"       • {reason}")
            
            # Show required skills
            required_skills = internship.get('required_skills', [])
            if required_skills:
                print(f"     Required: {', '.join(required_skills[:4])}")
                if len(required_skills) > 4:
                    print(f"       and {len(required_skills)-4} more skills...")
        
        if not valid_choices:
            print("\n❌ No companies have available capacity at this time.")
            return False
        
        print(f"\n{'='*60}")
        print(f"✅ Available options: {', '.join(map(str, valid_choices))}")
        print("0. Cancel selection")
        
        # Get user choice
        while True:
            try:
                choice = input(f"\nSelect your preferred company (0, {', '.join(map(str, valid_choices))}): ").strip()
                
                if choice == "0":
                    print("Selection cancelled.")
                    return False
                
                choice_num = int(choice)
                if choice_num in valid_choices:
                    selected_rec = recommendations[choice_num - 1]
                    break
                else:
                    print(f"Invalid choice. Available options: {', '.join(map(str, valid_choices))}")
            except ValueError:
                print("Please enter a valid number.")
        
        # Confirm selection
        selected_internship = selected_rec['internship']
        selected_scores = selected_rec['scores']
        
        print(f"\n🎯 SELECTED COMPANY: {selected_internship['company_name']}")
        print("="*50)
        print(f"Position: {selected_internship['title']}")
        print(f"Location: {selected_internship['location']}")
        print(f"Stipend: Rs.{selected_internship['stipend']:,}/month")
        print(f"Duration: {selected_internship['duration_months']} months")
        print(f"Match Score: {selected_scores['total_score']:.1f}/100")
        
        confirm = input(f"\nConfirm selection of {selected_internship['company_name']}? (y/n): ").lower()
        if not confirm.startswith('y'):
            print("Selection cancelled.")
            return False
        
        # Proceed with allocation
        return self.allocate_specific_internship(username, selected_internship['id'])
    
    def allocate_specific_internship(self, username: str, internship_id: str) -> bool:
        """Allocate specific internship to student"""
        candidate_id = f"CAND_{hashlib.md5(username.encode()).hexdigest()[:8].upper()}"
        
        # Check if already allocated
        try:
            alloc_df = pd.read_csv(self.allocation_csv)
            if candidate_id in alloc_df['candidate_id'].values:
                existing = alloc_df[alloc_df['candidate_id'] == candidate_id].iloc[0]
                print(f"\n❌ {username} is already allocated to {existing['company_name']}")
                return False
        except:
            pass
        
        # Load student data
        try:
            student_df = pd.read_csv(self.student_csv)
            student = student_df[student_df['candidate_id'] == candidate_id]
            if student.empty:
                print(f"❌ Student {username} not found in registered students.")
                return False
            student_data = student.iloc[0]
        except Exception as e:
            print(f"❌ Error loading student: {e}")
            return False
        
        # Get internship data
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM internships WHERE id = ?", (internship_id,))
            internship_row = cursor.fetchone()
            conn.close()
            
            if not internship_row:
                print("❌ Internship not found.")
                return False
            
            internship = json.loads(internship_row[2])
            current_capacity = internship_row[3]
            
            if current_capacity <= 0:
                print(f"❌ {internship['company_name']} has no available capacity.")
                return False
                
        except Exception as e:
            print(f"❌ Error loading internship: {e}")
            return False
        
        # Parse student data
        preferred_locations = []
        if pd.notna(student_data.get('preferred_locations')):
            preferred_locations = [loc.strip() for loc in str(student_data['preferred_locations']).split(',')]
        
        skills = []
        if pd.notna(student_data.get('skills')):
            skills = [skill.strip() for skill in str(student_data['skills']).split(',')]
        
        candidate = {
            'id': candidate_id,
            'username': username,
            'field_of_study': str(student_data.get('field_of_study', '')),
            'skills': skills,
            'education_level': str(student_data.get('education_level', "Bachelor's")),
            'cgpa': float(student_data.get('cgpa', 7.0)),
            'experience_months': int(student_data.get('experience_months', 0)),
            'preferred_locations': preferred_locations,
            'district_type': str(student_data.get('district_type', 'Urban')),
            'social_category': str(student_data.get('social_category', 'General')),
            'gender': str(student_data.get('gender', 'Male')),
            'past_participation': bool(student_data.get('past_participation', False)),
            'source': str(student_data.get('source', 'student_resume'))
        }
        
        # Calculate scores
        match_score = self.calculate_match_score(candidate, internship)
        
        # Calculate additional scores
        location_match_score = 0
        if preferred_locations and internship.get('location'):
            for loc in preferred_locations:
                if loc.lower() in internship['location'].lower():
                    location_match_score = 1.0
                    break
        
        diversity_bonus = 0
        if candidate['district_type'].lower() in ['rural', 'aspirational district']:
            diversity_bonus += 5
        if candidate['social_category'] in ['SC', 'ST', 'OBC']:
            diversity_bonus += 5
        if candidate['gender'].lower() == 'female':
            diversity_bonus += 5
        if not candidate['past_participation']:
            diversity_bonus += 3
        
        # Check capacity and allocate atomically
        success = self.update_internship_capacity(internship_id, decrement=True)
        if not success:
            print(f"❌ Failed to allocate - {internship['company_name']} capacity exhausted")
            return False
        
        # Save allocation
        allocation_result = self.save_student_allocation(
            candidate_id, username, internship, match_score, 
            location_match_score, diversity_bonus
        )
        
        if allocation_result:
            print(f"\n🎉 ALLOCATION SUCCESSFUL!")
            print(f"Student: {username}")
            print(f"Company: {internship['company_name']}")
            print(f"Position: {internship['title']}")
            print(f"Location: {internship['location']}")
            print(f"Stipend: Rs.{internship['stipend']}/month")
            print(f"Match Score: {match_score:.1f}/100")
            if location_match_score > 0:
                print(f"Location Match: Perfect")
            if diversity_bonus > 0:
                print(f"Diversity Bonus: +{diversity_bonus}")
            
            # Show remaining capacity
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT current_capacity FROM internships WHERE id = ?", (internship_id,))
                remaining = cursor.fetchone()[0]
                conn.close()
                print(f"Remaining capacity at {internship['company_name']}: {remaining}")
            except:
                pass
            
            return True
        else:
            # Rollback capacity if allocation failed
            self.update_internship_capacity(internship_id, decrement=False)
            print(f"❌ Allocation failed - capacity restored")
            return False
    
    def allocate_single_student(self, username: str):
        """Allocate internship to specific student with proper capacity checking"""
        candidate_id = f"CAND_{hashlib.md5(username.encode()).hexdigest()[:8].upper()}"
        
        # Check if already allocated in student allocations
        try:
            alloc_df = pd.read_csv(self.allocation_csv)
            if candidate_id in alloc_df['candidate_id'].values:
                existing = alloc_df[alloc_df['candidate_id'] == candidate_id].iloc[0]
                print(f"\n{username} already allocated!")
                self.display_allocation(existing.to_dict())
                return existing.to_dict()
        except:
            pass
        
        # Load student data from registered students CSV
        try:
            student_df = pd.read_csv(self.student_csv)
            student = student_df[student_df['candidate_id'] == candidate_id]
            if student.empty:
                print(f"Student {username} not found in registered students.")
                return None
            student_data = student.iloc[0]
        except Exception as e:
            print(f"Error loading student: {e}")
            return None
        
        # Parse student data safely
        preferred_locations = []
        if pd.notna(student_data.get('preferred_locations')):
            preferred_locations = [loc.strip() for loc in str(student_data['preferred_locations']).split(',')]
        
        skills = []
        if pd.notna(student_data.get('skills')):
            skills = [skill.strip() for skill in str(student_data['skills']).split(',')]
        
        candidate = {
            'id': candidate_id,
            'username': username,
            'field_of_study': str(student_data.get('field_of_study', '')),
            'skills': skills,
            'education_level': str(student_data.get('education_level', "Bachelor's")),
            'cgpa': float(student_data.get('cgpa', 7.0)),
            'experience_months': int(student_data.get('experience_months', 0)),
            'preferred_locations': preferred_locations,
            'district_type': str(student_data.get('district_type', 'Urban')),
            'social_category': str(student_data.get('social_category', 'General')),
            'gender': str(student_data.get('gender', 'Male')),
            'past_participation': bool(student_data.get('past_participation', False)),
            'source': str(student_data.get('source', 'student_resume'))
        }
        
        # Get available internships WITH capacity > 0
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM internships WHERE current_capacity > 0 ORDER BY current_capacity DESC")
            internship_rows = cursor.fetchall()
            conn.close()
        except Exception as e:
            print(f"Error loading internships: {e}")
            return None
        
        if not internship_rows:
            print("No internships with available capacity.")
            return None
        
        print(f"Found {len(internship_rows)} internships with available capacity")
        
        # Find best match from available internships
        best_match = None
        best_score = 0
        location_match_score = 0
        diversity_bonus = 0
        best_internship_id = None
        
        for row in internship_rows:
            try:
                internship = json.loads(row[2])
                current_capacity = row[3]  # Get current capacity
                
                # Double check capacity
                if current_capacity <= 0:
                    continue
                    
                score = self.calculate_match_score(candidate, internship)
                
                if score > best_score:
                    best_score = score
                    best_match = internship
                    best_internship_id = internship['id']
                    
                    # Calculate location match
                    location_match_score = 0
                    if preferred_locations and internship.get('location'):
                        for loc in preferred_locations:
                            if loc.lower() in internship['location'].lower():
                                location_match_score = 1.0
                                break
                    
                    # Calculate diversity bonus
                    diversity_bonus = 0
                    if candidate['district_type'].lower() in ['rural', 'aspirational district']:
                        diversity_bonus += 5
                    if candidate['social_category'] in ['SC', 'ST', 'OBC']:
                        diversity_bonus += 5
                    if candidate['gender'].lower() == 'female':
                        diversity_bonus += 5
                    if not candidate['past_participation']:
                        diversity_bonus += 3
                        
            except Exception as e:
                print(f"Error processing internship: {e}")
                continue
        
        if best_match and best_score >= 30 and best_internship_id:
            # CRITICAL: Check capacity one more time and update atomically
            if not self.check_internship_capacity(best_internship_id):
                print(f"Internship {best_match['company_name']} no longer has capacity (filled while processing)")
                return None
                
            # Try to allocate (atomic operation)
            success = self.update_internship_capacity(best_internship_id, decrement=True)
            if not success:
                print(f"Failed to allocate - internship {best_match['company_name']} capacity exhausted")
                return None
            
            # Save allocation to student allocations CSV and database
            allocation_result = self.save_student_allocation(
                candidate_id, username, best_match, best_score, 
                location_match_score, diversity_bonus
            )
            
            if allocation_result:
                # Display ML vs traditional score comparison if ML was used
                traditional_score = self._calculate_traditional_score(candidate, best_match)
                
                print(f"\n=== ALLOCATION SUCCESSFUL ===")
                print(f"Student: {username}")
                print(f"Company: {best_match['company_name']}")
                print(f"Position: {best_match['title']}")
                print(f"Location: {best_match['location']}")
                print(f"Stipend: Rs.{best_match['stipend']}/month")
                
                if self.ml_allocation_engine.is_trained:
                    print(f"🤖 ML Match Score: {best_score:.1f}/100")
                    print(f"📈 Traditional Score: {traditional_score:.1f}/100")
                    print(f"🎯 Score Improvement: {best_score - traditional_score:+.1f}")
                else:
                    print(f"Match Score: {best_score:.1f}/100 (Rule-based)")
                
                print(f"Location Match: {location_match_score}")
                print(f"Diversity Bonus: +{diversity_bonus}")
                
                # Show remaining capacity
                try:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT current_capacity FROM internships WHERE id = ?", (best_internship_id,))
                    remaining = cursor.fetchone()[0]
                    conn.close()
                    print(f"Remaining capacity at {best_match['company_name']}: {remaining}")
                except:
                    pass
                
                print(f"===============================")
                
                return allocation_result
            else:
                # Rollback capacity if allocation failed
                self.update_internship_capacity(best_internship_id, decrement=False)
                print(f"Allocation failed - capacity restored")
                return None
        else:
            print(f"No suitable internship found for {username} (best score: {best_score:.1f})")
            if not self.ml_allocation_engine.is_trained:
                print("💡 Tip: Train ML model with more data for better matching")
            return None
    
    def save_student_allocation(self, candidate_id: str, username: str, internship: Dict, 
                               match_score: float, location_match: float, diversity_bonus: float) -> Dict:
        """Save allocation to student allocations CSV and database"""
        try:
            # Save to database first (with unique constraint)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT INTO allocations (candidate_id, internship_id, match_score, allocated_date, status, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    candidate_id, internship['id'], match_score,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Allocated', 'student_resume'
                ))
                conn.commit()
            except sqlite3.IntegrityError:
                print(f"Student {username} already has an allocation in database")
                conn.close()
                return None
            
            conn.close()
            
            # Save to CSV
            df = pd.read_csv(self.allocation_csv)
            
            # Check if already exists in CSV
            if candidate_id in df['candidate_id'].values:
                print(f"Student {username} already allocated in CSV")
                return None
            
            new_row = pd.DataFrame([{
                'candidate_id': candidate_id,
                'username': username,
                'company_name': internship['company_name'],
                'position': internship.get('title', 'Intern'),
                'location': internship['location'],
                'sector': internship.get('sector', ''),
                'stipend': internship.get('stipend', 0),
                'match_score': match_score,
                'allocated_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'Allocated',
                'location_match_score': location_match,
                'diversity_bonus': diversity_bonus,
                'internship_id': internship.get('id', ''),
                'source': 'student_resume'
            }])
            
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.allocation_csv, index=False)
            print("Allocation saved successfully!")
            
            return new_row.iloc[0].to_dict()
            
        except Exception as e:
            print(f"Error saving student allocation: {e}")
            return None
    
    def view_student_allocation(self):
        """View allocation for specific student from student_resume folder"""
        username = input("\nEnter your username: ").strip()
        if not username:
            return
        
        candidate_id = f"CAND_{hashlib.md5(username.encode()).hexdigest()[:8].upper()}"
        
        try:
            # Only check student allocations CSV (not training data)
            df = pd.read_csv(self.allocation_csv)
            allocation = df[df['candidate_id'] == candidate_id]
            if not allocation.empty:
                print(f"\nShowing allocation from student_resume folder:")
                self.display_allocation(allocation.iloc[0].to_dict())
            else:
                print(f"No allocation found for {username} in student records")
        except Exception as e:
            print(f"Error: {e}")
    
    def display_allocation(self, allocation: Dict):
        """Display allocation details"""
        print("\n" + "="*60)
        print("STUDENT INTERNSHIP ALLOCATION DETAILS")
        print("="*60)
        print(f"Student: {allocation.get('username', 'N/A')}")
        print(f"Company: {allocation.get('company_name', 'N/A')}")
        print(f"Position: {allocation.get('position', 'N/A')}")
        print(f"Location: {allocation.get('location', 'N/A')}")
        print(f"Sector: {allocation.get('sector', 'N/A')}")
        print(f"Stipend: Rs.{allocation.get('stipend', 0)}/month")
        print(f"Match Score: {allocation.get('match_score', 0):.1f}/100")
        
        if 'location_match_score' in allocation and pd.notna(allocation['location_match_score']):
            print(f"Location Match: {allocation['location_match_score']}")
        if 'diversity_bonus' in allocation and pd.notna(allocation['diversity_bonus']):
            print(f"Diversity Bonus: +{allocation['diversity_bonus']}")
        
        print(f"Date: {allocation.get('allocated_date', 'N/A')}")
        print(f"Source: {allocation.get('source', 'student_resume')}")
        print("="*60)
    
    def get_system_insights(self):
        """Display system insights for student_resume data only"""
        try:
            # Only show insights from student_resume folder data
            student_df = pd.read_csv(self.student_csv) if os.path.exists(self.student_csv) else pd.DataFrame()
            alloc_df = pd.read_csv(self.allocation_csv) if os.path.exists(self.allocation_csv) else pd.DataFrame()
            
            print("\n" + "="*60)
            print("SYSTEM INSIGHTS (Student Resume Data Only)")
            print("="*60)
            
            # Student analytics from student_resume folder
            if not student_df.empty:
                print(f"\nREGISTERED STUDENT ANALYTICS:")
                print(f"  Total registered students: {len(student_df)}")
                print(f"  Average CGPA: {student_df['cgpa'].mean():.2f}")
                print(f"  Average experience: {student_df['experience_months'].mean():.1f} months")
                
                # Field distribution
                field_dist = student_df['field_of_study'].value_counts().head(5)
                print(f"\n  Top fields:")
                for field, count in field_dist.items():
                    print(f"    {field}: {count}")
                
                # Diversity breakdown
                if 'gender' in student_df.columns:
                    gender_dist = student_df['gender'].value_counts()
                    print(f"\n  Gender distribution:")
                    for gender, count in gender_dist.items():
                        print(f"    {gender}: {count} ({count/len(student_df)*100:.1f}%)")
                
                if 'social_category' in student_df.columns:
                    social_dist = student_df['social_category'].value_counts()
                    print(f"\n  Social category:")
                    for category, count in social_dist.items():
                        print(f"    {category}: {count} ({count/len(student_df)*100:.1f}%)")
            
            # Allocation analytics from student allocations
            if not alloc_df.empty:
                print(f"\nSTUDENT ALLOCATION ANALYTICS:")
                print(f"  Total student allocations: {len(alloc_df)}")
                print(f"  Average match score: {alloc_df['match_score'].mean():.1f}/100")
                print(f"  Average stipend: Rs.{alloc_df['stipend'].mean():.0f}/month")
                
                # Top companies for students
                company_dist = alloc_df['company_name'].value_counts().head(3)
                print(f"\n  Top companies for students:")
                for company, count in company_dist.items():
                    print(f"    {company}: {count} allocations")
            
            # Internship capacity status
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT company_name, current_capacity, original_capacity FROM internships ORDER BY current_capacity DESC")
                capacity_data = cursor.fetchall()
                conn.close()
                
                print(f"\nINTERNSHIP CAPACITY STATUS:")
                total_remaining = 0
                total_original = 0
                for company, current, original in capacity_data:
                    allocated = original - current
                    print(f"  {company}: {current}/{original} remaining ({allocated} allocated)")
                    total_remaining += current
                    total_original += original
                
                print(f"\n  TOTAL CAPACITY:")
                print(f"    Original capacity: {total_original}")
                print(f"    Remaining capacity: {total_remaining}")
                print(f"    Students allocated: {total_original - total_remaining}")
                
            except Exception as e:
                print(f"  Error getting capacity data: {e}")
            
            print("="*60)
            
        except Exception as e:
            print(f"Error generating insights: {e}")
    
    def view_training_vs_student_summary(self):
        """Show summary of training data vs student data"""
        print("\n" + "="*60)
        print("TRAINING VS STUDENT DATA SUMMARY")
        print("="*60)
        
        # Training data summary
        training_count = 0
        print("TRAINING DATA (data/ folder):")
        if os.path.exists(self.TRAINING_DATA_FOLDER):
            for field in FIELD_CATEGORIES.keys():
                field_path = os.path.join(self.TRAINING_DATA_FOLDER, field)
                if os.path.exists(field_path):
                    pdf_count = len([f for f in os.listdir(field_path) if f.endswith('.pdf')])
                    if pdf_count > 0:
                        print(f"  {field}: {pdf_count} PDFs")
                        training_count += pdf_count
        print(f"Total training files: {training_count}")
        
        # Student data summary
        student_count = 0
        print(f"\nSTUDENT DATA ({self.STUDENT_RESUME_FOLDER}/ folder):")
        if os.path.exists(self.STUDENT_RESUME_FOLDER):
            for field in FIELD_CATEGORIES.keys():
                field_path = os.path.join(self.STUDENT_RESUME_FOLDER, field)
                if os.path.exists(field_path):
                    pdf_count = len([f for f in os.listdir(field_path) if f.endswith('.pdf')])
                    if pdf_count > 0:
                        print(f"  {field}: {pdf_count} PDFs")
                        student_count += pdf_count
        print(f"Total student files: {student_count}")
        
        # CSV summary
        try:
            reg_df = pd.read_csv(self.student_csv) if os.path.exists(self.student_csv) else pd.DataFrame()
            alloc_df = pd.read_csv(self.allocation_csv) if os.path.exists(self.allocation_csv) else pd.DataFrame()
            print(f"\nCSV DATA:")
            print(f"  Registered students: {len(reg_df)}")
            print(f"  Student allocations: {len(alloc_df)}")
        except:
            print(f"\nCSV DATA: Error reading files")
        
        print("="*60)
    
    def view_ml_model_info(self):
        """Display ML model information and performance metrics"""
        print("\n" + "="*60)
        print("🤖 MACHINE LEARNING MODEL INFORMATION")
        print("="*60)
        
        model_info = self.ml_allocation_engine.get_model_info()
        
        print(f"Model Status: {'\u2705 Trained' if model_info['is_trained'] else '\u274c Not Trained'}")
        print(f"Model Type: {model_info['model_type']}")
        
        if model_info['is_trained']:
            print(f"Training Accuracy: {model_info['training_accuracy']:.3f}")
            print(f"Validation Accuracy: {model_info['validation_accuracy']:.3f}")
            print(f"Feature Count: {model_info['feature_count']}")
            
            # Performance interpretation
            val_acc = model_info['validation_accuracy']
            if val_acc >= 0.8:
                performance = "🚀 Excellent"
            elif val_acc >= 0.6:
                performance = "📊 Good"
            elif val_acc >= 0.4:
                performance = "⚠️  Fair"
            else:
                performance = "🔴 Needs Improvement"
            
            print(f"Performance Level: {performance}")
            
            # Show allocation method currently used
            print(f"\nAllocation Method: {'\u2699️  ML-Enhanced' if model_info['is_trained'] else '\ud83d\ufffd Rule-Based'}")
            
            if model_info['is_trained']:
                print(f"Allocation Process:")
                print(f"  1. 🤖 ML model predicts compatibility score")
                print(f"  2. 🔄 Falls back to rule-based if ML fails")
                print(f"  3. 🎯 Uses higher of ML vs traditional score")
        else:
            print(f"\n💡 Benefits of ML-based allocation:")
            print(f"  • Learns from successful placements")
            print(f"  • Adapts to student preferences")
            print(f"  • Improves matching accuracy over time")
            print(f"  • Considers complex feature interactions")
            
            print(f"\n🔄 To enable ML allocation:")
            print(f"  • Add more historical allocation data")
            print(f"  • Collect student feedback on placements")
            print(f"  • System will auto-train when sufficient data available")
        
        # Show data sources
        print(f"\nData Sources for Training:")
        print(f"  • Historical allocation outcomes")
        print(f"  • Student feedback and satisfaction ratings")
        print(f"  • Synthetic data based on domain expertise")
        print(f"  • Resume parsing results and skill analysis")
        
        print("="*60)
    
    def retrain_ml_model(self):
        """Manually retrain the ML model"""
        print("\n" + "="*60)
        print("🔄 RETRAINING ML ALLOCATION MODEL")
        print("="*60)
        
        confirm = input("\nThis will retrain the ML model with current data. Continue? (y/n): ")
        if not confirm.lower().startswith('y'):
            print("Retraining cancelled.")
            return
        
        print("\n📈 Initiating model retraining...")
        
        # Backup old model
        try:
            import shutil
            import datetime
            backup_dir = f"models/backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(backup_dir, exist_ok=True)
            
            if os.path.exists("models/allocation_model.pkl"):
                shutil.copy("models/allocation_model.pkl", f"{backup_dir}/allocation_model.pkl")
                print(f"💾 Old model backed up to {backup_dir}")
        except Exception as e:
            print(f"⚠️  Could not backup old model: {e}")
        
        # Retrain
        try:
            self.ml_allocation_engine._train_allocation_model()
            
            # Show updated info
            model_info = self.ml_allocation_engine.get_model_info()
            
            if model_info['is_trained']:
                print(f"\n✅ Model retrained successfully!")
                print(f"New Training Accuracy: {model_info['training_accuracy']:.3f}")
                print(f"New Validation Accuracy: {model_info['validation_accuracy']:.3f}")
            else:
                print(f"\n❌ Model retraining failed. Using rule-based allocation.")
                
        except Exception as e:
            print(f"\n❌ Error during retraining: {e}")
        
        print("="*60)

    def get_personalized_recommendations(self):
        """Get personalized internship recommendations for student"""
        print("\n" + "="*60)
        print("PERSONALIZED INTERNSHIP RECOMMENDATIONS")
        print("="*60)
        
        username = input("\nEnter your username: ").strip()
        if not username:
            return
        
        print("\nGenerating personalized recommendations...")
        recommendations = self.recommendation_engine.get_recommendations(username, top_n=5)
        
        if not recommendations:
            print(f"No recommendations found for {username}")
            return
        
        print(f"\nTOP INTERNSHIP RECOMMENDATIONS FOR {username.upper()}")
        print("="*60)
        
        for i, rec in enumerate(recommendations, 1):
            internship = rec['internship']
            scores = rec['scores']
            reasons = rec['recommendation_reasons']
            
            # Check current capacity
            capacity_info = ""
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT current_capacity FROM internships WHERE id = ?", (internship['id'],))
                result = cursor.fetchone()
                conn.close()
                if result:
                    capacity_info = f" (Available slots: {result[0]})"
            except:
                pass
            
            print(f"\n{i}. {internship['company_name']} - {internship['title']}")
            print("-" * 50)
            print(f"   Location: {internship['location']}")
            print(f"   Sector: {internship['sector']}")
            print(f"   Stipend: Rs.{internship['stipend']:,}/month")
            print(f"   Duration: {internship['duration_months']} months{capacity_info}")
            print(f"   Match Score: {scores['total_score']:.1f}/100")
            
            # Score breakdown
            print(f"\n   Score Breakdown:")
            print(f"      Field Match: {scores['field_match']}/25")
            print(f"      Skill Match: {scores['skill_match']:.1f}/25")
            print(f"      CGPA Match: {scores['cgpa_match']}/20")
            print(f"      Location Match: {scores['location_match']}/15")
            print(f"      Experience: {scores['experience_match']}/10")
            
            # Reasons
            print(f"\n   Why this is recommended:")
            for reason in reasons:
                print(f"      • {reason}")
            
            # Required skills
            required_skills = internship.get('required_skills', [])
            if required_skills:
                print(f"\n   Required Skills: {', '.join(required_skills)}")
            
            print()
    
    def get_skill_focused_recommendations(self):
        """Get recommendations focused on specific skill areas"""
        print("\n" + "="*60)
        print("SKILL-FOCUSED INTERNSHIP RECOMMENDATIONS")
        print("="*60)
        
        username = input("\nEnter your username: ").strip()
        if not username:
            return
        
        # Show available skill groups
        skill_groups = list(self.recommendation_engine.skill_groups.keys())
        print(f"\nAvailable skill focus areas:")
        for i, group in enumerate(skill_groups, 1):
            skills = ', '.join(self.recommendation_engine.skill_groups[group][:3])
            print(f"{i}. {group} ({skills}, ...)")
        
        print("0. All skills (no specific focus)")
        
        try:
            choice = int(input(f"\nSelect focus area (0-{len(skill_groups)}): "))
            skill_focus = None if choice == 0 else skill_groups[choice-1] if 1 <= choice <= len(skill_groups) else None
        except (ValueError, IndexError):
            skill_focus = None
        
        if skill_focus:
            print(f"\nFinding recommendations focused on {skill_focus} skills...")
        else:
            print(f"\nFinding recommendations across all skill areas...")
        
        recommendations = self.recommendation_engine.get_skill_based_recommendations(username, skill_focus)
        
        if not recommendations:
            print(f"No skill-based recommendations found for {username}")
            return
        
        focus_text = f" ({skill_focus} FOCUS)" if skill_focus else ""
        print(f"\nSKILL-BASED RECOMMENDATIONS FOR {username.upper()}{focus_text}")
        print("="*60)
        
        for i, rec in enumerate(recommendations, 1):
            internship = rec['internship']
            
            # Check capacity
            capacity_info = ""
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT current_capacity FROM internships WHERE id = ?", (internship['id'],))
                result = cursor.fetchone()
                conn.close()
                if result:
                    capacity_info = f" (Available slots: {result[0]})"
            except:
                pass
            
            print(f"\n{i}. {internship['company_name']} - {internship['title']}")
            print("-" * 50)
            print(f"   Location: {internship['location']}")
            print(f"   Stipend: Rs.{internship['stipend']:,}/month{capacity_info}")
            print(f"   Skill Match Score: {rec['skill_score']:.1f}/100")
            
            # Matching skills
            matching_skills = rec['matching_skills']
            if matching_skills:
                print(f"   Your matching skills: {', '.join(matching_skills)}")
            
            # Skill gaps
            skill_gaps = rec['skill_gap']
            if skill_gaps:
                print(f"   Skills to develop: {', '.join(skill_gaps[:3])}")
                if len(skill_gaps) > 3:
                    print(f"      and {len(skill_gaps)-3} more...")
            
            # Required skills
            required_skills = internship.get('required_skills', [])
            print(f"   All required skills: {', '.join(required_skills)}")
            print()

    def register_student(self):
        """Register a new student from student_resume folder"""
        print("\n" + "="*60)
        print("STUDENT REGISTRATION (Student Resume Folder)")
        print("="*60)
        
        # Field selection
        fields = list(FIELD_CATEGORIES.keys())
        print("\nAvailable Fields:")
        for i, field in enumerate(fields, 1):
            print(f"{i:2d}. {field}")
        
        while True:
            try:
                choice = int(input(f"\nSelect field (1-{len(fields)}): "))
                if 1 <= choice <= len(fields):
                    field = fields[choice - 1]
                    break
            except ValueError:
                pass
            print("Invalid choice. Try again.")
        
        # Username
        while True:
            username = input("\nEnter unique username: ").strip()
            if username and not self.check_username_exists(username):
                break
            print("Username exists or invalid. Choose another.")
        
        # Basic info
        name = input("Full Name: ").strip() or username
        try:
            age = int(input("Age (default 21): ") or "21")
        except ValueError:
            age = 21
        
        # Location preferences
        print(f"\nLocation preferences (examples: {', '.join(INDIAN_CITIES[:5])})")
        location_input = input("Preferred cities (comma-separated): ").strip()
        preferred_locations = []
        if location_input:
            preferred_locations = [normalize_city_name(loc.strip()) 
                                 for loc in location_input.split(',')]
            preferred_locations = [loc for loc in preferred_locations if loc]
        
        # Diversity info
        print("\nDiversity Information:")
        district_types = ["Urban", "Rural", "Aspirational District"]
        print("District types: 1.Urban 2.Rural 3.Aspirational District")
        district_choice = input("Choice (default 1): ").strip() or "1"
        district_type = district_types[int(district_choice)-1] if district_choice in "123" else "Urban"
        
        social_categories = ["General", "OBC", "SC", "ST"]
        print("Social categories: 1.General 2.OBC 3.SC 4.ST")
        social_choice = input("Choice (default 1): ").strip() or "1"
        social_category = social_categories[int(social_choice)-1] if social_choice in "1234" else "General"
        
        genders = ["Male", "Female", "Other"]
        print("Gender: 1.Male 2.Female 3.Other")
        gender_choice = input("Choice (default 1): ").strip() or "1"
        gender = genders[int(gender_choice)-1] if gender_choice in "123" else "Male"
        
        past_participation = input("Previous internship participation? (y/n): ").lower().startswith('y')
        
        # Resume
        while True:
            resume_path = input("\nResume PDF path: ").strip().strip('"\'')
            if os.path.isfile(resume_path) and resume_path.lower().endswith('.pdf'):
                break
            print("Invalid PDF path. Try again.")
        
        # Store resume in student_resume folder (separated by field)
        field_folder = f"{self.STUDENT_RESUME_FOLDER}/{field}"
        os.makedirs(field_folder, exist_ok=True)
        stored_path = f"{field_folder}/{username}_resume.pdf"
        shutil.copy(resume_path, stored_path)
        
        # Parse resume using ML
        print("Analyzing resume with trained ML models...")
        resume_data = self.resume_parser.parse_resume(stored_path, field, username)
        
        # Create candidate
        candidate = {
            'id': resume_data['id'],
            'username': username,
            'name': resume_data.get('name', name),
            'age': age,
            'field_of_study': resume_data['field'],
            'field': resume_data['field'],
            'email': resume_data['email'],
            'phone': resume_data['phone'],
            'skills': resume_data['skills'],
            'education_level': resume_data['education_level'],
            'cgpa': resume_data['cgpa'],
            'experience_months': resume_data['experience_months'],
            'resume_path': stored_path,
            'registration_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'preferred_locations': preferred_locations,
            'district_type': district_type,
            'social_category': social_category,
            'gender': gender,
            'past_participation': past_participation,
            'source': 'student_resume'
        }
        
        # Save data
        self.save_candidate_to_csv(candidate)
        self.save_candidate_to_db(candidate)
        
        print(f"\nRegistration successful!")
        print(f"ID: {candidate['id']}")
        print(f"Field: {candidate['field_of_study']}")
        print(f"Skills: {len(candidate['skills'])} detected")
        print(f"CGPA: {candidate['cgpa']}")
        print(f"Experience: {candidate['experience_months']} months")
        print(f"Locations: {', '.join(preferred_locations) if preferred_locations else 'None'}")
        print(f"Diversity: {district_type}, {social_category}, {gender}")
        print(f"Resume stored in: {stored_path}")
        
        # Ask for company selection or auto-allocation
        print("\n" + "="*60)
        print("INTERNSHIP ALLOCATION OPTIONS")
        print("="*60)
        print("1. Select from personalized company recommendations")
        print("2. Auto-allocate best match")
        print("3. Skip allocation for now")
        
        allocation_choice = input("\nChoose allocation option (1-3): ").strip()
        
        if allocation_choice == "1":
            print("\nLet's find your perfect company match...")
            success = self.select_company_from_recommendations(username)
            if not success:
                print("Company selection cancelled. You can allocate later using menu option 2.")
        elif allocation_choice == "2":
            print("\nAuto-allocating best matching internship...")
            allocation_result = self.allocate_single_student(username)
        else:
            print("Allocation skipped. You can allocate later using the main menu.")
        
        return candidate
    
    def save_candidate_to_csv(self, candidate: Dict):
        """Save candidate to registered students CSV"""
        try:
            df = pd.read_csv(self.student_csv)
            
            # Check if exists
            if candidate['id'] in df['candidate_id'].values:
                print(f"Candidate {candidate['username']} already exists in registered students CSV")
                return
            
            # Create row
            new_row = pd.DataFrame([{
                'candidate_id': candidate['id'],
                'username': candidate['username'],
                'name': candidate['name'],
                'email': candidate['email'],
                'phone': candidate['phone'],
                'age': candidate['age'],
                'field_of_study': candidate['field_of_study'],
                'education_level': candidate['education_level'],
                'cgpa': candidate['cgpa'],
                'skills': ', '.join(candidate['skills']),
                'experience_months': candidate['experience_months'],
                'resume_path': candidate['resume_path'],
                'registration_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'preferred_locations': ', '.join(candidate['preferred_locations']),
                'district_type': candidate['district_type'],
                'social_category': candidate['social_category'],
                'gender': candidate['gender'],
                'past_participation': candidate['past_participation'],
                'source': candidate['source']
            }])
            
            # Append and save
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.student_csv, index=False)
            print("Student data saved to registered students CSV successfully!")
            
        except Exception as e:
            print(f"Error saving to registered students CSV: {e}")
    
    def save_candidate_to_db(self, candidate: Dict):
        """Save candidate to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO candidates (id, username, field, data, created_at, source)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                candidate['id'], candidate['username'], candidate['field_of_study'],
                json.dumps(candidate), datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                candidate['source']
            ))
            conn.commit()
            conn.close()
            print("Student data saved to database successfully!")
        except Exception as e:
            print(f"Error saving to database: {e}")

def main():
    """Main application with enhanced ML-based allocation and company selection features"""
    print("="*70)
    print("🤖 ENHANCED ML INTERNSHIP ALLOCATION WITH SMART MATCHING")
    print("="*70)
    print("Training Data: data/{field}/ folder")
    print("Student Data: student_resume/{field}/ folder")
    print("Features: ML Training, Smart Allocation, Company Selection, Strict Capacity Control")
    print("="*70)
    
    try:
        system = InternshipAllocationSystem()
        
        # Show ML model status on startup
        model_info = system.ml_allocation_engine.get_model_info()
        if model_info['is_trained']:
            print(f"\n✅ ML Model Status: Trained (Accuracy: {model_info['validation_accuracy']:.2f})")
        else:
            print(f"\n⚠️  ML Model Status: Not trained - using rule-based allocation")
        
        while True:
            print("\nMAIN MENU:")
            print("1. Register Student (With ML-Enhanced Allocation)")
            print("2. Select Company from Recommendations")
            print("3. Get Personalized Recommendations")
            print("4. Get Skill-focused Recommendations")
            print("5. Auto-allocate Specific Student (ML-Enhanced)")
            print("6. View My Allocation")
            print("7. System Insights (Student Data)")
            print("8. Training vs Student Summary")
            print("9. 🤖 View ML Model Information")
            print("10. 🔄 Retrain ML Model")
            print("11. Exit")
            
            choice = input(f"\nEnter choice (1-11): ").strip()
            
            if choice == "1":
                system.register_student()
            elif choice == "2":
                username = input("Enter username: ").strip()
                if username:
                    # Check if already allocated
                    candidate_id = f"CAND_{hashlib.md5(username.encode()).hexdigest()[:8].upper()}"
                    try:
                        alloc_df = pd.read_csv(system.allocation_csv)
                        if candidate_id in alloc_df['candidate_id'].values:
                            existing = alloc_df[alloc_df['candidate_id'] == candidate_id].iloc[0]
                            print(f"\n{username} is already allocated to {existing['company_name']}")
                            continue
                    except:
                        pass
                    
                    success = system.select_company_from_recommendations(username)
                    if success:
                        print("Company selection and allocation completed!")
                    else:
                        print("Company selection was not completed.")
            elif choice == "3":
                system.get_personalized_recommendations()
            elif choice == "4":
                system.get_skill_focused_recommendations()
            elif choice == "5":
                username = input("Enter username: ").strip()
                if username:
                    allocation = system.allocate_single_student(username)
                    if allocation:
                        model_info = system.ml_allocation_engine.get_model_info()
                        if model_info['is_trained']:
                            print("\n✨ Used ML-enhanced allocation for optimal matching!")
            elif choice == "6":
                system.view_student_allocation()
            elif choice == "7":
                system.get_system_insights()
            elif choice == "8":
                system.view_training_vs_student_summary()
            elif choice == "9":
                system.view_ml_model_info()
            elif choice == "10":
                system.retrain_ml_model()
            elif choice == "11":
                print("Thank you for using the Enhanced ML Internship Allocation System!")
                break
            else:
                print("Invalid choice. Try again.")
                
    except Exception as e:
        print(f"System error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()