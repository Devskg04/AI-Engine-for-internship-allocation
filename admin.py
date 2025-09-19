import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
from typing import Dict, List, Tuple, Optional
import logging
import shutil
from pathlib import Path
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompanyManagementSystem:
    """Admin system for managing companies and student resume allocation"""
    
    def __init__(self, db_path: str = "admin_database.db"):
        print("Initializing Company Management System...")
        
        self.db_path = db_path
        self.companies_csv = "list_of_companies.csv"
        self.admin_allocations_csv = "admin_allocations.csv"
        self.admin_data_dir = "admin_data"
        
        # Reference to student system files
        self.student_allocation_csv = "allocations.csv"
        self.student_data_csv = "student_data.csv"
        
        self.setup_admin_directories()
        self.init_admin_database()
        self.init_admin_csv_files()
        print("Company Management System initialized successfully!")
    
    def setup_admin_directories(self):
        """Create necessary admin directories"""
        # Create main admin data directory
        Path(self.admin_data_dir).mkdir(parents=True, exist_ok=True)
        print(f"Created/verified admin data directory: {self.admin_data_dir}")
    
    def init_admin_database(self):
        """Initialize admin database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Companies table with all required fields
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS companies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                rating REAL,
                reviews INTEGER,
                company_type TEXT,
                headquarter TEXT,
                old_name TEXT,
                no_of_employees TEXT,
                tags TEXT,
                current_capacity INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            )
        ''')
        
        # Admin allocations tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS admin_allocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT,
                username TEXT,
                company_name TEXT,
                company_id INTEGER,
                resume_path TEXT,
                admin_resume_path TEXT,
                allocated_date TEXT,
                status TEXT DEFAULT 'Active',
                FOREIGN KEY (company_id) REFERENCES companies(id)
            )
        ''')
        
        # Company resume storage tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS company_resumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_name TEXT,
                candidate_id TEXT,
                username TEXT,
                resume_filename TEXT,
                storage_path TEXT,
                stored_date TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print("Admin database initialized successfully!")
    
    def init_admin_csv_files(self):
        """Initialize admin CSV files"""
        # Companies CSV with specified columns
        if not os.path.exists(self.companies_csv):
            companies_df = pd.DataFrame(columns=[
                'Unnamed: 0', 'Name', 'Rating', 'Reviews', 'Company_type', 
                'Headquarter', 'Old', 'No. of Employees', 'Tags', 'current_capacity'
            ])
            companies_df.to_csv(self.companies_csv, index=False)
            print(f"Created {self.companies_csv}")
        
        # Admin allocations CSV
        if not os.path.exists(self.admin_allocations_csv):
            admin_allocations_df = pd.DataFrame(columns=[
                'candidate_id', 'username', 'company_name', 'resume_original_path',
                'resume_admin_path', 'allocated_date', 'status'
            ])
            admin_allocations_df.to_csv(self.admin_allocations_csv, index=False)
            print(f"Created {self.admin_allocations_csv}")
    
    def add_company(self):
        """Add a new company to the system"""
        print("\n" + "="*60)
        print("ADD NEW COMPANY")
        print("="*60)
        
        # Get company information
        name = input("Company Name: ").strip()
        if not name:
            print("Company name is required!")
            return
        
        # Check if company already exists
        if self.check_company_exists(name):
            print(f"Company '{name}' already exists!")
            return
        
        try:
            rating = float(input("Rating (0.0-5.0): ") or "0.0")
            reviews = int(input("Number of Reviews: ") or "0")
        except ValueError:
            rating = 0.0
            reviews = 0
        
        company_type = input("Company Type (e.g., Technology, Finance): ").strip()
        headquarter = input("Headquarter Location: ").strip()
        old_name = input("Old/Previous Name (if any): ").strip()
        no_of_employees = input("Number of Employees (e.g., 1000-5000): ").strip()
        tags = input("Tags (comma-separated): ").strip()
        
        try:
            current_capacity = int(input("Current Internship Capacity: ") or "5")
        except ValueError:
            current_capacity = 5
        
        # Save to database
        company_id = self.save_company_to_db({
            'name': name,
            'rating': rating,
            'reviews': reviews,
            'company_type': company_type,
            'headquarter': headquarter,
            'old_name': old_name,
            'no_of_employees': no_of_employees,
            'tags': tags,
            'current_capacity': current_capacity
        })
        
        # Save to CSV
        self.save_company_to_csv({
            'Name': name,
            'Rating': rating,
            'Reviews': reviews,
            'Company_type': company_type,
            'Headquarter': headquarter,
            'Old': old_name,
            'No. of Employees': no_of_employees,
            'Tags': tags,
            'current_capacity': current_capacity
        })
        
        # Create company directory in admin_data
        company_dir = Path(self.admin_data_dir) / name
        company_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCompany '{name}' added successfully!")
        print(f"Company ID: {company_id}")
        print(f"Resume storage directory created: {company_dir}")
    
    def check_company_exists(self, company_name: str) -> bool:
        """Check if company exists in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM companies WHERE name = ?", (company_name,))
        exists = cursor.fetchone()[0] > 0
        conn.close()
        return exists
    
    def save_company_to_db(self, company_data: Dict) -> int:
        """Save company to database and return company ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO companies 
            (name, rating, reviews, company_type, headquarter, old_name, 
             no_of_employees, tags, current_capacity, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            company_data['name'],
            company_data['rating'],
            company_data['reviews'],
            company_data['company_type'],
            company_data['headquarter'],
            company_data['old_name'],
            company_data['no_of_employees'],
            company_data['tags'],
            company_data['current_capacity'],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        
        company_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return company_id
    
    def save_company_to_csv(self, company_data: Dict):
        """Save company to CSV file"""
        try:
            df = pd.read_csv(self.companies_csv)
            
            # Get next index
            if df.empty:
                next_index = 0
            else:
                next_index = df['Unnamed: 0'].max() + 1 if 'Unnamed: 0' in df.columns else len(df)
            
            company_data['Unnamed: 0'] = next_index
            
            # Add new row
            df = pd.concat([df, pd.DataFrame([company_data])], ignore_index=True)
            df.to_csv(self.companies_csv, index=False)
            print(f"Company saved to {self.companies_csv}")
            
        except Exception as e:
            print(f"Error saving company to CSV: {e}")
    
    def view_all_companies(self):
        """View all companies in the system"""
        try:
            df = pd.read_csv(self.companies_csv)
            if df.empty:
                print("No companies found in the system.")
                return
            
            print("\n" + "="*100)
            print("ALL COMPANIES IN THE SYSTEM")
            print("="*100)
            
            for _, row in df.iterrows():
                print(f"ID: {row.get('Unnamed: 0', 'N/A')} | Name: {row['Name']}")
                print(f"Rating: {row.get('Rating', 'N/A')}/5.0 | Reviews: {row.get('Reviews', 'N/A')}")
                print(f"Type: {row.get('Company_type', 'N/A')} | HQ: {row.get('Headquarter', 'N/A')}")
                print(f"Employees: {row.get('No. of Employees', 'N/A')} | Capacity: {row.get('current_capacity', 'N/A')}")
                if pd.notna(row.get('Tags')):
                    print(f"Tags: {row['Tags']}")
                print("-" * 80)
            
            print(f"Total companies: {len(df)}")
            
        except Exception as e:
            print(f"Error viewing companies: {e}")
    
    def monitor_student_allocations(self):
        """Monitor student allocations and copy resumes to company folders"""
        print("\nMonitoring student allocations for resume management...")
        
        try:
            # Read student allocation data
            if not os.path.exists(self.student_allocation_csv):
                print("No student allocation file found.")
                return
            
            allocation_df = pd.read_csv(self.student_allocation_csv)
            
            if allocation_df.empty:
                print("No student allocations found.")
                return
            
            # Read student data for resume paths
            if not os.path.exists(self.student_data_csv):
                print("No student data file found.")
                return
            
            student_df = pd.read_csv(self.student_data_csv)
            
            processed_count = 0
            
            for _, allocation in allocation_df.iterrows():
                candidate_id = allocation['candidate_id']
                username = allocation['username']
                company_name = allocation['company_name']
                
                # Check if already processed
                if self.check_resume_already_stored(candidate_id, company_name):
                    continue
                
                # Find student's resume path
                student_data = student_df[student_df['candidate_id'] == candidate_id]
                if student_data.empty:
                    continue
                
                student_row = student_data.iloc[0]
                original_resume_path = student_row.get('resume_path')
                
                if pd.isna(original_resume_path) or not os.path.exists(original_resume_path):
                    print(f"Resume not found for {username}")
                    continue
                
                # Copy resume to company folder
                success = self.copy_resume_to_company_folder(
                    candidate_id, username, company_name, original_resume_path
                )
                
                if success:
                    processed_count += 1
                    print(f"✓ Processed {username} -> {company_name}")
            
            print(f"\nProcessed {processed_count} new allocations")
            
        except Exception as e:
            print(f"Error monitoring allocations: {e}")
    
    def check_resume_already_stored(self, candidate_id: str, company_name: str) -> bool:
        """Check if resume is already stored for this candidate-company pair"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM company_resumes WHERE candidate_id = ? AND company_name = ?",
            (candidate_id, company_name)
        )
        exists = cursor.fetchone()[0] > 0
        conn.close()
        return exists
    
    def copy_resume_to_company_folder(self, candidate_id: str, username: str, 
                                    company_name: str, original_resume_path: str) -> bool:
        """Copy student resume to company folder in admin_data"""
        try:
            # Create company directory if it doesn't exist
            company_dir = Path(self.admin_data_dir) / company_name
            company_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate new filename
            resume_filename = f"{username}_{candidate_id}_resume.pdf"
            admin_resume_path = company_dir / resume_filename
            
            # Copy the resume file
            shutil.copy2(original_resume_path, admin_resume_path)
            
            # Record in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO company_resumes 
                (company_name, candidate_id, username, resume_filename, storage_path, stored_date)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                company_name,
                candidate_id,
                username,
                resume_filename,
                str(admin_resume_path),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            
            conn.commit()
            conn.close()
            
            # Record in admin allocations CSV
            self.save_admin_allocation_to_csv({
                'candidate_id': candidate_id,
                'username': username,
                'company_name': company_name,
                'resume_original_path': original_resume_path,
                'resume_admin_path': str(admin_resume_path),
                'allocated_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'Active'
            })
            
            return True
            
        except Exception as e:
            print(f"Error copying resume for {username} to {company_name}: {e}")
            return False
    
    def save_admin_allocation_to_csv(self, allocation_data: Dict):
        """Save admin allocation record to CSV"""
        try:
            df = pd.read_csv(self.admin_allocations_csv)
            
            # Check if allocation already exists
            existing = df[
                (df['candidate_id'] == allocation_data['candidate_id']) &
                (df['company_name'] == allocation_data['company_name'])
            ]
            
            if existing.empty:
                # Add new row
                df = pd.concat([df, pd.DataFrame([allocation_data])], ignore_index=True)
                df.to_csv(self.admin_allocations_csv, index=False)
            
        except Exception as e:
            print(f"Error saving admin allocation to CSV: {e}")
    
    def view_company_resumes(self, company_name: str = None):
        """View resumes stored for a specific company or all companies"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if company_name:
                cursor.execute(
                    "SELECT * FROM company_resumes WHERE company_name = ? ORDER BY stored_date DESC",
                    (company_name,)
                )
                print(f"\nRESUMES FOR {company_name.upper()}")
            else:
                cursor.execute("SELECT * FROM company_resumes ORDER BY company_name, stored_date DESC")
                print("\nALL COMPANY RESUMES")
            
            print("="*80)
            
            rows = cursor.fetchall()
            if not rows:
                print("No resumes found.")
                conn.close()
                return
            
            current_company = ""
            for row in rows:
                if row[1] != current_company:  # company_name is at index 1
                    current_company = row[1]
                    print(f"\n--- {current_company} ---")
                
                print(f"Student: {row[3]} (ID: {row[2]})")  # username, candidate_id
                print(f"Resume: {row[4]}")  # resume_filename
                print(f"Stored: {row[6]}")  # stored_date
                print(f"Path: {row[5]}")  # storage_path
                print("-" * 40)
            
            conn.close()
            
        except Exception as e:
            print(f"Error viewing company resumes: {e}")
    
    def get_company_statistics(self):
        """Get comprehensive statistics about companies and allocations"""
        print("\n" + "="*60)
        print("COMPANY MANAGEMENT STATISTICS")
        print("="*60)
        
        try:
            # Company statistics
            companies_df = pd.read_csv(self.companies_csv)
            print(f"Total Companies: {len(companies_df)}")
            
            if not companies_df.empty:
                total_capacity = companies_df['current_capacity'].sum()
                avg_rating = companies_df['Rating'].mean()
                print(f"Total Internship Capacity: {total_capacity}")
                print(f"Average Company Rating: {avg_rating:.2f}/5.0")
                
                # Company types distribution
                if 'Company_type' in companies_df.columns:
                    type_dist = companies_df['Company_type'].value_counts()
                    print(f"\nCompany Types Distribution:")
                    for comp_type, count in type_dist.head(5).items():
                        print(f"  {comp_type}: {count}")
            
            # Resume storage statistics
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM company_resumes")
            total_resumes = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT company_name) FROM company_resumes")
            companies_with_resumes = cursor.fetchone()[0]
            
            print(f"\nResume Management:")
            print(f"Total Resumes Stored: {total_resumes}")
            print(f"Companies with Resumes: {companies_with_resumes}")
            
            # Top companies by resume count
            cursor.execute('''
                SELECT company_name, COUNT(*) as resume_count 
                FROM company_resumes 
                GROUP BY company_name 
                ORDER BY resume_count DESC 
                LIMIT 5
            ''')
            
            top_companies = cursor.fetchall()
            if top_companies:
                print(f"\nTop Companies by Resume Count:")
                for company, count in top_companies:
                    print(f"  {company}: {count} resumes")
            
            conn.close()
            
            # Directory size information
            total_size = 0
            for company_dir in Path(self.admin_data_dir).iterdir():
                if company_dir.is_dir():
                    dir_size = sum(f.stat().st_size for f in company_dir.glob('*') if f.is_file())
                    total_size += dir_size
                    print(f"  {company_dir.name}: {len(list(company_dir.glob('*.pdf')))} files")
            
            print(f"\nTotal Storage Used: {total_size / (1024*1024):.2f} MB")
            
        except Exception as e:
            print(f"Error generating statistics: {e}")
    
    def search_student_by_company(self, company_name: str):
        """Search all students allocated to a specific company"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM company_resumes WHERE company_name = ? ORDER BY stored_date DESC",
                (company_name,)
            )
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                print(f"No students found for company: {company_name}")
                return
            
            print(f"\n" + "="*60)
            print(f"STUDENTS ALLOCATED TO {company_name.upper()}")
            print("="*60)
            
            for row in rows:
                print(f"Student ID: {row[2]}")  # candidate_id
                print(f"Username: {row[3]}")   # username
                print(f"Resume File: {row[4]}")  # resume_filename
                print(f"Allocated Date: {row[6]}")  # stored_date
                print(f"Resume Path: {row[5]}")  # storage_path
                
                # Verify file still exists
                if os.path.exists(row[5]):
                    print("Status: ✓ Resume file exists")
                else:
                    print("Status: ✗ Resume file missing")
                
                print("-" * 40)
            
            print(f"Total students: {len(rows)}")
            
        except Exception as e:
            print(f"Error searching students: {e}")
    
    def update_company_capacity(self, company_name: str, new_capacity: int):
        """Update company's internship capacity"""
        try:
            # Update in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE companies SET current_capacity = ?, updated_at = ? WHERE name = ?",
                (new_capacity, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), company_name)
            )
            
            rows_affected = cursor.rowcount
            conn.commit()
            conn.close()
            
            if rows_affected == 0:
                print(f"Company '{company_name}' not found.")
                return
            
            # Update in CSV
            df = pd.read_csv(self.companies_csv)
            df.loc[df['Name'] == company_name, 'current_capacity'] = new_capacity
            df.to_csv(self.companies_csv, index=False)
            
            print(f"Updated capacity for {company_name} to {new_capacity}")
            
        except Exception as e:
            print(f"Error updating company capacity: {e}")
    
    def backup_company_data(self):
        """Create backup of all company data"""
        try:
            backup_dir = Path("admin_backup")
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_subdir = backup_dir / f"backup_{timestamp}"
            backup_subdir.mkdir(exist_ok=True)
            
            # Backup CSV files
            shutil.copy2(self.companies_csv, backup_subdir / self.companies_csv)
            shutil.copy2(self.admin_allocations_csv, backup_subdir / self.admin_allocations_csv)
            
            # Backup database
            shutil.copy2(self.db_path, backup_subdir / self.db_path)
            
            # Backup admin_data directory
            if os.path.exists(self.admin_data_dir):
                shutil.copytree(self.admin_data_dir, backup_subdir / self.admin_data_dir)
            
            print(f"Backup created successfully at: {backup_subdir}")
            
        except Exception as e:
            print(f"Error creating backup: {e}")

def main():
    """Main admin application"""
    print("="*60)
    print("COMPANY MANAGEMENT & ADMIN SYSTEM")
    print("Resume Storage & Company Information Management")
    print("="*60)
    
    try:
        admin_system = CompanyManagementSystem()
        
        while True:
            print("\nADMIN MAIN MENU:")
            print("1. Add New Company")
            print("2. View All Companies")
            print("3. Update Company Capacity")
            print("4. Monitor Student Allocations (Process Resume Storage)")
            print("5. View Company Resumes (All)")
            print("6. View Resumes for Specific Company")
            print("7. Search Students by Company")
            print("8. Get Company Statistics")
            print("9. Create Data Backup")
            print("10. Exit")
            
            try:
                choice = input("\nEnter your choice (1-10): ").strip()
                
                if choice == "1":
                    admin_system.add_company()
                    
                elif choice == "2":
                    admin_system.view_all_companies()
                    
                elif choice == "3":
                    company_name = input("Enter company name: ").strip()
                    if company_name:
                        try:
                            new_capacity = int(input("Enter new capacity: "))
                            admin_system.update_company_capacity(company_name, new_capacity)
                        except ValueError:
                            print("Invalid capacity number.")
                    
                elif choice == "4":
                    admin_system.monitor_student_allocations()
                    
                elif choice == "5":
                    admin_system.view_company_resumes()
                    
                elif choice == "6":
                    company_name = input("Enter company name: ").strip()
                    if company_name:
                        admin_system.view_company_resumes(company_name)
                    
                elif choice == "7":
                    company_name = input("Enter company name: ").strip()
                    if company_name:
                        admin_system.search_student_by_company(company_name)
                    
                elif choice == "8":
                    admin_system.get_company_statistics()
                    
                elif choice == "9":
                    admin_system.backup_company_data()
                    
                elif choice == "10":
                    print("\nThank you for using the Admin System!")
                    break
                    
                else:
                    print("Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error in menu operation: {e}")
                continue
                
    except Exception as e:
        print(f"Admin System initialization failed: {e}")

if __name__ == "__main__":
    main()