import sqlite3

def add_source_column_to_candidates():
    """Add source column to candidates table if it doesn't exist"""
    conn = sqlite3.connect('internship_allocation.db')  # Update with your actual DB path
    cursor = conn.cursor()
    
    # Check if 'source' column exists in candidates table
    cursor.execute("PRAGMA table_info(candidates)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'source' not in columns:
        cursor.execute("ALTER TABLE candidates ADD COLUMN source TEXT DEFAULT 'student_resume'")
        print("Added 'source' column to candidates table.")
    else:
        print("'source' column already exists in candidates table.")
    
    conn.commit()
    conn.close()

def add_source_column_to_allocations():
    """Add source column to allocations table if it doesn't exist"""
    conn = sqlite3.connect('internship_allocation.db')  # Update with your actual DB path
    cursor = conn.cursor()
    
    # Check if 'source' column exists in allocations table
    cursor.execute("PRAGMA table_info(allocations)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'source' not in columns:
        cursor.execute("ALTER TABLE allocations ADD COLUMN source TEXT DEFAULT 'student_resume'")
        print("Added 'source' column to allocations table.")
    else:
        print("'source' column already exists in allocations table.")
    
    conn.commit()
    conn.close()

# Call both functions to ensure schema is updated
def update_database_schema():
    """Update database schema with missing columns"""
    print("Updating database schema...")
    add_source_column_to_candidates()
    add_source_column_to_allocations()
    print("Database schema update complete!")

# Run this to fix your database
if __name__ == "__main__":
    update_database_schema()