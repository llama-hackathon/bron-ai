import sqlite3
import json
import os
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class Job:
    id: Optional[int]
    video_path: str
    length: float
    frame_count: int
    process_frame_count: int
    framerate: float


@dataclass
class Frame:
    job_id: int
    frame_number: int
    description: str
    video_timestamp: float  # Time in seconds within the video
    structured_data: Optional[Dict[str, Any]] = None


class PreprocDB:
    def __init__(self, db_path: str = "data/preproc.db"):
        self.db_path = db_path
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize the database and create tables if they don't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create job table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS job (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_path TEXT NOT NULL UNIQUE,
                    length REAL NOT NULL,
                    frame_count INTEGER NOT NULL,
                    process_frame_count INTEGER NOT NULL,
                    framerate REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create frame_table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS frame_table (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER NOT NULL,
                    frame_number INTEGER NOT NULL,
                    description TEXT NOT NULL,
                    video_timestamp REAL NOT NULL,  -- Time in seconds within the video
                    structured_data TEXT,  -- JSON stored as TEXT
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES job (id) ON DELETE CASCADE,
                    UNIQUE(job_id, frame_number)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_frame_job_id ON frame_table(job_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_frame_number ON frame_table(frame_number)')
            
            conn.commit()
    
    def create_job(self, video_path: str, length: float, frame_count: int, process_frame_count: int, framerate: float) -> int:
        """Create a new job and return its ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO job (video_path, length, frame_count, process_frame_count, framerate)
                VALUES (?, ?, ?, ?, ?)
            ''', (video_path, length, frame_count, process_frame_count, framerate))
            conn.commit()
            return cursor.lastrowid
    
    def get_job(self, job_id: int) -> Optional[Job]:
        """Get a job by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM job WHERE id = ?', (job_id,))
            row = cursor.fetchone()
            if row:
                return Job(
                    id=row['id'],
                    video_path=row['video_path'],
                    length=row['length'],
                    frame_count=row['frame_count'],
                    process_frame_count=row['process_frame_count'],
                    framerate=row['framerate']
                )
            return None
    
    def get_job_by_video_path(self, video_path: str) -> Optional[Job]:
        """Get a job by video path"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM job WHERE video_path = ?', (video_path,))
            row = cursor.fetchone()
            if row:
                return Job(
                    id=row['id'],
                    video_path=row['video_path'],
                    length=row['length'],
                    frame_count=row['frame_count'],
                    process_frame_count=row['process_frame_count'],
                    framerate=row['framerate']
                )
            return None
    
    def update_job_process_count(self, job_id: int, process_frame_count: int):
        """Update the processed frame count for a job"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE job 
                SET process_frame_count = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (process_frame_count, job_id))
            conn.commit()
    
    def list_jobs(self) -> List[Job]:
        """Get all jobs"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM job ORDER BY created_at DESC')
            rows = cursor.fetchall()
            return [
                Job(
                    id=row['id'],
                    video_path=row['video_path'],
                    length=row['length'],
                    frame_count=row['frame_count'],
                    process_frame_count=row['process_frame_count'],
                    framerate=row['framerate']
                )
                for row in rows
            ]
    
    def add_frame(self, job_id: int, frame_number: int, description: str, video_timestamp: float, structured_data: Optional[Dict[str, Any]] = None):
        """Add a frame analysis result"""
        structured_data_json = json.dumps(structured_data) if structured_data else None
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO frame_table (job_id, frame_number, description, video_timestamp, structured_data)
                VALUES (?, ?, ?, ?, ?)
            ''', (job_id, frame_number, description, video_timestamp, structured_data_json))
            conn.commit()
    
    def get_frame(self, job_id: int, frame_number: int) -> Optional[Frame]:
        """Get a specific frame by job_id and frame_number"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM frame_table WHERE job_id = ? AND frame_number = ?
            ''', (job_id, frame_number))
            row = cursor.fetchone()
            if row:
                structured_data = json.loads(row['structured_data']) if row['structured_data'] else None
                return Frame(
                    job_id=row['job_id'],
                    frame_number=row['frame_number'],
                    description=row['description'],
                    video_timestamp=row['video_timestamp'],
                    structured_data=structured_data
                )
            return None
    
    def get_frames_for_job(self, job_id: int) -> List[Frame]:
        """Get all frames for a specific job"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM frame_table WHERE job_id = ? ORDER BY frame_number
            ''', (job_id,))
            rows = cursor.fetchall()
            frames = []
            for row in rows:
                structured_data = json.loads(row['structured_data']) if row['structured_data'] else None
                frames.append(Frame(
                    job_id=row['job_id'],
                    frame_number=row['frame_number'],
                    description=row['description'],
                    video_timestamp=row['video_timestamp'],
                    structured_data=structured_data
                ))
            return frames
    
    def delete_job(self, job_id: int):
        """Delete a job and all its frames"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM job WHERE id = ?', (job_id,))
            conn.commit()
    

# Example usage
if __name__ == "__main__":
    # Initialize database
    db = PreprocDB()
    
    # # Create a job
    # job_id = db.create_job(
    #     video_path="data/test_video.mp4",
    #     length=600.0,  # 10 minutes
    #     frame_count=18000,  # at 30fps
    #     process_frame_count=10,  # processing 1 frame per minute
    #     framerate=30.0
    # )
    
    # # Add some frame analysis results
    # db.add_frame(
    #     job_id=job_id,
    #     frame_number=0,
    #     description="Opening scene with landscape view",
    #     structured_data={"objects": ["tree", "sky", "mountain"], "mood": "serene"}
    # )
    
    # db.add_frame(
    #     job_id=job_id,
    #     frame_number=1,
    #     description="Person walking in the scene",
    #     structured_data={"objects": ["person", "path"], "action": "walking"}
    # )
    
    # # Query the data
    # job = db.get_job(job_id)
    # print(f"Job: {job}")
    
    # frames = db.get_frames_for_job(job_id)
    # for frame in frames:
    #     print(f"Frame {frame.frame_number}: {frame.description}")
    #     if frame.structured_data:
    #         print(f"  Data: {frame.structured_data}")
    
    # # Check progress
    # processed, total = db.get_job_progress(job_id)
    # print(f"Progress: {processed}/{total} frames processed") 