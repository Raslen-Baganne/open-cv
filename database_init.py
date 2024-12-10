import sqlite3

def init_database():
    try:
        conn = sqlite3.connect("FaceBase.db")
        
        # Create Peoples table with age and gender
        conn.execute('''CREATE TABLE IF NOT EXISTS Peoples
            (id INTEGER PRIMARY KEY,
            Name TEXT NOT NULL,
            Age INTEGER,
            Gender TEXT)''')
            
        # Create Detections table for logging detections
        conn.execute('''CREATE TABLE IF NOT EXISTS Detections
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            time DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(person_id) REFERENCES Peoples(id))''')
            
        # Create DetectionHistory table for logging detected faces
        conn.execute('''CREATE TABLE IF NOT EXISTS DetectionHistory
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            person_name TEXT,
            detection_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(person_id) REFERENCES Peoples(id))''')
            
        # Create Workers table
        conn.execute('''CREATE TABLE IF NOT EXISTS Workers
            (id INTEGER PRIMARY KEY,
            Name TEXT NOT NULL)''')
            
        conn.commit()
        print("Database initialized successfully!")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

def add_worker_to_database(name, age=None, gender=None):
    try:
        conn = sqlite3.connect("FaceBase.db")
        cursor = conn.cursor()
        
        # Get next available ID
        cursor.execute("SELECT MAX(id) FROM Peoples")
        max_id = cursor.fetchone()[0]
        new_id = 1 if max_id is None else max_id + 1
        
        # Insert new worker
        cursor.execute("INSERT INTO Peoples (id, Name, Age, Gender) VALUES (?, ?, ?, ?)",
                      (new_id, name, age, gender))
        conn.commit()
        return new_id
    except sqlite3.Error as e:
        print(f"Error adding worker to database: {e}")
        return None
    finally:
        conn.close()

if __name__ == "__main__":
    init_database()
