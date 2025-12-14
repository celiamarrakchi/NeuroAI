import json
from datetime import datetime
import os

class Database:
    def __init__(self, db_path='health_data.json'):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize the JSON database file if it doesn't exist"""
        if not os.path.exists(self.db_path):
            with open(self.db_path, 'w') as f:
                json.dump({'users': [], 'last_id': 0}, f, indent=2)
    
    def _read_data(self):
        """Read data from JSON file"""
        try:
            with open(self.db_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {'users': [], 'last_id': 0}
    
    def _write_data(self, data):
        """Write data to JSON file"""
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_user_data(self, data):
        """Save user data to JSON file"""
        db_data = self._read_data()
        
        # Increment user ID
        db_data['last_id'] += 1
        user_id = db_data['last_id']
        
        # Create user record with new fields
        user_record = {
            'id': user_id,
            'age': data.get('age', 0),
            'name': data.get('name', 'unknown'),
            'sex': data.get('sex', 'unknown'),
            'Headache': data.get('Headache', 'unknown'),
            'Nausea': data.get('Nausea', 'unknown'),
            'Vomiting': data.get('Vomiting', 'unknown'),
            'Seizures': data.get('Seizures', 'unknown'),
            'Vision_Problems': data.get('Vision_Problems', 'unknown'),
            'Balance_Issues': data.get('Balance_Issues', 'unknown'),
            'Memory_Problems': data.get('Memory_Problems', 'unknown'),
            'Speech_Difficulties': data.get('Speech_Difficulties', 'unknown'),
            'Weakness': data.get('Weakness', 'unknown'),
            'created_at': datetime.now().isoformat()
        }
        
        db_data['users'].append(user_record)
        self._write_data(db_data)
        
        return user_id
    
    def get_all_users(self):
        """Get all users from JSON file"""
        db_data = self._read_data()
        return db_data['users']
    
    def get_user_by_id(self, user_id):
        """Get specific user by ID"""
        db_data = self._read_data()
        for user in db_data['users']:
            if user['id'] == user_id:
                return user
        return None
    
    def delete_user(self, user_id):
        """Delete a user from JSON file"""
        db_data = self._read_data()
        db_data['users'] = [u for u in db_data['users'] if u['id'] != user_id]
        self._write_data(db_data)
    
    def get_statistics(self):
        """Get basic statistics from the database"""
        db_data = self._read_data()
        users = db_data['users']
        
        if not users:
            return {
                'total_users': 0,
                'average_age': 0,
                'symptom_counts': {}
            }
        
        total_users = len(users)
        total_age = sum(u.get('age', 0) for u in users)
        avg_age = total_age / total_users if total_users > 0 else 0
        
        # Count symptoms
        symptoms = ['Headache', 'Nausea', 'Vomiting', 'Seizures', 
                   'Vision_Problems', 'Balance_Issues', 'Memory_Problems', 
                   'Speech_Difficulties', 'Weakness']
        
        symptom_counts = {}
        for symptom in symptoms:
            count = sum(1 for u in users if u.get(symptom, '').lower() == 'yes')
            symptom_counts[symptom] = count
        
        return {
            'total_users': total_users,
            'average_age': round(avg_age, 2),
            'symptom_counts': symptom_counts
        }