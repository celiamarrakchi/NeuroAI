from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from pydantic import BaseModel, Field
from typing import Optional
import json

class ExtractedData(BaseModel):
    name: Optional[str] = Field(default=None, description="patient's full name")
    age: Optional[int] = Field(default=None, description="person's age in years")
    sex: Optional[str] = Field(default=None, description="male, female, or other")
    Headache: Optional[str] = Field(default=None, description="yes or no - does the person have headaches")
    Nausea: Optional[str] = Field(default=None, description="yes or no - does the person have nausea")
    Vomiting: Optional[str] = Field(default=None, description="yes or no - does the person have vomiting")
    Seizures: Optional[str] = Field(default=None, description="yes or no - does the person have seizures")
    Vision_Problems: Optional[str] = Field(default=None, description="yes or no - does the person have vision problems")
    Balance_Issues: Optional[str] = Field(default=None, description="yes or no - does the person have balance issues")
    Memory_Problems: Optional[str] = Field(default=None, description="yes or no - does the person have memory problems")
    Speech_Difficulties: Optional[str] = Field(default=None, description="yes or no - does the person have speech difficulties")
    Weakness: Optional[str] = Field(default=None, description="yes or no - does the person have weakness")

class HealthDataAgent:
    def __init__(self, api_key=""):
        self.llm = ChatGroq(
            temperature=0.7,
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile"
        )
        
        self.extraction_llm = ChatGroq(
            temperature=0,
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile"
        )
        
        self.chat_history = InMemoryChatMessageHistory()
        
        # Main conversation prompt - CRITICAL: Must end when complete
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly healthcare assistant collecting neurological health data.

You need to collect these 12 pieces of information:
1. Name, 2. Age, 3. Sex, 4. Headache, 5. Nausea, 6. Vomiting, 7. Seizures, 
8. Vision_Problems, 9. Balance_Issues, 10. Memory_Problems, 11. Speech_Difficulties, 12. Weakness

CRITICAL RULES:
- If ALL 12 fields are collected, YOU MUST say: "Thank you! I have collected all the information. Your data is being saved."
- DO NOT ask for more information if all fields are already collected
- Ask 2-3 questions at a time to be efficient
- Be conversational but focused on collecting the data

Already collected: {collected_data}
Still missing: {missing_fields}

If missing_fields is empty, END THE CONVERSATION immediately with a thank you message."""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Extraction prompt - MORE AGGRESSIVE
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract health data from the user's message. Be AGGRESSIVE in extraction - interpret ANY relevant information.

RULES:
- Any mention of "my name is <text>" or "I'm <text>" at the start can provide the name → name: exact text (no quotes)
- "I'm 25" or "25 years old" or "I am 25" → age: 25
- "male", "female", "man", "woman", "guy", "girl" → sex: male/female
- ANY mention of headaches, head pain → Headache: yes
- "no headaches" → Headache: no
- ANY mention of feeling sick, nauseous → Nausea: yes
- ANY mention of throwing up, vomit → Vomiting: yes
- ANY mention of seizures, fits, convulsions → Seizures: yes
- ANY mention of vision, seeing problems, blurry → Vision_Problems: yes
- ANY mention of balance, dizzy, coordination → Balance_Issues: yes
- ANY mention of memory, forgetting, can't remember → Memory_Problems: yes
- ANY mention of speech, talking difficulty → Speech_Difficulties: yes
- ANY mention of weakness, tired, fatigue, low energy → Weakness: yes

If someone says "no" to a symptom, mark it as "no"
If someone says "yes" or describes having it, mark it as "yes"
If not mentioned at all, use null

Return ONLY JSON with these exact field names:
{{"name": "string" or null, "age": int or null, "sex": "male"/"female"/"other" or null, "Headache": "yes"/"no" or null, 
"Nausea": "yes"/"no" or null, "Vomiting": "yes"/"no" or null, "Seizures": "yes"/"no" or null,
"Vision_Problems": "yes"/"no" or null, "Balance_Issues": "yes"/"no" or null, 
"Memory_Problems": "yes"/"no" or null, "Speech_Difficulties": "yes"/"no" or null, 
"Weakness": "yes"/"no" or null}}"""),
            ("human", "Extract from: {message}")
        ])
        
        self.chain = self.prompt | self.llm
        self.extraction_chain = self.extraction_prompt | self.extraction_llm
    
    def start_conversation(self):
        """Start the conversation"""
        return "Hello! I'm here to collect some neurological health information. Let's start: What is your full name, age, and sex?"
    
    def extract_data_with_ai(self, user_message, collected_data):
        """Use AI to extract data from user message"""
        try:
            response = self.extraction_chain.invoke({"message": user_message})
            content = response.content.strip()
            
            # Clean JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            extracted_raw = json.loads(content)
            extracted = {}
            
            # Extract name
            if 'name' not in collected_data and extracted_raw.get('name'):
                name_val = str(extracted_raw['name']).strip()
                if 1 <= len(name_val) <= 120:
                    extracted['name'] = name_val
            
            # Extract age
            if 'age' not in collected_data and extracted_raw.get('age'):
                age = int(extracted_raw['age'])
                if 1 <= age <= 120:
                    extracted['age'] = age
            
            # Extract sex
            if 'sex' not in collected_data and extracted_raw.get('sex'):
                sex = extracted_raw['sex'].lower()
                if sex in ['male', 'female', 'other', 'm', 'f']:
                    extracted['sex'] = 'male' if sex == 'm' else ('female' if sex == 'f' else sex)
            
            # Extract symptoms
            symptoms = ['Headache', 'Nausea', 'Vomiting', 'Seizures', 
                       'Vision_Problems', 'Balance_Issues', 'Memory_Problems', 
                       'Speech_Difficulties', 'Weakness']
            
            for symptom in symptoms:
                if symptom not in collected_data and extracted_raw.get(symptom):
                    value = str(extracted_raw[symptom]).lower()
                    if value in ['yes', 'no']:
                        extracted[symptom] = value
            
            print(f"DEBUG: Extracted data: {extracted}")
            return extracted
            
        except Exception as e:
            print(f"Extraction error: {e}")
            return {}
    
    def get_missing_fields(self, collected_data):
        """Get list of missing fields"""
        all_fields = ['name', 'age', 'sex', 'Headache', 'Nausea', 'Vomiting', 'Seizures',
                     'Vision_Problems', 'Balance_Issues', 'Memory_Problems', 
                     'Speech_Difficulties', 'Weakness']
        
        missing = [f for f in all_fields if f not in collected_data]
        print(f"DEBUG: Missing fields: {missing}")
        print(f"DEBUG: Collected so far: {collected_data}")
        return missing
    
    def process_message(self, user_message, collected_data):
        """Process user message and return response"""
        # Extract data
        extracted = self.extract_data_with_ai(user_message, collected_data)
        
        # Update collected data
        collected_data.update(extracted)
        
        # Check completion
        missing_fields = self.get_missing_fields(collected_data)
        is_complete = len(missing_fields) == 0
        
        print(f"DEBUG: Is complete? {is_complete}")
        
        if is_complete:
            response = "Perfect! I have collected all 12 pieces of information. Your health data is now being saved to the database. Thank you for your time!"
            return response, extracted, True
        
        # Generate response
        try:
            self.chat_history.add_user_message(user_message)
            
            response = self.chain.invoke({
                "input": user_message,
                "collected_data": str(collected_data),
                "missing_fields": ", ".join(missing_fields),
                "history": self.chat_history.messages
            })
            
            self.chat_history.add_ai_message(response.content)
            
            return response.content, extracted, False
        except Exception as e:
            response = f"I'm having trouble. Still need: {', '.join(missing_fields)}"
            print(f"Error: {e}")
            return response, extracted, False