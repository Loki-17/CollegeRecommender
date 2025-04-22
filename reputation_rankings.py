# Define a list of top Indian colleges with their national rankings/reputation score
# Based on various Indian education ranking systems like NIRF
top_colleges = {
    # IITs (Indian Institutes of Technology) - Premier engineering institutes
    "IIT Madras": 100,
    "IIT Delhi": 98,
    "IIT Bombay": 97,
    "IIT Kanpur": 96,
    "IIT Kharagpur": 95,
    "IIT Roorkee": 94,
    "IIT Guwahati": 93,
    "IIT Hyderabad": 92,
    "IIT Jodhpur": 86,
    
    # NITs (National Institutes of Technology)
    "NIT Tiruchirappalli": 88,
    "NIT Surathkal": 86,
    "NIT Rourkela": 84,
    "NIT Warangal": 82,
    "NIT Calicut": 80,
    "NIT Durgapur": 75,
    "NIT Kurukshetra": 74,
    
    # IIITs (Indian Institutes of Information Technology) 
    "IIIT Hyderabad": 85,
    "IIIT Delhi": 83,
    "IIIT Bangalore": 82,
    
    # Medical colleges
    "AIIMS Delhi": 100,
    "AIIMS Jodhpur": 90,
    "Christian Medical College Vellore": 95,
    
    # Other prestigious institutions
    "BITS Pilani": 92,
    "Jadavpur University": 82,
    "Anna University": 80,
    "Delhi University": 85,
    "Jamia Millia Islamia": 75,
    "BMS College of Engineering": 70,
    
    # Medium-ranked institutions
    "Chandigarh University": 60,
    "Amity University": 55,
    "Sharda University": 48,
    
    # Lower-ranked but still good institutions
    "Galgotias University": 35,
    "Lovely Professional University": 30,
}

# Print the rankings for reference
for college, rank in sorted(top_colleges.items(), key=lambda x: x[1], reverse=True):
    print(f"{college}: {rank}")
