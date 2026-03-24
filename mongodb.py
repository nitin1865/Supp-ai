from pymongo import MongoClient

MONGO_URI = "mongodb+srv://suppai_db_user:54312589Raj%40@supp-ai-database.s9pdtym.mongodb.net/?appName=Supp-ai-database"

# Create MongoDB client
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

# Test connection
try:
    client.admin.command('ping')
    print("✅ Connected to MongoDB successfully!")
except Exception as e:
    print("❌ MongoDB connection error:", e)

# Database
db = client["meal_planner"]

# Collections
users_collection = db["users"]
meal_collection = db["meals"]