from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict
from enum import Enum
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
from datetime import datetime
import uuid
import pandas as pd
import os
import openai
import re
import logging
from dotenv import load_dotenv

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Env ──────────────────────────────────────────────────────────────────────
load_dotenv()
openai_api_key  = os.getenv("OPENAI_API_KEY")
MONGODB_URL     = os.getenv("MONGODB_URL")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "meal_planner")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not MONGODB_URL:
    raise ValueError("MONGODB_URL not found in environment variables")

client_openai = openai.OpenAI(api_key=openai_api_key)
MODEL = "gpt-4o-mini"

# ─── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Personalized Meal Plan Generator API",
    description="Generate customized meal plans based on user health goals and preferences",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MongoDB globals ───────────────────────────────────────────────────────────
mongo_client: AsyncIOMotorClient = None
db = None

# ─── Enums ────────────────────────────────────────────────────────────────────
class DietType(str, Enum):
    OMNIVORE    = "Omnivore"
    VEGETARIAN  = "Vegetarian"
    VEGAN       = "Vegan"
    PESCATARIAN = "Pescatarian"
    LOW_CARB    = "Low-carb/Keto"
    GLUTEN_FREE = "Gluten-free"
    JAIN        = "Jain/Sattvic"
    OTHER       = "Other"

class ActivityLevel(str, Enum):
    LITTLE   = "Little movement"
    WALKING  = "Walking, stretching everyday"
    MODERATE = "Exercise 2-3 times a week"
    DAILY    = "Daily training"

class Gender(str, Enum):
    MALE   = "Male"
    FEMALE = "Female"
    OTHER  = "Other"

# ─── Pydantic models ──────────────────────────────────────────────────────────

# STEP 1 — Registration (email used only here)
class RegisterRequest(BaseModel):
    """
    Only endpoint that accepts email.
    Returns a user_id the frontend must store and send with every future request.
    """
    email: EmailStr
    name:  str = Field(..., min_length=1, max_length=100)

class RegisterResponse(BaseModel):
    message: str
    user_id: str   # frontend stores this — sent with every subsequent request
    name:    str

# STEP 2+ — All subsequent requests use user_id, never email
class UserProfile(BaseModel):
    """
    Health profile submitted after registration.
    Uses user_id (returned by /register) — no email field.
    """
    user_id:                str       = Field(..., min_length=1)
    name:                   str       = Field(..., min_length=1, max_length=100)
    age:                    int       = Field(..., ge=5, le=120)
    gender:                 Gender
    height:                 float     = Field(..., ge=100, le=250)
    weight:                 float     = Field(..., ge=20, le=300)
    diet:                   DietType
    activity_level:         ActivityLevel
    food_allergies:         List[str] = Field(default=[])
    health_goals:           List[str] = Field(default=[])
    disease:                List[str] = Field(default=[])
    supplement_preferences: List[str] = Field(default=[])
    food_type:              Optional[str] = None

class MealPlanRequest(BaseModel):
    user_profile: UserProfile

class MealModificationRequest(BaseModel):
    meal_plan:    str
    modification: str = Field(..., min_length=1)

class MealAdjustmentRequest(BaseModel):
    meal_plan:      str
    food_name:      str = Field(..., min_length=1)
    quantity_grams: int = Field(..., ge=1, le=1000)
    meal_replaced:  str

class CalorieInfo(BaseModel):
    found:             bool
    calories:          Optional[int]   = None
    calories_per_100g: Optional[float] = None
    food_name:         str
    quantity:          int
    fat:               Optional[float] = None
    carbs:             Optional[float] = None
    protein:           Optional[float] = None

class MealPlanResponse(BaseModel):
    meal_plan:     str
    bmi:           float
    bmr:           float
    calorie_range: str
    daily_goal:    str

# ─── Startup / Shutdown ───────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global mongo_client, db, meals_df, nutrition_df

    # Connect
    try:
        mongo_client = AsyncIOMotorClient(MONGODB_URL)
        db = mongo_client[MONGODB_DB_NAME]
        await db.command("ping")
        logger.info("Connected to MongoDB successfully")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        raise

    # ── Indexes ───────────────────────────────────────────────────────────────

    # user_profiles.email — unique, prevents duplicate registrations
    await db.user_profiles.create_index(
        [("email", ASCENDING)],
        unique=True,
        name="unique_email_idx",
    )

    # user_profiles.user_id — unique, used for every lookup after registration
    await db.user_profiles.create_index(
        [("user_id", ASCENDING)],
        unique=True,
        name="unique_user_id_idx",
    )

    # meal_plans.user_id — fast lookup of all plans for a user
    await db.meal_plans.create_index(
        [("user_id", ASCENDING)],
        name="meal_plans_user_id_idx",
    )

    # meal_plans compound — "get latest N plans for user_id" in one index scan
    await db.meal_plans.create_index(
        [("user_id", ASCENDING), ("created_at", DESCENDING)],
        name="meal_plans_user_id_created_at_idx",
    )

    logger.info("All MongoDB indexes are ready")

    # ── CSV data ──────────────────────────────────────────────────────────────
    try:
        if os.path.exists("meal.csv"):
            meals_df = pd.read_csv("meal.csv")
            if "Unnamed: 0" in meals_df.columns:
                meals_df = meals_df.drop(columns=["Unnamed: 0"])
            logger.info(f"Loaded {len(meals_df)} meals from meal.csv")
        else:
            logger.warning("meal.csv not found")
            meals_df = pd.DataFrame()

        if os.path.exists("nutrition_data.csv"):
            nutrition_df = pd.read_csv("nutrition_data.csv")
            logger.info(f"Loaded {len(nutrition_df)} items from nutrition_data.csv")
        else:
            logger.warning("nutrition_data.csv not found — AI fallback will be used")
            nutrition_df = pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    if mongo_client:
        mongo_client.close()
        logger.info("MongoDB connection closed")

# ─── MongoDB helpers ──────────────────────────────────────────────────────────

async def db_register_user(email: str, name: str) -> str:
    """
    Create a new user document.
    - Generates a UUID as user_id.
    - Stores email (lowercase) for duplicate checking only.
    - Raises 409 if email already registered.
    - Returns the new user_id string.
    """
    user_id = str(uuid.uuid4())
    try:
        await db.user_profiles.insert_one({
            "user_id":    user_id,
            "email":      email,           # lowercase, stored for uniqueness only
            "name":       name,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        })
        logger.info(f"Registered new user: {email} -> user_id={user_id}")
        return user_id
    except Exception as e:
        if "11000" in str(e) or "duplicate key" in str(e).lower():
            raise HTTPException(
                status_code=409,
                detail=f"Email '{email}' is already registered.",
            )
        raise


async def db_save_user_profile(user_profile: UserProfile) -> None:
    """
    Update an existing user's health profile, matched by user_id.
    Raises 404 if user_id does not exist.
    upsert=False — never silently create a new document.
    """
    doc = user_profile.model_dump()
    doc["updated_at"] = datetime.utcnow()

    result = await db.user_profiles.update_one(
        {"user_id": user_profile.user_id},
        {"$set": doc},
        upsert=False,
    )
    if result.matched_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"user_id '{user_profile.user_id}' not found. Register first via POST /register.",
        )


async def db_save_meal_plan(user_id: str, meal_plan_data: dict) -> str:
    """
    Insert a new meal plan record linked to user_id.
    History is preserved — one document per generation.
    Returns the inserted document _id as a string.
    """
    result = await db.meal_plans.insert_one({
        "user_id":       user_id,
        "meal_plan":     meal_plan_data.get("meal_plan"),
        "bmi":           meal_plan_data.get("bmi"),
        "bmr":           meal_plan_data.get("bmr"),
        "calorie_range": meal_plan_data.get("calorie_range"),
        "daily_goal":    meal_plan_data.get("daily_goal"),
        "created_at":    datetime.utcnow(),
    })
    return str(result.inserted_id)


async def db_get_user(user_id: str) -> dict:
    """
    Fetch user document by user_id.
    Raises 404 if not found.
    """
    user = await db.user_profiles.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(
            status_code=404,
            detail=f"user_id '{user_id}' not found. Register first via POST /register.",
        )
    return user

# ─── Business logic ───────────────────────────────────────────────────────────

def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    return round(weight_kg / (height_cm / 100) ** 2, 1)

def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> float:
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    return round(base + 5 if gender.lower().startswith("m") else base - 161, 1)

def get_calories_for_food(food_name: str, quantity_grams: int) -> Dict:
    try:
        if not nutrition_df.empty:
            matches = nutrition_df[
                nutrition_df["Food Item"].astype(str).str.contains(food_name, case=False, na=False)
            ]
            if not matches.empty:
                row = matches.iloc[0]
                c, f, b, p = (
                    float(row["Calories"]), float(row["Fat(g)"]),
                    float(row["Carbs(g)"]), float(row["Protein(g)"])
                )
                q = quantity_grams
                return {
                    "found":             True,
                    "calories":          int(c * q / 100),
                    "calories_per_100g": c,
                    "food_name":         row["Food Item"],
                    "quantity":          q,
                    "fat":               round(f * q / 100, 2),
                    "carbs":             round(b * q / 100, 2),
                    "protein":           round(p * q / 100, 2),
                }
    except Exception as e:
        logger.error(f"CSV lookup error: {e}")
    return _estimate_calories_ai(food_name, quantity_grams)

def _estimate_calories_ai(food_name: str, quantity_grams: int) -> Dict:
    """Fallback: ask OpenAI for nutrition info. Returns found=False on any error."""
    prompt = (
        f"Estimate nutritional info for {quantity_grams}g of {food_name}.\n"
        "Respond ONLY in this exact format, no other text:\n"
        "Calories: [number]\nFat: [number]g\nCarbs: [number]g\nProtein: [number]g"
    )
    try:
        resp = client_openai.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
        )
    except openai.RateLimitError as e:
        logger.error(f"OpenAI rate limit (calorie estimate): {e}")
        return {"found": False, "calories": None, "food_name": food_name, "quantity": quantity_grams}
    except openai.AuthenticationError as e:
        logger.error(f"OpenAI auth error (calorie estimate): {e}")
        return {"found": False, "calories": None, "food_name": food_name, "quantity": quantity_grams}
    except openai.APIConnectionError as e:
        logger.error(f"OpenAI connection error (calorie estimate): {e}")
        return {"found": False, "calories": None, "food_name": food_name, "quantity": quantity_grams}
    except openai.APIStatusError as e:
        logger.error(f"OpenAI API error {e.status_code} (calorie estimate): {e.message}")
        return {"found": False, "calories": None, "food_name": food_name, "quantity": quantity_grams}
    except Exception as e:
        logger.error(f"Unexpected error (calorie estimate): {e}")
        return {"found": False, "calories": None, "food_name": food_name, "quantity": quantity_grams}

    try:
        text  = resp.choices[0].message.content.strip()
        cal   = re.search(r"Calories:\s*(\d+)",   text)
        fat   = re.search(r"Fat:\s*([\d.]+)",     text)
        carbs = re.search(r"Carbs:\s*([\d.]+)",   text)
        prot  = re.search(r"Protein:\s*([\d.]+)", text)
        if cal:
            total = int(cal.group(1))
            return {
                "found":             True,
                "calories":          total,
                "calories_per_100g": round(total * 100 / quantity_grams, 1),
                "food_name":         food_name,
                "quantity":          quantity_grams,
                "fat":               float(fat.group(1))   if fat   else 0,
                "carbs":             float(carbs.group(1)) if carbs else 0,
                "protein":           float(prot.group(1))  if prot  else 0,
            }
    except Exception as e:
        logger.error(f"Failed to parse OpenAI calorie response: {e}")
    return {"found": False, "calories": None, "food_name": food_name, "quantity": quantity_grams}

def _filter_meals(user_profile: UserProfile) -> pd.DataFrame:
    if meals_df.empty:
        return meals_df
    df   = meals_df.copy()
    diet = user_profile.diet.value.lower()

    if diet == "vegetarian":
        df = df[df["Diet Type"].astype(str).str.lower().str.contains("vegetarian|vegan", na=False)]
    elif diet == "vegan":
        df = df[df["Diet Type"].astype(str).str.lower().str.contains("vegan", na=False)]
    elif diet == "pescatarian":
        df = df[~df["Diet Type"].astype(str).str.lower().str.contains("non-veg|meat|chicken", na=False)]

    for allergen in user_profile.food_allergies:
        a = allergen.lower().strip()
        if a and a != "none":
            df = df[~df["Allergens"].astype(str).str.lower().str.contains(a, na=False)]

    chunks = [
        df[df["Meal Time"].astype(str).str.contains(t, case=False, na=False)].head(10)
        for t in ["Breakfast", "Lunch", "Dinner", "Snack"]
    ]
    chunks = [c for c in chunks if not c.empty]
    result = pd.concat(chunks, ignore_index=True) if chunks else df.head(50)
    result = result.drop_duplicates(subset=["Dish Name"])
    return result if not result.empty else df.head(50)

def _build_meal_plan(user_profile: UserProfile) -> Dict:
    bmi = calculate_bmi(user_profile.weight, user_profile.height)
    bmr = calculate_bmr(user_profile.weight, user_profile.height,
                        user_profile.age, user_profile.gender.value)

    if bmi < 18.5:
        calorie_range, calorie_goal = f"{int(bmr*1.15)}-{int(bmr*1.30)}", "ABOVE BMR"
    elif bmi >= 25:
        calorie_range, calorie_goal = f"{int(bmr*0.75)}-{int(bmr*0.90)}", "BELOW BMR"
    else:
        calorie_range, calorie_goal = f"{int(bmr*0.95)}-{int(bmr*1.05)}", "AROUND BMR"

    meals_list = _filter_meals(user_profile).to_string(index=False)
    prompt = f"""Expert dietitian: Create 5-meal plan.

PROFILE: {user_profile.name}, {user_profile.age}y, {user_profile.gender.value}
BMI: {bmi} ({calorie_goal}) | Calories: {calorie_range}kcal
Diet: {user_profile.diet.value} | Activity: {user_profile.activity_level.value}
Health Goals: {', '.join(user_profile.health_goals) if user_profile.health_goals else 'General wellness'}

MEALS (pick 5):
{meals_list}

OUTPUT (one line each):
Breakfast: [Name] (cal, P, C, F) - reason
Mid-Morning Snack: [Name] (cal, P, C, F) - reason
Lunch: [Name] (cal, P, C, F) - reason
Afternoon Snack: [Name] (cal, P, C, F) - reason
Dinner: [Name] (cal, P, C, F) - reason"""

    try:
        resp = client_openai.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400,
        )
    except openai.RateLimitError as e:
        logger.error(f"OpenAI rate limit: {e}")
        raise HTTPException(status_code=429, detail="OpenAI rate limit reached. Please try again in a moment.")
    except openai.AuthenticationError as e:
        logger.error(f"OpenAI authentication failed: {e}")
        raise HTTPException(status_code=500, detail="OpenAI authentication error. Check your API key.")
    except openai.APIConnectionError as e:
        logger.error(f"OpenAI connection error: {e}")
        raise HTTPException(status_code=503, detail="Could not reach OpenAI. Check your internet connection.")
    except openai.APIStatusError as e:
        logger.error(f"OpenAI API error {e.status_code}: {e.message}")
        raise HTTPException(status_code=502, detail=f"OpenAI returned an error: {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error calling OpenAI: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error generating meal plan.")

    return {
        "meal_plan":     resp.choices[0].message.content.strip(),
        "bmi":           bmi,
        "bmr":           bmr,
        "calorie_range": calorie_range,
        "daily_goal":    calorie_goal,
    }

def parse_meal_calories(meal_text: str) -> Dict[str, int]:
    result = {}
    for line in meal_text.splitlines():
        line = line.strip()
        for mt in ["Breakfast", "Mid-Morning Snack", "Lunch", "Afternoon Snack", "Dinner"]:
            if line.startswith(mt + ":"):
                m = re.search(r"\((\d+)", line)
                if m:
                    result[mt] = int(m.group(1))
                break
    return result

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {"message": "Meal Plan Generator API", "status": "running", "version": "4.0.0"}


@app.get("/health", tags=["Health"])
async def health_check():
    mongo_ok = False
    try:
        await db.command("ping")
        mongo_ok = True
    except Exception:
        pass
    return {
        "status":                "healthy" if mongo_ok else "degraded",
        "mongodb_connected":     mongo_ok,
        "meals_loaded":          not meals_df.empty     if "meals_df"     in globals() else False,
        "nutrition_data_loaded": not nutrition_df.empty if "nutrition_df" in globals() else False,
        "openai_configured":     bool(openai_api_key),
    }


@app.post("/register", response_model=RegisterResponse, tags=["Users"])
async def register(request: RegisterRequest):
    """
    **Step 1 — Create an account.**

    - Email is accepted here and nowhere else.
    - Email is stored lowercase (Alice@gmail.com == alice@gmail.com).
    - Returns a **user_id** (UUID). The frontend must store this and send it
      with every subsequent API call.
    - Returns **409** if the email is already registered.

    ```
    POST /register
    { "email": "alice@example.com", "name": "Alice" }

    Response:
    { "user_id": "550e8400-e29b-41d4-a716-446655440000", "name": "Alice", ... }
    ```
    """
    email   = request.email.lower().strip()
    user_id = await db_register_user(email, request.name)
    return RegisterResponse(
        message="Registration successful! Save your user_id — you will need it for all future requests.",
        user_id=user_id,
        name=request.name,
    )


@app.post("/generate-meal-plan", response_model=MealPlanResponse, tags=["Meal Planning"])
async def create_meal_plan(request: MealPlanRequest):
    """
    **Step 2 — Generate a personalized meal plan.**

    - Requires `user_id` (returned by `/register`) inside `user_profile`.
    - No email needed.
    - Returns **404** if `user_id` is not found.
    - Updates the user's health profile and saves the generated plan to MongoDB.

    ```
    POST /generate-meal-plan
    {
      "user_profile": {
        "user_id": "550e8400-...",
        "name": "Alice",
        "age": 28,
        ...
      }
    }
    ```
    """
    # Verify user exists before doing any expensive work
    await db_get_user(request.user_profile.user_id)

    result = _build_meal_plan(request.user_profile)
    await db_save_user_profile(request.user_profile)
    await db_save_meal_plan(request.user_profile.user_id, result)
    return MealPlanResponse(**result)


@app.post("/modify-meal-plan", tags=["Meal Planning"])
async def modify_plan(request: MealModificationRequest):
    """
    Swap one meal in an existing plan.

    Body: `meal_plan` (current text), `modification` (what to change).
    No user_id needed — operates only on the plan text provided.
    """
    if meals_df.empty:
        raise HTTPException(status_code=500, detail="Meal data not loaded")

    try:
        limited = (
            meals_df.groupby("Meal Time").head(15).to_string(index=False)
            if "Meal Time" in meals_df.columns
            else meals_df.head(50).to_string(index=False)
        )
    except Exception:
        limited = meals_df.head(50).to_string(index=False)

    prompt = f"""Dietitian: User wants: "{request.modification}"

Current plan:
{request.meal_plan}

Available meals:
{limited}

Replace ONE meal only. Keep calories +-30kcal. Don't repeat.

Output:
Breakfast: [name] (cal) - reason
Mid-Morning Snack: [name] (cal) - reason
Lunch: [name] (cal) - reason
Afternoon Snack: [name] (cal) - reason
Dinner: [name] (cal) - reason"""

    try:
        resp = client_openai.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )
    except openai.RateLimitError as e:
        logger.error(f"OpenAI rate limit: {e}")
        raise HTTPException(status_code=429, detail="OpenAI rate limit reached. Please try again in a moment.")
    except openai.AuthenticationError as e:
        logger.error(f"OpenAI authentication failed: {e}")
        raise HTTPException(status_code=500, detail="OpenAI authentication error. Check your API key.")
    except openai.APIConnectionError as e:
        logger.error(f"OpenAI connection error: {e}")
        raise HTTPException(status_code=503, detail="Could not reach OpenAI. Check your internet connection.")
    except openai.APIStatusError as e:
        logger.error(f"OpenAI API error {e.status_code}: {e.message}")
        raise HTTPException(status_code=502, detail=f"OpenAI returned an error: {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error calling OpenAI: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error modifying meal plan.")
    return {"updated_meal_plan": resp.choices[0].message.content.strip()}


@app.post("/adjust-meal-plan", tags=["Meal Planning"])
async def adjust_plan(request: MealAdjustmentRequest):
    """
    Re-balance the plan when the user eats something outside it.

    Body: `meal_plan`, `food_name`, `quantity_grams`, `meal_replaced`.
    No user_id needed — operates only on the plan text provided.
    """
    if meals_df.empty:
        raise HTTPException(status_code=500, detail="Meal data not loaded")

    food_info = get_calories_for_food(request.food_name, request.quantity_grams)
    if not food_info["found"]:
        raise HTTPException(status_code=404, detail="Could not find calorie info for this food")

    try:
        limited = (
            meals_df.groupby("Meal Time").head(12).to_string(index=False)
            if "Meal Time" in meals_df.columns
            else meals_df.head(40).to_string(index=False)
        )
    except Exception:
        limited = meals_df.head(40).to_string(index=False)

    prompt = f"""Dietitian: Adjust plan.

Current:
{request.meal_plan}

User ate: {food_info['food_name']} ({food_info['quantity']}g) = {food_info['calories']}kcal
Replaces: {request.meal_replaced}

Available meals:
{limited}

Adjust other meals to balance daily calories.

Output:
Breakfast: [name] (cal) - reason
Mid-Morning Snack: [name] (cal) - reason
Lunch: [name] (cal) - reason
Afternoon Snack: [name] (cal) - reason
Dinner: {food_info['food_name']} ({food_info['calories']}kcal) - consumed"""

    try:
        resp = client_openai.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )
    except openai.RateLimitError as e:
        logger.error(f"OpenAI rate limit: {e}")
        raise HTTPException(status_code=429, detail="OpenAI rate limit reached. Please try again in a moment.")
    except openai.AuthenticationError as e:
        logger.error(f"OpenAI authentication failed: {e}")
        raise HTTPException(status_code=500, detail="OpenAI authentication error. Check your API key.")
    except openai.APIConnectionError as e:
        logger.error(f"OpenAI connection error: {e}")
        raise HTTPException(status_code=503, detail="Could not reach OpenAI. Check your internet connection.")
    except openai.APIStatusError as e:
        logger.error(f"OpenAI API error {e.status_code}: {e.message}")
        raise HTTPException(status_code=502, detail=f"OpenAI returned an error: {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error calling OpenAI: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error adjusting meal plan.")
    return {"adjusted_meal_plan": resp.choices[0].message.content.strip()}


@app.post("/get-food-calories", response_model=CalorieInfo, tags=["Nutrition"])
async def get_food_calories(food_name: str, quantity_grams: int):
    """Calorie + macro lookup. Query params: `food_name`, `quantity_grams`."""
    return get_calories_for_food(food_name, quantity_grams)


@app.post("/parse-calories", tags=["Nutrition"])
async def extract_meal_calories(meal_plan: str):
    """Parse meal plan text and return calories per meal + total."""
    calories = parse_meal_calories(meal_plan)
    return {"calories": calories, "total_calories": sum(calories.values())}


@app.get("/calculate-bmi", tags=["Health Metrics"])
async def bmi_endpoint(weight_kg: float, height_cm: float):
    if weight_kg <= 0 or height_cm <= 0:
        raise HTTPException(status_code=400, detail="Invalid weight or height")
    bmi = calculate_bmi(weight_kg, height_cm)
    if bmi < 18.5:  category = "Underweight"
    elif bmi < 25:  category = "Normal weight"
    elif bmi < 30:  category = "Overweight"
    else:           category = "Obese"
    return {"bmi": bmi, "category": category}


@app.get("/calculate-bmr", tags=["Health Metrics"])
async def bmr_endpoint(weight_kg: float, height_cm: float, age: int, gender: Gender):
    if weight_kg <= 0 or height_cm <= 0 or age <= 0:
        raise HTTPException(status_code=400, detail="Invalid input values")
    return {"bmr": calculate_bmr(weight_kg, height_cm, age, gender.value)}


# ─── Error handlers ───────────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
