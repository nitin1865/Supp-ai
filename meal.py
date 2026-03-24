import streamlit as st
import pandas as pd
import os
import openai
import matplotlib.pyplot as plt
import re
from dotenv import load_dotenv

st.set_page_config(page_title="🥗 Personalized Meal Plan Generator", layout="wide")

load_dotenv()

# ---------- OPENAI CONFIGURATION ----------
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("⚠️ OPENAI_API_KEY not found. Please set it in your .env file")
    st.stop()

client = openai.OpenAI(api_key=openai_api_key)

# Using gpt-4o-mini for cost efficiency
# Cost: ~$0.00015 per 1K input tokens, ~$0.0006 per 1K output tokens
MODEL = "gpt-4o-mini"

# ---------- LOAD DATASETS ----------
@st.cache_data
def load_meals(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df

meals_df = load_meals("meal.csv")

@st.cache_data
def load_nutrition(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

nutrition = load_nutrition("nutrition_data.csv")

def get_calories_for_food(food_name: str, quantity_grams: int) -> dict:
    """
    Search for calorie information for a specific food and quantity from the dataset
    Returns: dict with 'found', 'calories', 'food_name', 'quantity' keys
    """
    try:
        # Search for the food in the dataset (case-insensitive)
        food_matches = nutrition[nutrition['Food Item'].str.contains(food_name, case=False, na=False)]
        
        if not food_matches.empty:
            # Take the first match
            food_row = food_matches.iloc[0]
            
            # Convert to numeric, handling potential string values
            try:
                calories_per_100g = float(food_row['Calories'])
                fat_per_100g = float(food_row['Fat(g)'])
                carbs_per_100g = float(food_row['Carbs(g)'])
                protein_per_100g = float(food_row['Protein(g)'])
            except (ValueError, TypeError) as e:
                st.error(f"Data conversion error for {food_row['Food Item']}: {str(e)}")
                return estimate_calories_with_ai(food_name, quantity_grams)
            
            # Calculate nutritional values for the given quantity
            total_calories = int((calories_per_100g * quantity_grams) / 100)
            total_fat = round((fat_per_100g * quantity_grams) / 100, 2)
            total_carbs = round((carbs_per_100g * quantity_grams) / 100, 2)
            total_protein = round((protein_per_100g * quantity_grams) / 100, 2)
            
            return {
                'found': True,
                'calories': total_calories,
                'calories_per_100g': calories_per_100g,
                'food_name': food_row['Food Item'],
                'quantity': quantity_grams,
                'fat': total_fat,
                'carbs': total_carbs,
                'protein': total_protein
            }
        else:
            # If not found in dataset, use AI to estimate
            return estimate_calories_with_ai(food_name, quantity_grams)
            
    except Exception as e:
        st.error(f"Error searching for food: {str(e)}")
        return {
            'found': False,
            'calories': None,
            'food_name': food_name,
            'quantity': quantity_grams
        }

def estimate_calories_with_ai(food_name: str, quantity_grams: int) -> dict:
    """Use OpenAI to estimate calories when food is not in dataset"""
    try:
        prompt = f"""Estimate the nutritional information for {quantity_grams}g of {food_name}.

Provide ONLY the response in this exact format, no other text:
Calories: [number]
Fat: [number]g
Carbs: [number]g  
Protein: [number]g"""
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse the AI response
        calories_match = re.search(r'Calories:\s*(\d+)', response_text)
        fat_match = re.search(r'Fat:\s*([\d.]+)', response_text)
        carbs_match = re.search(r'Carbs:\s*([\d.]+)', response_text)
        protein_match = re.search(r'Protein:\s*([\d.]+)', response_text)
        
        if calories_match:
            return {
                'found': True,
                'calories': int(calories_match.group(1)),
                'calories_per_100g': int((int(calories_match.group(1)) * 100) / quantity_grams),
                'food_name': food_name,
                'quantity': quantity_grams,
                'fat': float(fat_match.group(1)) if fat_match else 0,
                'carbs': float(carbs_match.group(1)) if carbs_match else 0,
                'protein': float(protein_match.group(1)) if protein_match else 0
            }
    
    except Exception as e:
        st.error(f"AI estimation failed: {str(e)}")
    
    return {
        'found': False,
        'calories': None,
        'food_name': food_name,
        'quantity': quantity_grams
    }

# ---------- BMI and BMR functions ----------
def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 1)

def calculate_bmr(weight_kg, height_cm, age, gender):
    if gender.lower().startswith('m'):
        return round(10 * weight_kg + 6.25 * height_cm - 5 * age + 5, 1)
    else:
        return round(10 * weight_kg + 6.25 * height_cm - 5 * age - 161, 1)

# ---------- HELPER: FILTER MEALS BY DIETARY PREFERENCES ----------
def filter_meals_by_preferences(meals_df: pd.DataFrame, user_profile: dict) -> pd.DataFrame:
    """
    Filter meals to reduce token count while respecting user preferences
    Optimized for the specific CSV structure
    """
    filtered_df = meals_df.copy()
    
    # Your CSV column names
    name_col = 'Dish Name'
    time_col = 'Meal Time'
    diet_col = 'Diet Type'
    allergen_col = 'Allergens'
    
    try:
        # Filter by dietary pattern
        diet_type = user_profile['diet'].lower()
        if diet_type == 'vegetarian':
            # Keep only vegetarian meals
            filtered_df = filtered_df[filtered_df[diet_col].astype(str).str.lower().str.contains('vegetarian|vegan', na=False)]
        elif diet_type == 'vegan':
            # Keep only vegan meals
            filtered_df = filtered_df[filtered_df[diet_col].astype(str).str.lower().str.contains('vegan', na=False)]
        elif diet_type == 'pescatarian':
            # Keep vegetarian + fish
            filtered_df = filtered_df[~filtered_df[diet_col].astype(str).str.lower().str.contains('non-veg|meat|chicken', na=False)]
        
        # Filter out allergens
        for allergen in user_profile['food_allergies']:
            allergen_lower = allergen.lower().strip()
            if allergen_lower and allergen_lower != 'none':
                filtered_df = filtered_df[~filtered_df[allergen_col].astype(str).str.lower().str.contains(allergen_lower, na=False)]
        
        # Group by meal time and take top 10 meals per time
        meal_times = ['Breakfast', 'Lunch', 'Dinner', 'Snack']
        limited_meals = []
        
        for meal_time in meal_times:
            meals_of_time = filtered_df[filtered_df[time_col].astype(str).str.contains(meal_time, case=False, na=False)]
            if not meals_of_time.empty:
                limited_meals.append(meals_of_time.head(10))
        
        # If no meals matched by time, just take top meals
        if not limited_meals:
            result_df = filtered_df.head(50)
        else:
            result_df = pd.concat(limited_meals, ignore_index=True)
        
        # Remove duplicates
        result_df = result_df.drop_duplicates(subset=[name_col])
        
        return result_df if not result_df.empty else filtered_df.head(50)
        
    except Exception as e:
        # Fallback: return limited meals if any error
        st.warning(f"⚠️ Note: Using limited meal selection due to filtering. Error: {str(e)}")
        return filtered_df.head(50)

# ---------- MEAL RECOMMENDATION ----------
def generate_meal_plan(user_profile: dict, meals_df: pd.DataFrame) -> str:
    bmi_value = calculate_bmi(user_profile['weight'], user_profile['height'])
    bmr = calculate_bmr(
        user_profile['weight'], user_profile['height'], user_profile['age'], user_profile['gender']
    )

    # OPTIMIZE: Filter meals to reduce token count
    filtered_meals = filter_meals_by_preferences(meals_df, user_profile)
    meals_list = filtered_meals.to_string(index=False)
    
    # Calculate target calories
    if bmi_value < 18.5:
        calorie_range = f"{int(bmr * 1.15)}-{int(bmr * 1.30)}"
        calorie_goal = "ABOVE BMR"
    elif bmi_value >= 25:
        calorie_range = f"{int(bmr * 0.75)}-{int(bmr * 0.90)}"
        calorie_goal = "BELOW BMR"
    else:
        calorie_range = f"{int(bmr * 0.95)}-{int(bmr * 1.05)}"
        calorie_goal = "AROUND BMR"

    prompt = f"""Expert dietitian: Create 5-meal plan.

PROFILE: {user_profile['name']}, {user_profile['age']}y, {user_profile['gender']}
BMI: {bmi_value} ({calorie_goal}) | Calories: {calorie_range}kcal
Diet: {user_profile['diet']} | Activity: {user_profile['activity_level']}

MEALS (pick 5):
{meals_list}

OUTPUT (one line each):
Breakfast: [Name] (cal, P, C, F) - reason
Mid-Morning Snack: [Name] (cal, P, C, F) - reason
Lunch: [Name] (cal, P, C, F) - reason
Afternoon Snack: [Name] (cal, P, C, F) - reason
Dinner: [Name] (cal, P, C, F) - reason"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating meal plan: {str(e)}")
        return "Unable to generate meal plan. Please try again."

# ---------- MEAL CONFIGURATION BY USER ----------
def meal_change_according_to_user(user_input: str, recommended_meal: str, meals_df: pd.DataFrame) -> str:
    # OPTIMIZE: Use only subset of meals (top 15 per meal time)
    try:
        limited_meals = meals_df.groupby('Meal Time').head(15).to_string(index=False)
    except:
        # Fallback if groupby fails
        limited_meals = meals_df.head(50).to_string(index=False)
    
    prompt = f"""Dietitian: User wants: "{user_input}"

Current plan:
{recommended_meal}

Available meals:
{limited_meals}

Replace ONE meal only. Keep calories ±30kcal. Don't repeat.

Output:
Breakfast: [name] (cal) - reason
Mid-Morning Snack: [name] (cal) - reason
Lunch: [name] (cal) - reason
Afternoon Snack: [name] (cal) - reason
Dinner: [name] (cal) - reason"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error modifying meal plan: {str(e)}")
        return recommended_meal

# ---------- MEAL PLAN ADJUSTMENT ----------
def adjust_meal_plan_with_consumed_food(consumed_food_info: dict, current_meal_plan: str, meal_replaced: str) -> str:
    """Adjust the meal plan based on what the user actually consumed"""
    # OPTIMIZE: Use only subset of meals
    try:
        limited_meals = meals_df.groupby('Meal Time').head(12).to_string(index=False)
    except:
        # Fallback if groupby fails
        limited_meals = meals_df.head(40).to_string(index=False)
    
    prompt = f"""Dietitian: Adjust plan.

Current:
{current_meal_plan}

User ate: {consumed_food_info['food_name']} ({consumed_food_info['quantity']}g) = {consumed_food_info['calories']}kcal
Replaces: {meal_replaced}

Available meals:
{limited_meals}

Adjust other meals to balance daily calories.

Output:
Breakfast: [name] (cal) - reason
Mid-Morning Snack: [name] (cal) - reason
Lunch: [name] (cal) - reason
Afternoon Snack: [name] (cal) - reason
Dinner: {consumed_food_info['food_name']} ({consumed_food_info['calories']}kcal) - consumed"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error adjusting meal plan: {str(e)}")
        return current_meal_plan

# ---------- PARSE MEAL CALORIES ----------
def parse_meal_calories(meal_text: str) -> dict:
    """Parse meal plan text to extract calories for each meal type"""
    calories = {}
    meal_types = ['Breakfast', 'Mid-Morning Snack', 'Lunch', 'Afternoon Snack', 'Dinner']

    for line in meal_text.splitlines():
        line = line.strip()
        for meal_type in meal_types:
            if line.startswith(meal_type + ':'):
                # robust calorie extraction
                match = re.search(
                    r'(\d{2,4})\s*(kcal|cal|Calories)?',
                    line,
                    re.IGNORECASE
                )
                if match:
                    calories[meal_type] = int(match.group(1))
                break

    return calories

    """Parse meal plan text to extract calories for each meal type"""
    calories = {}
    meal_types = ['Breakfast', 'Mid-Morning Snack', 'Lunch', 'Afternoon Snack', 'Dinner']
    
    for line in meal_text.splitlines():
        line = line.strip()
        if ':' in line and '(' in line:
            for meal_type in meal_types:
                if line.startswith(meal_type + ':'):
                    try:
                        match = re.search(r'\((\d+)', line)
                        if match:
                            calories[meal_type] = int(match.group(1))
                            break
                    except:
                        continue
    return calories

# ---------- STREAMLIT UI ----------
st.title("🥗 Personalized Meal Plan Generator")
st.markdown("*Create customized meal plans based on your health goals*")
st.markdown(f"**Powered by OpenAI ({MODEL})** 🚀")
st.divider()

# Create layout columns
col1, col2, col3 = st.columns([4, 2, 2], gap="large")

with col1:
    st.header("👤 Tell us about yourself")
    
    # User Input Form
    with st.form("user_info_form"):
        # Personal Information
        st.subheader("📋 Personal Details")
        
        pers_col1, pers_col2 = st.columns(2)
        
        with pers_col1:
            name = st.text_input("Name", placeholder="Enter your name")
            age = st.number_input("Age", min_value=5, max_value=120, step=1, value=25)
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, step=1.0, value=170.0)
        
        with pers_col2:
            food_type = st.selectbox(
                "Preferred Cuisine",
                options=meals_df['Cuisine'].unique() if 'meals_df' in locals() else ['Indian', 'Continental', 'Asian', 'Mediterranean']
            )
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, step=1.0, value=70.0)
        
        st.divider()
        
        # Health Goals
        st.subheader("🎯 Health & Fitness Goals")
        health_goals = st.multiselect(
            "What are your primary health goals? (Select all that apply)",
            options=[
                "Weight management", "Energy boost", "Hormonal balance", "Bone & joint health",
                "Muscle gain", "Skin, hair & nails", "Gut health", "Immune support",
                "Fertility & reproductive health", "Menstrual wellness / Menopause support", "Other"
            ],
            help="Choose your main health objectives to get personalized recommendations"
        )

        st.divider()
        
        # Dietary Preferences
        st.subheader("🍽️ Dietary Information")
        diet_col1, diet_col2 = st.columns(2)
        
        with diet_col1:
            diet = st.selectbox(
                "Dietary Pattern",
                options=["Omnivore", "Vegetarian", "Vegan", "Pescatarian", "Low-carb/Keto", "Gluten-free", "Jain/Sattvic", "Other"]
            )
            
            activity_level = st.selectbox(
                "Activity Level",
                options=[
                    "Little movement", 
                    "Walking, stretching everyday", 
                    "Exercise 2–3 times a week", 
                    "Daily training"
                ]
            )
        
        with diet_col2:
            food_allergies = st.multiselect(
                "Food Allergies & Intolerances",
                options=[
                    "Dairy (lactose or casein)", "Gluten (celiac or intolerance)", "Soy", "Peanuts", 
                    "Tree nuts", "Eggs", "Shellfish", "Fish", "Corn",
                    "Nightshades (tomatoes, peppers)", "Artificial sweeteners", "Caffeine", "Alcohol"
                ]
            )

        st.divider()
        
        # Health Conditions Section
        st.subheader("🏥 Health Conditions")
        disease = st.multiselect(
            "Do you currently manage any of these conditions?",
            options=[
                "Diabetes / Insulin resistance", "PCOS/PCOD", "High blood pressure / cholesterol", 
                "Thyroid imbalance (Hypo/Hyper)", "Hormonal Imbalance", "Digestive disorders (IBS, IBD, acidity)", 
                "Menstrual issues or menopause symptoms", "Autoimmune disease"
            ]
        )

        st.divider()
        
        # Supplement Preferences Section
        st.subheader("💊 Supplement Preferences (Optional)")
        supplement_preferences = st.multiselect(
            "What type of supplements do you prefer?",
            options=[
                "Herbal options", "Clinical-grade with scientific backing", "Vegan options",
                "Gluten-free options", "Gelatin-free options", "Soy-free options", "Corn-free options",
                "Cruelty-free options"
            ]
        )
        
        # Submit button
        st.markdown("")
        submitted = st.form_submit_button(
            "🚀 Generate My Meal Plan", 
            type="primary", 
            use_container_width=True
        )

    # Generate Meal Plan
    if submitted:
        if not name:
            st.error("Please enter your name!")
        else:
            user_profile = {
                "name": name,
                "age": age,
                "gender": gender,
                "height": height,
                "weight": weight,
                "health_goals": health_goals if health_goals else ["General wellness"],
                "food_allergies": food_allergies if food_allergies else ["None"],
                "diet": diet,
                "disease": disease if disease else ["None"],
                "activity_level": activity_level,
                "supplement_preferences": supplement_preferences if supplement_preferences else ["None"]
            }
            
            with st.spinner("🔄 Creating your personalized meal plan..."):
                meal_plan_text = generate_meal_plan(user_profile, meals_df)
                st.session_state['meal_plan'] = meal_plan_text
                
                # Initialize meal tracking
                if 'meal_completion' not in st.session_state:
                    st.session_state['meal_completion'] = {}
                
                st.success("✅ Your meal plan is ready!")

    # Display Meal Plan
    if 'meal_plan' in st.session_state:
        st.divider()
        st.header("🍽️ Your Personalized Meal Plan")
        st.markdown(st.session_state['meal_plan'])

    # Meal Modification Section
    if 'meal_plan' in st.session_state:
        st.divider()
        st.header("🔄 Modify Your Plan")
        
        with st.form("modification_form"):
            st.write("Need changes to your meal plan? Let me know!")
            user_input = st.text_area(
                "Describe the changes you'd like:",
                placeholder="e.g., 'Replace chicken with tofu', 'Add more vegetables', 'Reduce calories'"
            )
            change_submitted = st.form_submit_button("✨ Apply Changes", type="primary")
        
        if change_submitted and user_input:
            with st.spinner("🔄 Updating your meal plan..."):
                updated_meal = meal_change_according_to_user(user_input, st.session_state['meal_plan'], meals_df)
                st.session_state['meal_plan'] = updated_meal
            
            st.success("✅ Your meal plan has been updated!")
            st.rerun()

# Progress Tracking Column
with col2:
    st.header("📊 Daily Progress")
    if 'meal_plan' in st.session_state:
        calories_map = parse_meal_calories(st.session_state['meal_plan'])
        
        if calories_map:
            total_required = sum(calories_map.values())
            
            # Initialize tracking
            if 'meal_completion' not in st.session_state:
                st.session_state['meal_completion'] = {meal: False for meal in calories_map.keys()}

            st.subheader("✅ Track Your Meals")
            
            consumed_calories = 0

            # Meal checkboxes
            for meal_type, cal in calories_map.items():
                completed = st.checkbox(
                    f"{meal_type} ({cal} kcal)",
                    key=f"meal_{meal_type}",
                    value=st.session_state['meal_completion'].get(meal_type, False)
                )

                st.session_state['meal_completion'][meal_type] = completed
                if completed:
                    consumed_calories += cal

            # Progress metrics
            remaining_calories = max(total_required - consumed_calories, 0)
            progress_percentage = (consumed_calories / total_required) * 100 if total_required > 0 else 0

            st.divider()

            # Display metrics
            st.metric("Consumed", f"{consumed_calories} kcal")
            st.metric("Remaining", f"{remaining_calories} kcal")
            st.metric("Progress", f"{progress_percentage:.1f}%")

            # Progress bar
            st.progress(min(progress_percentage / 100, 1.0))

            if consumed_calories >= total_required:
                st.success("🎉 Daily goal completed!")
            elif consumed_calories > 0:
                st.info(f"📈 {progress_percentage:.1f}% complete")

        else:
            st.error("Unable to parse meal plan calories")
    else:
        st.info("🍽️ Generate a meal plan to start tracking!")

# Adjustment Helper Column
with col3:
    st.header("🔧 Ate Something Different?")
    
    if 'meal_plan' in st.session_state:
        st.write("Tell me what you ate and I'll adjust your plan!")
        
        with st.form("adjustment_form"):
            # Meal selection
            meal_to_replace = st.selectbox(
                "Which meal did you replace?",
                options=["Breakfast", "Mid-Morning Snack", "Lunch", "Afternoon Snack", "Dinner"]
            )
            
            # Food input
            food_name = st.text_input(
                "What did you eat?",
                placeholder="e.g., Pizza, Burger, Rice"
            )
            
            quantity_grams = st.number_input(
                "Quantity (in grams)",
                min_value=1,
                max_value=1000,
                value=100,
                step=10
            )
            
            adjust_submitted = st.form_submit_button("🔄 Adjust My Plan", type="primary")
            
            if adjust_submitted and food_name:
                with st.spinner("🔄 Searching for calories and adjusting plan..."):
                    # Get calorie information
                    food_info = get_calories_for_food(food_name, quantity_grams)
                    
                    if food_info['found']:
                        # Show what was consumed
                        st.success(f"✅ **Found:** {food_info['food_name']}")
                        st.info(f"📊 **{food_info['quantity']}g = {food_info['calories']} kcal**")
                        
                        # Adjust the meal plan
                        adjusted_plan = adjust_meal_plan_with_consumed_food(
                            food_info, 
                            st.session_state['meal_plan'], 
                            meal_to_replace
                        )
                        st.session_state['meal_plan'] = adjusted_plan
                        
                        st.success("✅ Your meal plan has been adjusted!")
                        st.rerun()
                    else:
                        st.error("❌ Could not find calorie information for this food. Please try a different name.")

        st.divider()

        # Quick tips
        st.subheader("💡 Tips")
        tips = [
            "🥤 Drink water before meals",
            "🥗 Fill half plate with vegetables", 
            "🚶‍♀️ Walk after eating",
            "⏰ Eat slowly and mindfully"
        ]

        for tip in tips:
            st.write(tip)
    else:
        st.info("🍽️ Generate a meal plan first!")

# Footer
st.divider()
st.markdown("---")
st.markdown("🌟 **Your journey to better health starts here!** 🌟")