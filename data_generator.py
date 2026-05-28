import pandas as pd
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)

# Configuration for Mumbai Real Estate
localities = {
    'South Mumbai': {'base_rate': 52000, 'pool_prob': 0.6, 'gym_prob': 0.7},
    'Worli': {'base_rate': 44000, 'pool_prob': 0.7, 'gym_prob': 0.8},
    'Bandra West': {'base_rate': 46000, 'pool_prob': 0.5, 'gym_prob': 0.6},
    'Juhu': {'base_rate': 41000, 'pool_prob': 0.6, 'gym_prob': 0.6},
    'Andheri West': {'base_rate': 25000, 'pool_prob': 0.4, 'gym_prob': 0.5},
    'Powai': {'base_rate': 22000, 'pool_prob': 0.8, 'gym_prob': 0.9},
    'Chembur': {'base_rate': 19000, 'pool_prob': 0.3, 'gym_prob': 0.4},
    'Borivali West': {'base_rate': 18000, 'pool_prob': 0.3, 'gym_prob': 0.4},
    'Vashi (Navi Mumbai)': {'base_rate': 15000, 'pool_prob': 0.5, 'gym_prob': 0.6},
    'Thane West': {'base_rate': 13000, 'pool_prob': 0.6, 'gym_prob': 0.7}
}

num_records = 1600
data = []

for _ in range(num_records):
    # Select locality
    locality = np.random.choice(list(localities.keys()))
    loc_info = localities[locality]
    
    # BHK distribution: 1 BHK (30%), 2 BHK (45%), 3 BHK (20%), 4 BHK (5%)
    bhk = int(np.random.choice([1, 2, 3, 4], p=[0.30, 0.45, 0.20, 0.05]))
    
    # Area based on BHK
    if bhk == 1:
        area = int(np.random.uniform(400, 650))
    elif bhk == 2:
        area = int(np.random.uniform(750, 1100))
    elif bhk == 3:
        area = int(np.random.uniform(1200, 1800))
    else:  # 4 BHK
        area = int(np.random.uniform(2000, 3400))
        
    # Bathrooms: typically matching BHK or +/- 1, min 1
    bathrooms = max(1, bhk + int(np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])))
    if bhk >= 3 and bathrooms < 2:
        bathrooms = bhk - 1
        
    # Property age (0 to 25 years)
    age = int(np.random.choice(range(26), p=[0.1] + [0.9/25]*25)) # Higher chance of new properties (0 years)
    
    # Floors structure
    total_floors = int(np.random.choice(range(4, 41), p=[0.05]*5 + [0.75/32]*32))
    floor = int(np.random.randint(1, total_floors + 1))
    
    # Amenities
    parking = int(np.random.choice([0, 1], p=[0.15 if bhk > 1 else 0.4, 0.85 if bhk > 1 else 0.6]))
    lift = 1 if total_floors > 4 else int(np.random.choice([0, 1], p=[0.3, 0.7]))
    
    # Local probabilities for pool and gym
    swimming_pool = int(np.random.choice([0, 1], p=[1 - loc_info['pool_prob'], loc_info['pool_prob']]))
    gym = int(np.random.choice([0, 1], p=[1 - loc_info['gym_prob'], loc_info['gym_prob']]))
    
    # Pricing formula (base price + adjustments + noise)
    # Rate per sqft
    base_rate = loc_info['base_rate']
    
    # Calculate base price
    base_price = area * base_rate
    
    # Age depreciation (max 20% discount for 20+ years)
    age_discount = max(0.8, 1.0 - (age * 0.01))
    
    # Floor premium (higher floors in high-rise cost more, up to +6% premium)
    floor_premium = 1.0 + (floor / total_floors) * 0.06 if total_floors > 10 else 1.0
    
    # Amenity additions (in INR)
    amenity_value = 0
    if parking:
        amenity_value += 800000  # 8 Lakhs
    if swimming_pool:
        amenity_value += 1200000  # 12 Lakhs
    if gym:
        amenity_value += 600000  # 6 Lakhs
    if lift:
        amenity_value += 300000  # 3 Lakhs
        
    # Calculate adjusted price (in INR)
    price_inr = (base_price * age_discount * floor_premium) + amenity_value
    
    # Add random noise (normal distribution, standard deviation 4.5%)
    noise = np.random.normal(0, 0.045 * price_inr)
    final_price_inr = price_inr + noise
    
    # Convert to Lakhs INR (1 Lakh = 100,000)
    price_lakhs = round(final_price_inr / 100000, 2)
    
    data.append({
        'Locality': locality,
        'Area_SqFt': area,
        'BHK': bhk,
        'Bathrooms': bathrooms,
        'Property_Age_Years': age,
        'Floor_Num': floor,
        'Total_Floors': total_floors,
        'Parking_Available': parking,
        'Swimming_Pool': swimming_pool,
        'Gym_Available': gym,
        'Lift_Available': lift,
        'Price_Lakhs': price_lakhs
    })

df = pd.DataFrame(data)

# Save to CSV
output_path = os.path.join(os.path.dirname(__file__), 'mumbai_house_data.csv')
df.to_csv(output_path, index=False)
print(f"Dataset generated successfully at {output_path} with {len(df)} records.")
print("\nFirst 5 rows of dataset:")
print(df.head())
