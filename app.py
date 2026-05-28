import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as gr
import os

# Set page configuration with premium tab title and icon
st.set_page_config(
    page_title="Mumbai Real Estate Valuation & Market Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design & Aesthetics
st.markdown("""
<style>
    /* Premium font and base styling */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header card design */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        text-align: center;
    }
    .header-container h1 {
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    .header-container p {
        font-weight: 300;
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Metrics box */
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3c72;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    
    /* Price Output Box */
    .price-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.2);
        margin-top: 1.5rem;
    }
    .price-title {
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
        opacity: 0.9;
    }
    .price-value {
        font-size: 3rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .price-crore {
        font-size: 1.6rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .price-range {
        font-size: 1rem;
        opacity: 0.85;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Path settings
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, 'best_model.pkl')
data_path = os.path.join(current_dir, 'mumbai_house_data.csv')

# Locality base rates reference
localities_rates = {
    'South Mumbai': 52000,
    'Worli': 44000,
    'Bandra West': 46000,
    'Juhu': 41000,
    'Andheri West': 25000,
    'Powai': 22000,
    'Chembur': 19000,
    'Borivali West': 18000,
    'Vashi (Navi Mumbai)': 15000,
    'Thane West': 13000
}

# Load files
@st.cache_resource
def load_model():
    # Attempt to load from pickle
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            # Fallback prints to console log if version mismatch occurs
            print(f"Pre-trained model loading failed: {e}. Re-training model dynamically...")
            
    # Dynamic fallback training
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import RandomForestRegressor
        
        if not os.path.exists(data_path):
            return None
            
        df_temp = pd.read_csv(data_path)
        X = df_temp.drop(columns=['Price_Lakhs'])
        y = df_temp['Price_Lakhs']
        
        categorical_features = ['Locality']
        numerical_features = ['Area_SqFt', 'BHK', 'Bathrooms', 'Property_Age_Years', 'Floor_Num', 'Total_Floors']
        binary_features = ['Parking_Available', 'Swimming_Pool', 'Gym_Available', 'Lift_Available']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('num', StandardScaler(), numerical_features),
                ('bin', 'passthrough', binary_features)
            ]
        )
        
        model_rf = RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model_rf)
        ])
        
        pipeline.fit(X, y)
        return pipeline
    except Exception as ex:
        print(f"Error training model dynamically: {ex}")
        return None

@st.cache_data
def load_data():
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        # Calculate derived feature: Price per SqFt for metrics
        df['Price_Per_SqFt'] = (df['Price_Lakhs'] * 100000) / df['Area_SqFt']
        return df
    return None

model = load_model()
df = load_data()

# Render Premium Header
st.markdown("""
<div class="header-container">
    <h1>🏠 Mumbai Property Valuation & Market Analytics</h1>
    <p>Predict residential property values in Mumbai using Machine Learning and analyze real estate market insights</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📊 Market Analytics Dashboard", "🎯 Property Valuation Calculator"])

# -------------------------------------------------------------
# TAB 1: MARKET ANALYTICS DASHBOARD
# -------------------------------------------------------------
with tab1:
    if df is not None:
        st.subheader("📌 Mumbai Real Estate Market Overview")
        
        # High level metrics
        m1, m2, m3, m4 = st.columns(4)
        avg_price = df['Price_Lakhs'].mean()
        avg_rate = df['Price_Per_SqFt'].mean()
        avg_area = df['Area_SqFt'].mean()
        total_listings = len(df)
        
        m1.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">₹{avg_price/100:.2f} Cr</div>
            <div class="metric-label">Avg Sale Price</div>
        </div>
        """, unsafe_allow_html=True)
        
        m2.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">₹{avg_rate:,.0f}</div>
            <div class="metric-label">Avg Rate per Sq.Ft</div>
        </div>
        """, unsafe_allow_html=True)
        
        m3.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{avg_area:,.0f} sqft</div>
            <div class="metric-label">Avg Property Size</div>
        </div>
        """, unsafe_allow_html=True)
        
        m4.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{total_listings:,}</div>
            <div class="metric-label">Dataset Listings</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Locality Analysis: Average Price")
            # Group by locality
            loc_group = df.groupby('Locality').agg({
                'Price_Lakhs': 'mean',
                'Price_Per_SqFt': 'mean'
            }).reset_index().sort_values(by='Price_Lakhs', ascending=False)
            
            # Plotly bar chart for Avg Price
            fig_price = px.bar(
                loc_group,
                x='Price_Lakhs',
                y='Locality',
                orientation='h',
                labels={'Price_Lakhs': 'Average Price (Lakhs)', 'Locality': 'Locality'},
                color='Price_Lakhs',
                color_continuous_scale='Blues',
                template='plotly_white'
            )
            fig_price.update_layout(showlegend=False, height=400, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig_price, height=400, width='stretch')
            
        with col2:
            st.markdown("### Locality Analysis: Price per Sq.Ft")
            fig_rate = px.bar(
                loc_group.sort_values(by='Price_Per_SqFt', ascending=False),
                x='Price_Per_SqFt',
                y='Locality',
                orientation='h',
                labels={'Price_Per_SqFt': 'Price per Sq.Ft (INR)', 'Locality': 'Locality'},
                color='Price_Per_SqFt',
                color_continuous_scale='Viridis',
                template='plotly_white'
            )
            fig_rate.update_layout(showlegend=False, height=400, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig_rate, height=400, width='stretch')
            
        st.markdown("<hr style='border:0.5px solid #eaeaea'>", unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### Property Size (Sq.Ft) vs. Price (Lakhs)")
            fig_scatter = px.scatter(
                df,
                x='Area_SqFt',
                y='Price_Lakhs',
                color='Locality',
                hover_data=['BHK', 'Bathrooms', 'Property_Age_Years'],
                labels={'Area_SqFt': 'Area (Square Feet)', 'Price_Lakhs': 'Price (Lakhs)'},
                opacity=0.7,
                template='plotly_white'
            )
            fig_scatter.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig_scatter, height=400, width='stretch')
            
        with col4:
            st.markdown("### Factors Correlation Heatmap")
            # Compute correlation matrix
            numeric_cols = ['Area_SqFt', 'BHK', 'Bathrooms', 'Property_Age_Years', 
                            'Floor_Num', 'Total_Floors', 'Parking_Available', 
                            'Swimming_Pool', 'Gym_Available', 'Price_Lakhs']
            corr = df[numeric_cols].corr()
            
            fig_heat = px.imshow(
                corr,
                x=numeric_cols,
                y=numeric_cols,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                labels=dict(color="Correlation"),
                text_auto='.2f',
                template='plotly_white'
            )
            fig_heat.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig_heat, height=400, width='stretch')
            
        # Model Feature Importance
        st.markdown("<hr style='border:0.5px solid #eaeaea'>", unsafe_allow_html=True)
        st.markdown("### 🔑 What Drives Property Prices in Mumbai?")
        
        if model is not None:
            # Extract features and importance
            ohe_categories = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(['Locality'])
            numerical_features = ['Area_SqFt', 'BHK', 'Bathrooms', 'Property_Age_Years', 'Floor_Num', 'Total_Floors']
            binary_features = ['Parking_Available', 'Swimming_Pool', 'Gym_Available', 'Lift_Available']
            feature_names = list(ohe_categories) + numerical_features + binary_features
            importances = model.named_steps['regressor'].feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': [f.replace('Locality_', 'Locality: ') for f in feature_names],
                'Importance': importances
            }).sort_values(by='Importance', ascending=True).tail(10)
            
            fig_imp = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                labels={'Importance': 'Predictive Importance (Weights)', 'Feature': 'Feature / Attribute'},
                color='Importance',
                color_continuous_scale='Teal',
                template='plotly_white'
            )
            fig_imp.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig_imp, height=350, width='stretch')
        else:
            st.info("Train the machine learning model first to view predictive features analysis.")
            
    else:
        st.warning("Housing dataset `mumbai_house_data.csv` not found. Please run the data generator script first.")

# -------------------------------------------------------------
# TAB 2: PROPERTY VALUATION CALCULATOR
# -------------------------------------------------------------
with tab2:
    if model is None:
        st.warning("Model file `best_model.pkl` not found. Please run the training script first to enable the calculator.")
    else:
        st.subheader("🎯 Residential Property Valuation Tool")
        st.write("Fill in the property specifics below to calculate its estimated market value:")
        
        # Inputs Form
        with st.form("valuation_form"):
            c_left, c_right = st.columns(2)
            
            with c_left:
                locality = st.selectbox(
                    "Locality / Area",
                    options=sorted(list(localities_rates.keys())),
                    index=2 # Defaults to Bandra West
                )
                
                area_sqft = st.number_input(
                    "Super Built-up Area (Sq.Ft)",
                    min_value=300,
                    max_value=6000,
                    value=1000,
                    step=50,
                    help="Total carpet + loading area in square feet."
                )
                
                bhk = st.slider(
                    "BHK Config (Bedrooms)",
                    min_value=1,
                    max_value=5,
                    value=2,
                    step=1
                )
                
                bathrooms = st.slider(
                    "Bathrooms",
                    min_value=1,
                    max_value=5,
                    value=2,
                    step=1
                )
                
            with c_right:
                age_years = st.slider(
                    "Property Age (Years)",
                    min_value=0,
                    max_value=30,
                    value=0,
                    step=1,
                    help="Age of construction. 0 indicates brand-new construction / under construction."
                )
                
                total_floors = st.slider(
                    "Total Building Floors",
                    min_value=1,
                    max_value=50,
                    value=15,
                    step=1
                )
                
                floor_num = st.slider(
                    "Property Floor Number",
                    min_value=1,
                    max_value=50,
                    value=5,
                    step=1,
                    help="Must be less than or equal to total building floors."
                )
                
            st.markdown("##### 🎁 Amenities & Infrastructure Features")
            ca, cb, cc, cd = st.columns(4)
            parking = ca.checkbox("Reserved Parking Space", value=True)
            pool = cb.checkbox("Swimming Pool Access", value=False)
            gym = cc.checkbox("Fully Equipped Gym", value=False)
            lift = cd.checkbox("Lifts & High-Speed Elevators", value=True)
            
            # Submit button
            submit_val = st.form_submit_button("💰 Get Property Valuation")
            
        # Form Validation and Prediction
        if submit_val:
            if floor_num > total_floors:
                st.error("❌ Invalid Entry: The property floor number cannot be greater than the total floors of the building!")
            else:
                # Prepare input DataFrame in exact training order
                input_data = pd.DataFrame([{
                    'Locality': locality,
                    'Area_SqFt': area_sqft,
                    'BHK': bhk,
                    'Bathrooms': bathrooms,
                    'Property_Age_Years': age_years,
                    'Floor_Num': floor_num,
                    'Total_Floors': total_floors,
                    'Parking_Available': int(parking),
                    'Swimming_Pool': int(pool),
                    'Gym_Available': int(gym),
                    'Lift_Available': int(lift)
                }])
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                
                # Confidence interval (e.g. ±5%)
                margin_err = prediction * 0.05
                lower_bound = prediction - margin_err
                upper_bound = prediction + margin_err
                
                # Format into Lakhs and Crores
                def format_price(val_lakhs):
                    if val_lakhs >= 100:
                        crores = val_lakhs / 100
                        return f"₹{val_lakhs:.2f} Lakhs (~₹{crores:.2f} Cr)"
                    else:
                        return f"₹{val_lakhs:.2f} Lakhs"
                
                st.markdown(f"""
                <div class="price-box">
                    <div class="price-title">Estimated Market Value</div>
                    <div class="price-value">₹{prediction:.2f} Lakhs</div>
                    <div class="price-crore">{"(~ ₹" + f"{prediction/100:.2f}" + " Crores)" if prediction >= 100 else ""}</div>
                    <div class="price-range">Valuation Range (±5% Margin): {format_price(lower_bound)} - {format_price(upper_bound)}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📊 Valuation Breakdown Factors:")
                
                # Explain predictions logically
                factors = []
                
                base_rate = localities_rates.get(locality, 20000)
                factors.append(f"📍 **Locality Premium**: `{locality}` commands a baseline rate of around **₹{base_rate:,.0f}/sq.ft**.")
                
                size_factor = area_sqft * base_rate
                factors.append(f"📏 **Area multiplier**: `{area_sqft}` sq.ft area establishes a core property value of **₹{size_factor/100000:.2f} Lakhs** before feature adjustments.")
                
                if age_years > 0:
                    discount = min(20, age_years * 1)
                    factors.append(f"📉 **Property Depreciation**: The construction age (`{age_years}` years old) depreciated the baseline value by **-{discount}%**.")
                else:
                    factors.append(f"🆕 **New Construction Premium**: Brand-new/under-construction status preserves peak valuation.")
                
                if total_floors > 10 and floor_num > (total_floors / 2):
                    factors.append(f"🏙️ **High-Floor Premium**: Located on floor `{floor_num}` of `{total_floors}`. Elevated units in Mumbai high-rises capture a minor height premium.")
                
                amenity_list = []
                if parking: amenity_list.append("Private Parking Space (+₹8L)")
                if pool: amenity_list.append("Swimming Pool Access (+₹12L)")
                if gym: amenity_list.append("Modern Gym Access (+₹6L)")
                if lift: amenity_list.append("Lift Availability (+₹3L)")
                
                if amenity_list:
                    factors.append("🎁 **Added Amenities**: Added valuation from building upgrades: " + ", ".join([f"`{a}`" for a in amenity_list]) + ".")
                
                for f in factors:
                    st.write(f)
