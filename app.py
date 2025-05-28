# Enhanced Solar Analysis Streamlit App with Updated Detector Integration

import streamlit as st
from detector import SolarAnalysisModel, AnalysisResult
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import logging
import time
from typing import Optional

# Configure environment and logging
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["STREAMLIT_PATHS_NO_WATCH"] = "torch"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Solar AI Assistant Pro",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF6B35;
        margin: 0.5rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .api-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .api-success { background: #d4edda; color: #155724; }
    .api-error { background: #f8d7da; color: #721c24; }
    .api-warning { background: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize all session state variables."""
    default_states = {
        "analysis_complete": False,
        "analysis_result": None,
        "solar_calculations": None,
        "processed_image": None,
        "processing": False,
        "api_key_set": False,
        "api_key": None,
        "api_test_result": None
    }
    
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def display_confidence_indicator(confidence: float) -> str:
    """Return CSS class based on confidence level."""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"

def display_validation_warnings(analysis_result: AnalysisResult):
    """Display validation warnings and errors."""
    if analysis_result.validation_errors:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("⚠️ Analysis Warnings Detected")
        for error in analysis_result.validation_errors:
            st.write(f"• {error}")
        st.markdown('</div>', unsafe_allow_html=True)

def create_confidence_chart(confidence: float, source: str):
    """Create a confidence chart using plotly."""
    confidence_pct = confidence * 100
    
    # Determine color based on confidence level
    if confidence >= 0.7:
        color = "#28a745"  # Green
    elif confidence >= 0.4:
        color = "#ffc107"  # Yellow
    else:
        color = "#dc3545"  # Red
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence_pct,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence ({source.upper()})"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 70], 'color': "#fff3cd"},
                {'range': [70, 100], 'color': "#d4edda"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75, 
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_system_comparison_chart(solar_calculations: dict):
    """Create comparison chart for different panel types."""
    panel_types = ["standard", "premium", "compact"]
    
    # Get base calculations
    current_specs = solar_calculations["system_specifications"]
    base_area = 240  # m2 default usable area
    
    comparison_data = []
    
    for panel_type in panel_types:
        # Panel specifications from detector.py
        panel_specs = {
            "standard": {"area_m2": 1.95, "wattage": 350, "cost_per_w": 2.5},
            "premium": {"area_m2": 1.95, "wattage": 400, "cost_per_w": 3.0},
            "compact": {"area_m2": 1.65, "wattage": 300, "cost_per_w": 2.8}
        }
        
        specs = panel_specs[panel_type]
        panel_count = int((base_area * 0.75) / specs["area_m2"])
        system_kw = panel_count * specs["wattage"] / 1000
        annual_kwh = system_kw * 1500  # Average solar production
        cost = system_kw * 1000 * specs["cost_per_w"]
        
        comparison_data.append({
            "Panel Type": panel_type.title(),
            "System Size (kW)": round(system_kw, 1),
            "Annual Production (kWh)": round(annual_kwh),
            "System Cost ($)": round(cost),
            "Cost per kW ($)": round(cost / max(system_kw, 0.1))
        })
    
    df = pd.DataFrame(comparison_data)
    
    fig = px.bar(
        df, 
        x="Panel Type", 
        y=["System Size (kW)", "Annual Production (kWh)"],
        title="Panel Type Comparison",
        barmode='group',
        color_discrete_sequence=['#FF6B35', '#F7931E']
    )
    
    fig.update_layout(height=400)
    return fig, df

def test_api_connection(api_key: str):
    """Test API connection and store result in session state."""
    if api_key and api_key != st.session_state.get("last_tested_key"):
        with st.spinner("Testing API connection..."):
            model = SolarAnalysisModel(api_key)
            test_result = model.test_api_connection()
            st.session_state.api_test_result = test_result
            st.session_state.last_tested_key = api_key
            return test_result
    return st.session_state.get("api_test_result")

def display_api_status(test_result: dict):
    """Display API connection status."""
    if not test_result:
        return
        
    if test_result["status"] == "success":
        st.markdown(
            f'<div class="api-status api-success">✅ API Connection: {test_result["message"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="api-status api-error">❌ API Connection: {test_result["message"]}</div>',
            unsafe_allow_html=True
        )
        
        # Show more details in expander
        with st.expander("🔍 API Error Details"):
            st.json(test_result)

def main():
    """Main application function."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header section
    st.markdown("""
    <div class="main-header">
        <h1 style="color: black; margin: 0;">☀️ Solar AI Assistant Pro</h1>
        <p style="color: black; margin: 0; opacity: 0.9;">Advanced Rooftop Analysis with AI-Powered Validation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input with testing
        st.subheader("API Configuration")
        api_key = st.text_input(
            "OpenRouter API Key (Optional)", 
            type="password",
            help="Enter your OpenRouter API key for enhanced AI analysis. Leave blank to use fallback analysis.",
            placeholder="sk-or-v1-..."
        )
        
        # Test API connection when key is provided
        if api_key:
            test_result = test_api_connection(api_key)
            display_api_status(test_result)
            
            if test_result and test_result["status"] == "success":
                st.session_state.api_key_set = True
                st.session_state.api_key = api_key
            else:
                st.session_state.api_key_set = False
        else:
            st.session_state.api_key_set = False
            st.info("💡 Using fallback analysis mode")
        
        st.divider()
        
        # Analysis parameters
        st.subheader("📊 Analysis Parameters")
        
        panel_type = st.selectbox(
            "Panel Type",
            ["standard", "premium", "compact"],
            help="Select solar panel type for calculations"
        )
        
        region = st.selectbox(
            "Region",
            ["usa", "europe", "australia", "india"],
            help="Select your geographic region for solar irradiance calculations"
        )
        
        st.divider()
        
        # System information
        st.subheader("ℹ️ System Info")
        st.info("""
        **Enhanced Features:**
        - 🤖 AI-powered roof analysis
        - 📊 Confidence scoring
        - ✅ Validation checks
        - 🔄 Multiple panel comparisons
        - 💰 Financial projections
        - 🎯 Enhanced error handling
        """)
        
        # API Models info
        if st.session_state.api_key_set:
            with st.expander("🔍 AI Models Used"):
                st.write("""
                **Primary Models (in order):**
                - OpenAI GPT-4o
                - OpenAI GPT-4o Mini
                - Anthropic Claude-3 Sonnet
                - Google Gemini Pro Vision
                
                **Fallback:** Rule-based analysis
                """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📤 Upload Rooftop Image")
        st.write("Upload a clear satellite or aerial view of the rooftop for analysis.")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG. Recommended: High-resolution top-down view."
        )
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="📸 Uploaded Rooftop Image", use_container_width=True)
                
                # Analysis button
                col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                with col_btn2:
                    analyze_button = st.button(
                        "🚀 Run Solar Analysis", 
                        type="primary", 
                        use_container_width=True,
                        disabled=st.session_state.processing
                    )
                    
                    if analyze_button:
                        st.session_state.processing = True
                        st.rerun()
                        
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
                logger.error(f"Image loading error: {e}")
    
    with col2:
        if uploaded_file and not st.session_state.analysis_complete:
            st.header("📋 Analysis Checklist")
            st.write("✅ Image uploaded successfully")
            st.write("⏳ Ready for analysis")
            
            if st.session_state.api_key_set:
                test_result = st.session_state.get("api_test_result")
                if test_result and test_result["status"] == "success":
                    st.write("✅ AI analysis enabled")
                else:
                    st.write("⚠️ API issues detected")
            else:
                st.write("⚠️ Fallback mode (no API key)")
        
        elif st.session_state.analysis_complete:
            st.header("📈 Quick Stats")
            if st.session_state.analysis_result and st.session_state.solar_calculations:
                specs = st.session_state.solar_calculations["system_specifications"]
                st.metric("System Size", f"{specs['system_size_kw']} kW")
                st.metric("Panel Count", specs['panel_count'])
                st.metric("Confidence", f"{st.session_state.analysis_result.confidence_score:.1%}")
    
    # Processing section
    if st.session_state.processing and uploaded_file:
        process_analysis(image, api_key, panel_type, region)
    
    # Results section
    if st.session_state.analysis_complete:
        display_results()

def process_analysis(image: Image.Image, api_key: str, panel_type: str, region: str):
    """Process the solar analysis with comprehensive error handling."""
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Use API key from session state if not provided
        if not api_key and st.session_state.get('api_key'):
            api_key = st.session_state.api_key

        status_text.text("🔧 Initializing analysis model...")
        progress_bar.progress(10)

        # Initialize model with API key
        model = SolarAnalysisModel(openrouter_api_key=api_key if api_key else None)

        status_text.text("🖼️ Processing image...")
        progress_bar.progress(30)

        # Convert image to numpy array and encode
        image_array = np.array(image)
        image_base64 = model.encode_image_to_base64(image_array)

        status_text.text("🤖 Running AI analysis...")
        progress_bar.progress(60)

        # Analyze with the updated method that returns (AnalysisResult, used_fallback)
        analysis_result, used_fallback = model.analyze_with_openrouter(image_base64)

        # Display appropriate messages based on analysis source
        if used_fallback:
            st.warning("🔄 AI analysis unavailable. Using fallback analysis.")
            if api_key:
                st.info("💡 Check API key or try again later for AI-powered analysis.")
        else:
            if analysis_result.source == 'ai':
                st.success("🤖 AI analysis completed successfully!")
            else:
                st.info(f"📊 Analysis completed using {analysis_result.source} method")

        status_text.text("⚡ Calculating solar potential...")
        progress_bar.progress(80)

        # Calculate solar system specifications
        solar_calculations = model.calculate_solar_system(analysis_result, panel_type, region)

        status_text.text("🎨 Generating visualization...")
        progress_bar.progress(90)

        # Generate panel layout visualization
        processed_image = model.generate_panel_layout_visualization(
            image_array, analysis_result, solar_calculations
        )

        # Store results in session state
        st.session_state.analysis_result = analysis_result
        st.session_state.solar_calculations = solar_calculations
        st.session_state.processed_image = processed_image
        st.session_state.analysis_complete = True
        st.session_state.processing = False

        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")

        # Show success message with analysis details
        col1, col2 = st.columns(2)
        with col1:
            st.success("🎉 Solar analysis completed successfully!")
        with col2:
            confidence_pct = analysis_result.confidence_score * 100
            if confidence_pct > 70:
                st.success(f"🎯 High confidence: {confidence_pct:.1f}%")
            elif confidence_pct > 40:
                st.warning(f"⚠️ Medium confidence: {confidence_pct:.1f}%")
            else:
                st.error(f"⚠️ Low confidence: {confidence_pct:.1f}%")

        # Auto-refresh to show results
        time.sleep(1)
        st.rerun()

    except Exception as e:
        st.session_state.processing = False
        logger.error(f"Analysis failed: {e}")

        # Show detailed error information
        st.error(f"❌ Analysis failed: {str(e)}")
        
        # Add expandable error details for debugging
        with st.expander("🔍 Error Details (for debugging)"):
            st.code(f"""
Error Type: {type(e).__name__}
Error Message: {str(e)}
API Key Present: {bool(api_key)}
Analysis Mode: {'AI' if api_key else 'Fallback'}
            """)
            
        progress_bar.empty()
        status_text.empty()

        # Offer suggestions
        st.info("💡 **Troubleshooting suggestions:**")
        st.write("• Check your internet connection")
        st.write("• Verify API key is correct")
        st.write("• Try with a different image")
        st.write("• Use fallback mode (remove API key)")

def display_results():
    """Display comprehensive analysis results."""
    
    analysis_result = st.session_state.analysis_result
    solar_calculations = st.session_state.solar_calculations
    processed_image = st.session_state.processed_image
    
    st.header("📊 Analysis Results")
    
    # Confidence and validation section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        confidence = analysis_result.confidence_score
        confidence_class = display_confidence_indicator(confidence)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Analysis Quality</h4>
            <p class="{confidence_class}">Confidence: {confidence:.1%}</p>
            <p><strong>Source:</strong> {analysis_result.source.upper()}</p>
            <p><strong>Validation:</strong> {len(analysis_result.validation_errors)} warnings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.plotly_chart(
            create_confidence_chart(confidence, analysis_result.source),
            use_container_width=True
        )
    
    # Display validation warnings
    display_validation_warnings(analysis_result)
    
    # Main results layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🏠 Rooftop Analysis")
        roof_data = analysis_result.data.get("roof_dimensions", {})
        
        st.markdown(f"""
        <div class="metric-card" style="color: black !important;">
            <h5 style="color: black !important;">📐 Roof Dimensions:</h5>
            • <strong>Total Area:</strong> {roof_data.get('total_area_m2', 'N/A')} m²<br>
            • <strong>Usable Area:</strong> {roof_data.get('usable_area_m2', 'N/A')} m²<br>
            • <strong>Shape:</strong> {roof_data.get('roof_shape', 'N/A')}<br>
            • <strong>Orientation:</strong> {roof_data.get('orientation', 'N/A')}<br>
            • <strong>Tilt:</strong> {roof_data.get('tilt_angle_degrees', 'N/A')}°
        </div>
        """, unsafe_allow_html=True)
        
        # Obstacles information
        obstacles = analysis_result.data.get("obstacles_and_shading", {})
        if obstacles:
            st.markdown(f"""
                <div class="metric-card" style="color:black;">
                <h5>🌳 Shading & Obstacles:</h5>
                • <strong>Tree Shading:</strong> {obstacles.get('tree_shading_percentage', 0)}%<br>
                • <strong>Chimneys:</strong> {obstacles.get('chimney_count', 0)}<br>
                • <strong>Vents:</strong> {obstacles.get('vent_count', 0)}
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("⚡ Solar System Design")
        
        if processed_image is not None:
            st.image(
                processed_image,
                caption="🔋 Proposed Panel Layout",
                use_container_width=True
            )
        else:
            st.warning("Panel layout visualization not available")
    
    # System specifications
    st.subheader("📋 System Specifications & Financial Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    specs = solar_calculations["system_specifications"]
    financial = solar_calculations["financial_analysis"]
    
    with col1:
        st.metric(
            "⚡ System Size",
            f"{specs['system_size_kw']} kW",
            f"{specs['panel_count']} panels"
        )
    
    with col2:
        st.metric(
            "🔋 Annual Production",
            f"{specs['annual_production_kwh']:,} kWh",
            f"{specs.get('panel_efficiency', 0.2)*100:.0f}% efficiency"
        )
    
    with col3:
        st.metric(
            "💰 Annual Savings",
            f"${financial['annual_electricity_savings']:,}",
            "Estimated"
        )
    
    with col4:
        st.metric(
            "💵 System Cost",
            f"${financial.get('system_cost_estimate', 0):,}",
            f"{financial.get('payback_period_years', 0)} yr payback"
        )
    
    # Panel comparison section
    st.subheader("📊 Panel Type Comparison")
    
    try:
        comparison_fig, comparison_df = create_system_comparison_chart(solar_calculations)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(comparison_fig, use_container_width=True)
        with col2:
            st.dataframe(comparison_df, use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate comparison chart: {e}")
    
    # Export results section
    st.subheader("📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Prepare JSON export
        export_data = {
            "analysis_metadata": {
                "confidence_score": analysis_result.confidence_score,
                "analysis_source": analysis_result.source,
                "validation_errors": analysis_result.validation_errors,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "roof_analysis": analysis_result.data,
            "solar_calculations": solar_calculations
        }
        
        st.download_button(
            "📄 Download Full Report (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name=f"solar_analysis_report_{int(time.time())}.json",
            mime="application/json"
        )
    
    with col2:
        # Prepare CSV export
        summary_data = {
            "Metric": [
                "System Size (kW)", "Panel Count", "Annual Production (kWh)", 
                "Annual Savings ($)", "System Cost ($)", "Confidence Score (%)"
            ],
            "Value": [
                specs['system_size_kw'], specs['panel_count'], 
                specs['annual_production_kwh'], financial['annual_electricity_savings'],
                financial.get('system_cost_estimate', 0),
                f"{analysis_result.confidence_score:.1%}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        st.download_button(
            "📊 Download Summary (CSV)",
            data=summary_df.to_csv(index=False),
            file_name=f"solar_analysis_summary_{int(time.time())}.csv",
            mime="text/csv"
        )
    
    with col3:
        # Reset button
        if st.button("🔄 Analyze New Image", type="secondary"):
            # Clear session state
            keys_to_clear = [
                "analysis_complete", "analysis_result", "solar_calculations", 
                "processed_image", "api_test_result", "last_tested_key"
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # Additional insights section
    if analysis_result.confidence_score > 0.7:
        st.subheader("💡 Recommendations")
        st.success("""
        **High Confidence Analysis - Recommended Actions:**
        - ✅ Proceed with detailed engineering assessment
        - ✅ Contact solar installers for quotes
        - ✅ Consider financing options
        - ✅ Check local permits and incentives
        """)
    elif analysis_result.confidence_score > 0.4:
        st.subheader("💡 Recommendations")
        st.warning("""
        **Medium Confidence Analysis - Suggested Next Steps:**
        - 🔍 Consider higher resolution imagery
        - 🤝 Schedule professional site assessment
        - 📞 Consult with local solar experts
        - 📋 Verify measurements on-site
        """)
    else:
        st.subheader("💡 Recommendations")
        st.error("""
        **Low Confidence Analysis - Important Notes:**
        - ⚠️ Results may not be reliable
        - 🏠 Professional site visit strongly recommended
        - 📸 Try uploading a clearer image
        - 🔄 Consider using AI analysis mode
        """)

if __name__ == "__main__":
    main()