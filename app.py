"""
Parametric Insurance Dashboard
- Upload GeoJSON with farm embeddings, VI timeseries, and similarity
- View any existing baseline triggers (if present)
- Download updated GeoJSON (preserves original + adds/updates baseline fields if inferred)
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from keplergl import KeplerGl
from streamlit_keplergl import keplergl_static
import io

# Page config
st.set_page_config(
    page_title="🛡️ Parametric Insurance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .status-good { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-danger { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'original_geojson' not in st.session_state:
    st.session_state.original_geojson = None
if 'enriched_geojson' not in st.session_state:
    st.session_state.enriched_geojson = None
if 'summary_df' not in st.session_state:
    st.session_state.summary_df = pd.DataFrame()
if 'baselines_loaded' not in st.session_state:
    st.session_state.baselines_loaded = False
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = pd.DataFrame()


def enrich_geojson_with_baselines(geojson_data):
    """
    Enrich each feature with baseline metadata if not present.
    For now, we do NOT infer thresholds — we only display what's there.
    But we prepare structure for future AI inference.
    """
    enriched_features = []
    summary_rows = []

    for feature in geojson_data['features']:
        props = feature.get('properties', {})
        farm_id = props.get('farm_id', 'UNKNOWN')

        # Extract optional baseline fields (may not exist)
        germ_threshold = props.get('germ_threshold')
        stress_threshold = props.get('stress_threshold')
        stress_drop = props.get('stress_drop')
        germ_conf = props.get('germ_confidence', 'unknown')
        stress_conf = props.get('stress_confidence', 'unknown')

        # Compute summary stats from VI_timeseries if available
        ndvi_mean, ndvi_p10 = 0.0, 0.0
        vi_timeseries = props.get('VI_timeseries', [])
        if vi_timeseries and isinstance(vi_timeseries, list):
            ndvi_vals = [pt.get('NDVI') for pt in vi_timeseries if pt.get('NDVI') is not None]
            if ndvi_vals:
                ndvi_mean = float(np.mean(ndvi_vals))
                ndvi_p10 = float(np.percentile(ndvi_vals, 10))

        # Build summary row (for analytics)
        summary_rows.append({
            'farm_id': farm_id,
            'name': props.get('name', f'Farm {farm_id}'),
            'crop': props.get('crop', 'Unknown'),
            'germ_threshold': germ_threshold if germ_threshold is not None else 0.0,
            'germ_confidence': germ_conf,
            'stress_threshold': stress_threshold if stress_threshold is not None else 0.0,
            'stress_drop': stress_drop if stress_drop is not None else 0.0,
            'stress_confidence': stress_conf,
            'similar_farms': len(props.get('top_k', [])),
            'ndvi_mean': ndvi_mean,
            'ndvi_p10': ndvi_p10,
            'has_germ': germ_threshold is not None,
            'has_stress': stress_threshold is not None,
        })

        # For now, we don't add new baseline fields unless they exist.
        # But we ensure consistent structure in enriched output.
        enriched_props = props.copy()
        # Optionally, you can add computed fields here in the future.

        enriched_features.append({
            "type": "Feature",
            "geometry": feature["geometry"],
            "properties": enriched_props
        })

    enriched_geojson = {
        "type": "FeatureCollection",
        "features": enriched_features
    }

    summary_df = pd.DataFrame(summary_rows)
    return enriched_geojson, summary_df


def create_kepler_config():
    return {
        "version": "v1",
        "config": {
            "visState": {
                "filters": [],
                "layers": [{
                    "id": "farms",
                    "type": "geojson",
                    "config": {
                        "dataId": "farms",
                        "label": "Farms",
                        "color": [18, 147, 154],
                        "columns": {"geojson": "_geojson"},
                        "isVisible": True,
                        "visConfig": {
                            "opacity": 0.8,
                            "strokeOpacity": 0.8,
                            "thickness": 2,
                            "strokeColor": [18, 147, 154],
                            "colorRange": {
                                "name": "Global Warming",
                                "type": "sequential",
                                "category": "Uber",
                                "colors": ["#5A1846", "#900C3F", "#C70039", "#E3611C", "#F1920E", "#FFC300"]
                            },
                            "stroked": True,
                            "filled": True,
                            "enable3d": False,
                            "wireframe": False
                        },
                        "textLabel": [{
                            "field": {"name": "farm_id", "type": "string"},
                            "color": [255, 255, 255],
                            "size": 10,
                            "offset": [0, 0],
                            "anchor": "middle",
                            "alignment": "center"
                        }]
                    }
                }],
                "interactionConfig": {
                    "tooltip": {
                        "fieldsToShow": {
                            "farms": [
                                {"name": "farm_id", "format": None},
                                {"name": "name", "format": None},
                                {"name": "crop", "format": None},
                                {"name": "germ_threshold", "format": ".3f"},
                                {"name": "stress_threshold", "format": ".3f"},
                                {"name": "ndvi_mean", "format": ".3f"}
                            ]
                        },
                        "enabled": True
                    }
                }
            },
            "mapState": {
                "bearing": 0,
                "dragRotate": False,
                "latitude": -1.3,
                "longitude": 36.8,
                "pitch": 0,
                "zoom": 10,
            },
            "mapStyle": {
                "styleType": "satellite",
                "visibleLayerGroups": {
                    "label": True, "road": True, "building": True, "water": True, "land": True
                }
            }
        }
    }


def plot_vi_timeseries(vi_data, farm_name):
    """Fixed: proper condition and handling"""
    if not vi_data:
        return None
    df = pd.DataFrame(vi_data)
    if 'date' not in df.columns:
        return None
    df['date'] = pd.to_datetime(df['date'])
    fig = go.Figure()
    indices = ['NDVI', 'EVI', 'NDMI', 'BSI']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    for idx, color in zip(indices, colors):
        if idx in df.columns:
            valid = df[df[idx].notnull()]
            if not valid.empty:
                fig.add_trace(go.Scatter(
                    x=valid['date'],
                    y=valid[idx],
                    mode='lines+markers',
                    name=idx,
                    line=dict(color=color)
                ))
    fig.update_layout(
        title=f"Vegetation & Soil Index - {farm_name}",
        xaxis_title="Date",
        yaxis_title="Index Value",
        hovermode='x unified',
        height=400
    )
    return fig


def plot_baseline_comparison(props, ndvi_mean, ndvi_p10):
    """Compare thresholds against historical VI (if available)"""
    fig = go.Figure()
    
    # If we had full distribution we'd show box, but we only have mean/p10
    # So we simulate a simple range
    ndvi_min = max(0.0, ndvi_mean - 0.3)
    ndvi_max = min(1.0, ndvi_mean + 0.3)
    fig.add_trace(go.Scatter(
        x=['Historical Range'],
        y=[(ndvi_min + ndvi_max) / 2],
        error_y=dict(type='data', array=[(ndvi_max - ndvi_min) / 2]),
        mode='markers',
        name='NDVI Range'
    ))

    # Add thresholds if present
    if props.get('germ_threshold') is not None:
        fig.add_hline(
            y=props['germ_threshold'],
            line_dash="dash",
            line_color="red",
            annotation_text="Germination Threshold"
        )
    if props.get('stress_threshold') is not None:
        fig.add_hline(
            y=props['stress_threshold'],
            line_dash="dash",
            line_color="orange",
            annotation_text="Stress Threshold"
        )

    fig.update_layout(
        title="Thresholds vs Historical NDVI",
        yaxis_title="NDVI",
        height=400,
        yaxis_range=[0, 1]
    )
    return fig


def display_baseline_details(props, farm_name):
    st.subheader(f"📋 Insurance Triggers for {farm_name}")
    
    has_germ = 'germ_threshold' in props
    has_stress = 'stress_threshold' in props

    if not (has_germ or has_stress):
        st.info("No parametric triggers defined for this farm.")
        return

    col1, col2 = st.columns(2)

    with col1:
        if has_germ:
            st.markdown("### 🌱 Germination Insurance")
            st.metric("NDVI Threshold", f"{props['germ_threshold']:.3f}")
            conf = props.get('germ_confidence', 'unknown')
            cls = 'status-good' if conf == 'high' else 'status-warning' if conf == 'medium' else 'status-danger'
            st.markdown(f"<p class='{cls}'>Confidence: {conf.upper()}</p>", unsafe_allow_html=True)
            if 'germ_rationale' in props:
                st.caption(props['germ_rationale'])
        else:
            st.markdown("### 🌱 Germination Insurance")
            st.warning("Not configured")

    with col2:
        if has_stress:
            st.markdown("### 🌾 Crop Stress Insurance")
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("NDVI Threshold", f"{props['stress_threshold']:.3f}")
            with col2b:
                drop = props.get('stress_drop', 'N/A')
                st.metric("Drop Threshold", f"{drop:.3f}" if isinstance(drop, (int, float)) else str(drop))
            conf = props.get('stress_confidence', 'unknown')
            cls = 'status-good' if conf == 'high' else 'status-warning' if conf == 'medium' else 'status-danger'
            st.markdown(f"<p class='{cls}'>Confidence: {conf.upper()}</p>", unsafe_allow_html=True)
            if 'basis_risk' in props:
                st.success(f"**Basis Risk:** {props['basis_risk']}")
        else:
            st.markdown("### 🌾 Crop Stress Insurance")
            st.warning("Not configured")


# Main App
def main():
    st.markdown("<h1 class='main-header'>🛡️ Parametric Insurance Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("*Upload your farm GeoJSON with embeddings, VI timeseries, and similarity*")

    with st.sidebar:
        st.header("📂 Data")

        geojson_file = st.file_uploader("Upload Farm GeoJSON", type=["geojson"])
        
        if geojson_file and st.button("🔄 Load Data"):
            try:
                with st.spinner("Parsing GeoJSON..."):
                    stringio = io.StringIO(geojson_file.getvalue().decode("utf-8"))
                    original_geojson = json.load(stringio)
                    
                    if original_geojson.get("type") != "FeatureCollection":
                        st.error("GeoJSON must be a FeatureCollection.")
                        return

                    enriched_geojson, summary_df = enrich_geojson_with_baselines(original_geojson)

                    st.session_state.original_geojson = original_geojson
                    st.session_state.enriched_geojson = enriched_geojson
                    st.session_state.summary_df = summary_df
                    st.session_state.baselines_loaded = True
                    st.success(f"✅ Loaded {len(summary_df)} farms")
            except Exception as e:
                st.error(f"Error loading file: {e}")

        st.markdown("---")

        # Export updated GeoJSON
        if st.session_state.baselines_loaded:
            st.subheader("📥 Export")
            geojson_str = json.dumps(st.session_state.enriched_geojson, indent=2)
            st.download_button(
                label="⬇️ Download Updated GeoJSON",
                data=geojson_str,
                file_name="farms_with_analysis.geojson",
                mime="application/geo+json"
            )

            # Filters
            st.markdown("---")
            st.subheader("🔍 Filters")
            crops = st.session_state.summary_df['crop'].dropna().unique()
            selected_crops = st.multiselect(
                "Crop Type", 
                crops, 
                default=crops.tolist() if len(crops) <= 10 else crops[:5].tolist()
            )
            conf_levels = ['high', 'medium', 'low', 'unknown']
            selected_conf = st.multiselect(
                "Stress Confidence", 
                conf_levels, 
                default=conf_levels
            )
            
            filtered = st.session_state.summary_df.copy()
            if selected_crops:
                filtered = filtered[filtered['crop'].isin(selected_crops)]
            if selected_conf:
                filtered = filtered[filtered['stress_confidence'].isin(selected_conf)]
            st.session_state.filtered_df = filtered
            st.info(f"Showing {len(filtered)} / {len(st.session_state.summary_df)} farms")

    if not st.session_state.baselines_loaded:
        st.info("👈 Upload your GeoJSON file to begin.")
        st.markdown("### 📌 Your GeoJSON Should Include")
        st.markdown("""
        - `farm_id` in properties
        - Optional: `germ_threshold`, `stress_threshold`, `stress_drop`
        - `VI_timeseries`: list of `{date, NDVI, EVI, NDMI, BSI}`
        - `top_k`: list of similar farms
        - Embedding fields like `A00`–`A63` (ignored but preserved)
        """)
        return

    # Metrics
    df = st.session_state.summary_df
    st.subheader("📊 Portfolio Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Farms", len(df))
    with col2: st.metric("With Germination Trigger", df['has_germ'].sum())
    with col3: st.metric("With Stress Trigger", df['has_stress'].sum())
    with col4: st.metric("Crop Types", df['crop'].nunique())

    tab1, tab2, tab3 = st.tabs(["🗺️ Map", "📈 Analytics", "🔍 Farm Details"])

    with tab1:
        st.subheader("Farm Map")
        map_config = create_kepler_config()
        kepler_map = KeplerGl(height=600, config=map_config)
        kepler_map.add_data(data=st.session_state.enriched_geojson, name='farms')
        keplergl_static(kepler_map, height=600)

    with tab2:
        fd = st.session_state.filtered_df
        if fd.empty:
            st.warning("No farms match the current filters.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(
                    fd[fd['stress_threshold'] > 0],
                    x='stress_threshold',
                    color='crop',
                    nbins=20,
                    title="Stress Threshold Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                conf_df = fd.groupby(['crop', 'stress_confidence']).size().reset_index(name='count')
                fig = px.bar(
                    conf_df,
                    x='crop',
                    y='count',
                    color='stress_confidence',
                    barmode='group',
                    title="Confidence by Crop"
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fd = st.session_state.filtered_df
        if fd.empty:
            st.warning("No farms match filters.")
            return

        farm_options = fd.apply(
            lambda r: f"{r['farm_id']} - {r.get('name', 'Unnamed')} ({r['crop']})", 
            axis=1
        ).tolist()
        selected = st.selectbox("Select Farm", farm_options)
        farm_id = selected.split(' - ')[0]

        # Find feature
        selected_feature = None
        for feat in st.session_state.enriched_geojson['features']:
            if feat['properties'].get('farm_id') == farm_id:
                selected_feature = feat
                break

        if not selected_feature:
            st.error("Farm not found.")
            return

        props = selected_feature['properties']
        farm_name = props.get('name', farm_id)

        display_baseline_details(props, farm_name)

        # Interactive map with selected farm and top-k
        st.subheader("🗺️ Farm Location & Similar Farms")
        top_k = props.get('top_k', [])
        
        if top_k:
            # Create filtered GeoJSON with selected farm + top-k
            top_k_ids = [item['target'] for item in top_k[:10]]
            filtered_features = []
            
            for feat in st.session_state.enriched_geojson['features']:
                fid = feat['properties'].get('farm_id')
                if fid == farm_id:
                    # Mark as selected farm
                    feat_copy = json.loads(json.dumps(feat))
                    feat_copy['properties']['map_category'] = 'Selected Farm'
                    feat_copy['properties']['similarity'] = 1.0
                    filtered_features.append(feat_copy)
                elif fid in top_k_ids:
                    # Mark as similar farm with similarity score
                    feat_copy = json.loads(json.dumps(feat))
                    feat_copy['properties']['map_category'] = 'Similar Farm'
                    sim_score = next((item['similarity'] for item in top_k if item['target'] == fid), 0.0)
                    feat_copy['properties']['similarity'] = sim_score
                    filtered_features.append(feat_copy)
            
            detail_geojson = {
                "type": "FeatureCollection",
                "features": filtered_features
            }
            
            # Create Kepler config for detail view
            detail_config = {
                "version": "v1",
                "config": {
                    "visState": {
                        "filters": [],
                        "layers": [{
                            "id": "detail_farms",
                            "type": "geojson",
                            "config": {
                                "dataId": "detail_farms",
                                "label": "Farms",
                                "color": [18, 147, 154],
                                "columns": {"geojson": "_geojson"},
                                "isVisible": True,
                                "visConfig": {
                                    "opacity": 0.8,
                                    "strokeOpacity": 1,
                                    "thickness": 3,
                                    "colorRange": {
                                        "name": "Custom",
                                        "type": "diverging",
                                        "category": "Custom",
                                        "colors": ["#E3611C", "#FFC300", "#2ecc71"]
                                    },
                                    "strokeColorField": {"name": "map_category", "type": "string"},
                                    "strokeColorScale": "ordinal",
                                    "stroked": True,
                                    "filled": True,
                                    "enable3d": False,
                                    "wireframe": False
                                },
                                "textLabel": [{
                                    "field": {"name": "farm_id", "type": "string"},
                                    "color": [255, 255, 255],
                                    "size": 12,
                                    "offset": [0, 0],
                                    "anchor": "middle",
                                    "alignment": "center"
                                }]
                            }
                        }],
                        "interactionConfig": {
                            "tooltip": {
                                "fieldsToShow": {
                                    "detail_farms": [
                                        {"name": "farm_id", "format": None},
                                        {"name": "name", "format": None},
                                        {"name": "map_category", "format": None},
                                        {"name": "similarity", "format": ".3f"},
                                        {"name": "crop", "format": None}
                                    ]
                                },
                                "enabled": True
                            }
                        }
                    },
                    "mapState": {
                        "bearing": 0,
                        "dragRotate": False,
                        "latitude": -1.3,
                        "longitude": 36.8,
                        "pitch": 0,
                        "zoom": 11,
                    },
                    "mapStyle": {
                        "styleType": "satellite",
                        "visibleLayerGroups": {
                            "label": True, "road": True, "building": True, "water": True, "land": True
                        }
                    }
                }
            }
            
            detail_map = KeplerGl(height=500, config=detail_config)
            detail_map.add_data(data=detail_geojson, name='detail_farms')
            keplergl_static(detail_map, height=500)
            
            # Show similarity table with interactive selection
            st.subheader("🔗 Similar Farms Ranked by Similarity")
            sim_data = []
            for item in top_k[:10]:
                target_id = item['target']
                target_feat = next((f for f in st.session_state.enriched_geojson['features'] 
                                   if f['properties'].get('farm_id') == target_id), None)
                if target_feat:
                    target_props = target_feat['properties']
                    sim_data.append({
                        "Rank": len(sim_data) + 1,
                        "Farm ID": target_id,
                        "Name": target_props.get('name', 'Unnamed'),
                        "Crop": target_props.get('crop', 'Unknown'),
                        "Similarity": f"{item['similarity']:.3f}",
                        "Has Stress Trigger": "✓" if target_props.get('stress_threshold') else "✗",
                        "Has Germ Trigger": "✓" if target_props.get('germ_threshold') else "✗"
                    })
            
            if sim_data:
                st.dataframe(sim_data, use_container_width=True, hide_index=True)
        else:
            st.info("No similar farms data available for this farm.")

        # Vegetation indices charts
        st.subheader("📈 Vegetation Index Trends")
        col1, col2 = st.columns(2)
        with col1:
            vi_data = props.get('VI_timeseries', [])
            fig = plot_vi_timeseries(vi_data, farm_name)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Get NDVI stats for plot
            ndvi_vals = [pt['NDVI'] for pt in vi_data if 'NDVI' in pt and pt['NDVI'] is not None]
            ndvi_mean = float(np.mean(ndvi_vals)) if ndvi_vals else 0.0
            ndvi_p10 = float(np.percentile(ndvi_vals, 10)) if len(ndvi_vals) > 1 else ndvi_mean
            fig = plot_baseline_comparison(props, ndvi_mean, ndvi_p10)
            if fig:
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
