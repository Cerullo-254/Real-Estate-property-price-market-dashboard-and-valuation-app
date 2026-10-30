"""
Property Market Dashboard & Valuation App
-----------------------------------------
Every chart in this app is a PNG that was saved from the Colab notebook.
The app only loads those images - it does not redraw them.

Folder layout expected next to this file:
    assets/          <- chart PNGs saved from the notebook
    model_assets/    <- rf_model.joblib, xgb_model.json, ensemble.json,
                        model_columns.json, summary.json
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
ASSETS = BASE / "assets"
MODEL_DIR = BASE / "model_assets"

st.set_page_config(
    page_title="Property Market Dashboard & Valuation",
    page_icon="🏠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Numbers taken from the notebook outputs. If model_assets/summary.json exists
# (created by the notebook export cell) it overrides these.
# ---------------------------------------------------------------------------
DEFAULT_SUMMARY = {
    "n_properties": 7000,
    "mean_price": 21944620.0,
    "median_price": 19157037.0,
    "min_price": 1500000.0,
    "max_price": 144829423.0,
    "avg_price_by_location_m": {
        "Westlands": 31.8, "Runda": 29.6, "Karen": 28.5, "Kilimani": 27.4,
        "Kileleshwa": 26.9, "Ruaka": 18.8, "Thika Road": 16.4,
        "Syokimau": 14.7, "Mombasa": 14.1, "Kitengela": 11.6,
    },
    "avg_price_by_type_m": {
        "Townhouse": 22.2, "Maisonette": 22.0, "Apartment": 21.9,
        "Bungalow": 21.8, "Villa": 21.8,
    },
    "avg_price_per_sqm_by_location": {
        "Westlands": 198014.47, "Runda": 181774.74, "Karen": 172599.38,
        "Kilimani": 167247.42, "Kileleshwa": 163102.24, "Ruaka": 117138.14,
        "Thika Road": 102951.57, "Syokimau": 91148.17, "Mombasa": 87992.70,
        "Kitengela": 73226.27,
    },
    "metrics": {
        "Random Forest": {"r2": 0.895032, "rmse": 2750484.8, "mae": 2162962.4},
        "XGBoost": {"r2": 0.898262, "rmse": 2701835.95, "mae": 2098387.3},
        "Voting Regressor (final)": {"r2": 0.899019, "rmse": 2649830.8, "mae": 2085694.5},
    },
    "final_model": "Voting Regressor (final)",
    "locations": ["Karen", "Kileleshwa", "Kilimani", "Kitengela", "Mombasa",
                  "Ruaka", "Runda", "Syokimau", "Thika Road", "Westlands"],
    "property_types": ["Apartment", "Bungalow", "Maisonette", "Townhouse", "Villa"],
    "plot_sizes": [0, 100, 200, 300, 500, 1000],
    "ranges": {
        "Bedrooms": [1, 6], "Bathrooms": [1, 5], "Floor_Area_Sqm": [33.6, 1000.0],
        "Age_Years": [0, 34], "Parking_Spaces": [0, 4],
        "Distance_CBD_KM": [2, 35], "Amenities_Score": [1, 10],
    },
}


@st.cache_data
def load_summary():
    summary = dict(DEFAULT_SUMMARY)
    path = MODEL_DIR / "summary.json"
    if path.exists():
        summary.update(json.loads(path.read_text()))
    return summary


class VotingModel:
    """The final 'Voting model': it averages the Random Forest and XGBoost estimates.

    This is exactly what scikit-learn's VotingRegressor does. It is rebuilt here from the two
    saved parts so that the XGBoost part can be stored in XGBoost's own portable file format
    (xgb_model.json). That format opens on any computer and any XGBoost version, which avoids
    the 'input stream corrupted' error that happens when a pickled XGBoost model is opened on a
    machine with a different XGBoost version.
    """

    def __init__(self, rf, xgb, weights):
        self.rf, self.xgb = rf, xgb
        w = np.array(weights, dtype=float)
        self.weights = w / w.sum()

    def predict(self, X):
        return self.weights[0] * self.rf.predict(X) + self.weights[1] * self.xgb.predict(X)


@st.cache_resource(show_spinner="Loading the valuation model...")
def load_model():
    """Returns (model, columns, error_message). model is None if it could not be loaded."""
    cols_path = MODEL_DIR / "model_columns.json"
    rf_path = MODEL_DIR / "rf_model.joblib"
    xgb_path = MODEL_DIR / "xgb_model.json"
    ens_path = MODEL_DIR / "ensemble.json"
    if not (rf_path.exists() and xgb_path.exists() and cols_path.exists()):
        return None, None, "missing"
    try:
        import xgboost as xgb

        rf = joblib.load(rf_path)
        booster = xgb.XGBRegressor()
        booster.load_model(str(xgb_path))
        weights = [1, 1]
        if ens_path.exists():
            weights = json.loads(ens_path.read_text()).get("weights") or [1, 1]
        columns = json.loads(cols_path.read_text())
        return VotingModel(rf, booster, weights), columns, None
    except Exception as e:  # show a readable message instead of a crash
        return None, None, f"{type(e).__name__}: {e}"


S = load_summary()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def kes(x):
    return f"KES {x:,.0f}"


def kes_m(x):
    return f"KES {x / 1e6:,.1f}M"


def show_image(names):
    """Show the first PNG in `names` that exists in assets/."""
    for name in names:
        path = ASSETS / f"{name}.png"
        if path.exists():
            st.image(str(path), width="stretch")
            return
    st.info(
        f"Chart not found: assets/{names[0]}.png - "
        "run the export cell in the notebook and copy the files into this app's assets/ folder."
    )


def explain(what, means):
    """Plain-language explanation shown under every chart."""
    with st.container(border=True):
        st.markdown(f"**What you are looking at:** {what}")
        st.markdown(f"**What it means for you:** {means}")


def build_features(df_inputs, columns):
    """Turn user inputs into the exact columns the model was trained on."""
    X = pd.DataFrame(0, index=range(len(df_inputs)), columns=columns, dtype=float)
    for col in ["Bedrooms", "Bathrooms", "Floor_Area_Sqm", "Plot_Size_Sqm", "Age_Years",
                "Parking_Spaces", "Distance_CBD_KM", "Amenities_Score"]:
        if col in X.columns:
            X[col] = df_inputs[col].values
    for i, (loc, ptype) in enumerate(zip(df_inputs["Location"], df_inputs["Property_Type"])):
        for col in (f"Location_{loc}", f"Property_Type_{ptype}"):
            if col in X.columns:
                X.loc[i, col] = 1
    return X


def predict_prices(model, columns, df_inputs):
    # The model works on the logarithm of the price, so convert back with exp().
    return np.exp(model.predict(build_features(df_inputs, columns)))


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏠 Property Market Dashboard & Valuation")
st.caption(
    f"Based on {S['n_properties']:,} property listings. "
    "Prices are in Kenya Shillings (KES). 1M means 1 million."
)

tab_dash, tab_value, tab_help = st.tabs(
    ["📊 Market dashboard", "💰 Value a property", "ℹ️ How to use / About"]
)

# ===========================================================================
# TAB 1 - DASHBOARD
# ===========================================================================
with tab_dash:
    final = S["metrics"][S["final_model"]]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average price", kes_m(S["mean_price"]),
              help="All prices added up and divided by the number of properties.")
    c2.metric("Median price", kes_m(S["median_price"]),
              help="The middle price: half of all properties cost less than this, half cost more.")
    c3.metric("Properties analysed", f"{S['n_properties']:,}")
    c4.metric("Typical estimate error", kes_m(final["mae"]),
              help="When the valuation model estimated properties it had never seen, it was off by about this much on average.")

    # ---------------- 1. WHAT HAPPENED ----------------
    st.header("1. What happened?")
    st.write("How property prices are spread across the market today.")

    st.subheader("Price distribution")
    left, right = st.columns([3, 2])
    with left:
        show_image(["price_distribution"])
    with right:
        explain(
            "Each bar counts how many properties fall into a price band. Taller bar = more properties at that price. "
            "The line is a smoothed outline of the bars.",
            f"Most properties cost less than KES 50M. Only a few luxury homes cost more than KES 100M. "
            f"These few expensive homes stretch the chart to the right and pull the average "
            f"({kes_m(S['mean_price'])}) above the median ({kes_m(S['median_price'])}).",
        )

    st.subheader("Average price")
    show_image(["avg_price_by_location_and_type"])
    explain(
        "Each group of bars is a location. The five bars in a group are the five property types "
        "(see the colour key). The taller the bar, the higher the average price.",
        f"The overall average price is {kes_m(S['mean_price'])}. Location changes the average a lot "
        "(for example Westlands compared with Kitengela), while the five bars inside each location are "
        "close to each other - so property type changes the average very little.",
    )

    st.subheader("Median price")
    left, right = st.columns([2, 3])
    with left:
        show_image(["price_boxplot"])
    with right:
        explain(
            "The 'box' holds the middle half of all properties. The green line inside the box is the median "
            "(the middle price). The small circles at the top are unusually expensive properties.",
            f"A typical property costs about {kes_m(S['median_price'])}, and the middle half of properties cost "
            "roughly KES 13M to KES 28M. Use the median as the 'typical price' because the very expensive homes "
            "do not distort it the way they distort the average.",
        )

    st.subheader("Price by location")
    left, right = st.columns([3, 2])
    with left:
        show_image(["price_by_location", "price_by_category_boxplots"])
    with right:
        explain(
            "One box per location. The line inside each box is that location's typical (median) price; "
            "a taller box means prices there vary more.",
            "Location makes a big difference. Westlands, Runda, Karen, Kilimani and Kileleshwa are the "
            "high-priced areas; Kitengela, Mombasa and Syokimau are the lowest-priced.",
        )
        loc_df = pd.DataFrame({
            "Average price (KES M)": S["avg_price_by_location_m"],
            "Price per sqm (KES)": S["avg_price_per_sqm_by_location"],
        }).sort_values("Average price (KES M)", ascending=False)
        st.dataframe(
            loc_df,
            width="stretch",
            column_config={
                "Average price (KES M)": st.column_config.ProgressColumn(
                    format="%.1f", min_value=0, max_value=float(loc_df.iloc[:, 0].max())),
                "Price per sqm (KES)": st.column_config.NumberColumn(format="%,.0f"),
            },
        )
        st.caption(
            f"{loc_df.index[0]} is about {loc_df.iloc[0, 0] / loc_df.iloc[-1, 0]:.1f} times as expensive "
            f"as {loc_df.index[-1]}. 'Price per sqm' = price for each square metre of floor space."
        )

    st.subheader("Price by property type")
    left, right = st.columns([3, 2])
    with left:
        show_image(["price_by_type", "price_by_category_boxplots"])
    with right:
        explain(
            "One box per property type (apartment, villa and so on). The line inside each box is the typical price.",
            "The five boxes look almost the same. A villa is not priced very differently from an apartment "
            "or a townhouse on its own - where the property is and how big it is matter much more.",
        )
        type_df = pd.DataFrame({"Average price (KES M)": S["avg_price_by_type_m"]}).sort_values(
            "Average price (KES M)", ascending=False)
        st.dataframe(type_df, width="stretch")

    st.subheader("Price versus floor area")
    left, right = st.columns([3, 2])
    with left:
        show_image(["price_vs_floor_area", "price_vs_size_age_distance"])
    with right:
        explain(
            "Each dot is one property. Left to right = floor area (square metres); bottom to top = price. "
            "The black line shows the general trend.",
            "The line rises steadily: the bigger the property, the higher the price. Floor area is the "
            "most reliable single guide to price. The dots form two bands because location also matters - "
            "the same size costs more in a premium area than in a budget area.",
        )

    # ---------------- 2. WHY ----------------
    st.header("2. Why did it happen?")
    st.write(
        "The charts below show which property details the prediction models rely on most when they "
        "decide a price. A longer bar means that detail has a bigger influence on the price. "
        "Each location has its own bar (for example 'Location_Kitengela' means 'the property is in Kitengela')."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Key price factors - Random Forest")
        show_image(["rf_feature_importance"])
        explain(
            "The Random Forest is one of the two prediction methods. It ranks the top 10 details it uses to set a price.",
            "Floor area is by far the biggest influence (over half of the total). After that come specific "
            "locations such as Kitengela, Mombasa and Syokimau - the low-priced areas that clearly separate "
            "cheaper properties from the rest. Bedrooms, bathrooms and distance to the CBD count for very little.",
        )
    with right:
        st.subheader("Key price factors - XGBoost")
        show_image(["xgb_feature_importance"])
        explain(
            "XGBoost is the second prediction method. It also ranks the top 10 details it uses to set a price.",
            "Floor area is again the longest single bar, but location takes up most of the chart: the ten location "
            "bars together make up about three-quarters (77%) of XGBoost's weighting. The premium areas "
            "(Westlands, Runda, Karen) and the low-priced areas (Kitengela, Mombasa, Syokimau) all appear in "
            "the top ten, so where a property is strongly moves its price up or down.",
        )

    st.markdown(
        """
**In short**

1. **Size and location drive price.** Floor area is the single longest bar in both charts, but the ten location bars together carry even more weight than floor area (about 58% against 37% in the Voting model).
2. **Property type, bedrooms, bathrooms and distance to the CBD have little influence** on their own.
3. **Age of the property and plot size** are not among the top influences either (see the chart below).
"""
    )
    with st.expander("Do age and distance from the city centre change the price?"):
        show_image(["price_vs_age_distance", "price_vs_size_age_distance"])
        explain(
            "Each dot is a property. The left chart compares price with the property's age; the right chart "
            "compares price with its distance from the city centre (CBD). The black line is the general trend.",
            "Both lines are flat: older or more distant properties are not priced noticeably lower or higher. "
            "Do not expect a discount just because a property is older or further out.",
        )

    # ---------------- 3. WHAT IS LIKELY ----------------
    st.header("3. What is likely to happen?")
    st.write(
        "The valuation tool uses a **Voting model**. It combines two prediction methods (Random Forest and "
        "XGBoost) and averages their answers, which gives a slightly steadier estimate than either one alone. "
        "Use the **Value a property** tab to try it."
    )

    st.subheader("How accurate is the model?")
    left, right = st.columns([3, 2])
    with left:
        show_image(["model_comparison"])
    with right:
        explain(
            "Each bar is one prediction method; a longer bar means a more accurate method. The scale starts at 0.89, "
            "so the differences look larger than they really are.",
            f"All three methods are very close. The Voting model scores highest, so it is the one used in the "
            f"valuation tool. It explains about {final['r2']:.0%} of the differences between property prices.",
        )
    m_df = pd.DataFrame(S["metrics"]).T[["r2", "mae"]]
    m_df.columns = ["Accuracy score", "Typical error (KES M)"]
    m_df["Typical error (KES M)"] = m_df["Typical error (KES M)"] / 1e6
    m_df.index = [i.replace("Voting Regressor", "Voting model") for i in m_df.index]
    st.dataframe(
        m_df.style.format({"Accuracy score": "{:.1%}", "Typical error (KES M)": "{:.2f}"}),
        width="stretch",
    )
    st.caption(
        "Accuracy score: how much of the price differences the model explains (100% would be perfect). "
        "Typical error: how far off an estimate usually is."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Estimated price vs real price")
        show_image(["actual_vs_predicted"])
        explain(
            "Each dot is a property the model had never seen. Across = its real price; up = the price the model "
            "estimated. The red line is a perfect estimate. (Prices are shown on a log scale, so the numbers "
            "on the axes are not in shillings.)",
            "Dots hug the red line, so estimates follow real prices closely. At the low-price end the dots "
            "spread out and sit slightly under the line, so cheaper properties are more likely to be "
            "under-estimated.",
        )
    with right:
        st.subheader("Where estimates are less reliable")
        show_image(["residuals"])
        explain(
            "Each dot is one property. The height of a dot shows how wrong the estimate was: on the dotted "
            "line = exactly right, above = estimate too low, below = estimate too high. Left to right = "
            "estimated price from low to high.",
            "The dots bunch tightly on the right (higher-priced properties) and fan out on the left. In plain "
            "terms: estimates are most dependable for mid- and high-priced properties, and should be double-checked "
            "for low-priced ones.",
        )

    # ---------------- 4. WHAT SHOULD MANAGEMENT DO ----------------
    st.header("4. What should management consider doing?")
    st.markdown(
        """
| # | Consider | Why (from the analysis above) |
|---|----------|-------------------------------|
| 1 | **Make location the first decision** when buying, pricing or investing. | Location has the biggest effect on price; the most expensive area costs about 2.7 times the cheapest. |
| 2 | **Compare deals using price per square metre** within the same area. | Floor area is the strongest property-level driver, and price per sqm ranges from about KES 73,000 to KES 198,000 by area. |
| 3 | **Do not pay or charge extra just for the property type.** | Average prices of all five types are within KES 0.4M of each other. |
| 4 | **Test the value of extras (parking, amenities) before spending on them.** | These showed no clear effect on price in this data. This shows a pattern, not proof of cause. |
| 5 | **Use the valuation tool as a first estimate, then add an expert's opinion** - especially for lower-priced properties. | A typical estimate is off by about KES 2.1M, and the errors are larger for low-priced properties. |
| 6 | **Refresh the model regularly with new sales.** | The model only knows the market as it was in the data; prices change over time. |
"""
    )

# ===========================================================================
# TAB 2 - VALUATION
# ===========================================================================
EXAMPLE = {
    "v_loc": "Westlands", "v_type": "Apartment", "v_bed": 3, "v_bath": 2,
    "v_area": 120.0, "v_plot": 0, "v_age": 5, "v_park": 2, "v_dist": 8, "v_amen": 8,
}
DEFAULTS = {
    "v_loc": "Karen", "v_type": "Villa", "v_bed": 3, "v_bath": 2,
    "v_area": 150.0, "v_plot": 200, "v_age": 10, "v_park": 2, "v_dist": 15, "v_amen": 5,
}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def load_example():
    for k, v in EXAMPLE.items():
        st.session_state[k] = v


with tab_value:
    st.header("Estimate a property's price")
    st.write("Enter the property's details and press **Estimate price**.")

    model, columns, model_error = load_model()
    if model is None and model_error == "missing":
        st.error(
            "The valuation model files are missing from the model_assets folder "
            "(rf_model.joblib, xgb_model.json, model_columns.json). "
            "Run the export cell at the end of the notebook and copy the files into this app's folder."
        )
    elif model is None:
        st.error(
            "The valuation model could not be loaded. This usually means the library versions on this "
            "computer differ from the ones used in Colab. Install the versions in the project's "
            "requirements.txt (created by the notebook export cell).\n\n"
            f"Technical detail: {model_error}"
        )
    else:
        R = S["ranges"]
        st.button("Fill in an example property", on_click=load_example)

        with st.form("valuation_form"):
            a, b, c = st.columns(3)
            with a:
                location = st.selectbox("Location", S["locations"], key="v_loc")
                ptype = st.selectbox("Property type", S["property_types"], key="v_type")
                area = st.number_input("Floor area (sqm)", min_value=float(int(R["Floor_Area_Sqm"][0])), max_value=1000.0,
                                       step=5.0, key="v_area")
            with b:
                bedrooms = st.number_input("Bedrooms", int(R["Bedrooms"][0]), int(R["Bedrooms"][1]),
                                           key="v_bed")
                bathrooms = st.number_input("Bathrooms", int(R["Bathrooms"][0]), int(R["Bathrooms"][1]),
                                            key="v_bath")
                plot = st.selectbox("Plot size (sqm)", S["plot_sizes"], key="v_plot",
                                    format_func=lambda x: "No plot (0)" if x == 0 else f"{x:,}")
            with c:
                age = st.number_input("Age of property (years)", int(R["Age_Years"][0]),
                                      int(R["Age_Years"][1]), key="v_age")
                parking = st.number_input("Parking spaces", int(R["Parking_Spaces"][0]),
                                          int(R["Parking_Spaces"][1]), key="v_park")
                distance = st.number_input("Distance to CBD (km)", int(R["Distance_CBD_KM"][0]),
                                           int(R["Distance_CBD_KM"][1]), key="v_dist")
                amenities = st.slider("Amenities score (1 = few, 10 = many)",
                                      int(R["Amenities_Score"][0]), int(R["Amenities_Score"][1]),
                                      key="v_amen")
            submitted = st.form_submit_button("Estimate price", type="primary")

        if submitted:
            row = pd.DataFrame([{
                "Location": location, "Property_Type": ptype, "Bedrooms": bedrooms,
                "Bathrooms": bathrooms, "Floor_Area_Sqm": area, "Plot_Size_Sqm": plot,
                "Age_Years": age, "Parking_Spaces": parking, "Distance_CBD_KM": distance,
                "Amenities_Score": amenities,
            }])
            estimate = float(predict_prices(model, columns, row)[0])
            mae = S["metrics"][S["final_model"]]["mae"]
            low, high = max(estimate - mae, 0), estimate + mae

            st.divider()
            st.subheader("Result")
            r1, r2, r3 = st.columns(3)
            r1.metric("Estimated price", kes(estimate))
            r2.metric("Likely range", f"{kes_m(low)} - {kes_m(high)}",
                      help="The estimate plus or minus the model's typical error on properties it had not seen.")
            r3.metric("Price per sqm", kes(estimate / area))
            st.caption("Estimated with the Voting model (Random Forest and XGBoost averaged).")

            loc_avg = S["avg_price_by_location_m"].get(location)
            if loc_avg:
                diff = estimate / 1e6 - loc_avg
                word = "above" if diff >= 0 else "below"
                st.write(
                    f"For comparison, the average property in **{location}** costs **KES {loc_avg:.1f}M**. "
                    f"This property is estimated **KES {abs(diff):.1f}M {word}** that average."
                )

            warnings = []
            if area > 500:
                warnings.append("Very large floor area: few properties this big were in the data, so treat the estimate with extra care.")
            if estimate < 8e6:
                warnings.append("The model is less accurate for lower-priced properties - get a professional valuation before deciding.")
            if ptype != "Apartment" and plot == 0:
                warnings.append("A non-apartment with no plot is unusual - check the plot size.")
            for w in warnings:
                st.warning(w)

            st.subheader("Same property in other locations")
            what_if = pd.concat([row] * len(S["locations"]), ignore_index=True)
            what_if["Location"] = S["locations"]
            what_if["Estimated price (KES M)"] = predict_prices(model, columns, what_if) / 1e6
            what_if = (what_if[["Location", "Estimated price (KES M)"]]
                       .sort_values("Estimated price (KES M)", ascending=False)
                       .set_index("Location"))
            st.dataframe(
                what_if, width="stretch",
                column_config={"Estimated price (KES M)": st.column_config.ProgressColumn(
                    format="%.1f", min_value=0, max_value=float(what_if.iloc[:, 0].max()))},
            )
            st.caption("Everything is kept the same except the location - this shows how much location alone moves the price.")

# ===========================================================================
# TAB 3 - HELP
# ===========================================================================
with tab_help:
    st.header("How to use this app")
    st.markdown(
        """
**Market dashboard** answers four questions for decision-makers:
what happened, why it happened, what is likely to happen, and what to consider doing.
Every chart has a short explanation underneath it.

**Value a property**
1. Open the **Value a property** tab.
2. Choose the location and property type, then enter size, rooms, age, parking, distance to the CBD and amenities score.
   (Press **Fill in an example property** to see how it works.)
3. Press **Estimate price**. You get an estimated price, a likely range, the price per sqm, and how the same property would be priced in other locations.

**About the model**
- Learned from 7,000 property listings (Kenya, prices in KES).
- The final model is a *Voting model*: it averages a Random Forest and an XGBoost model.
- A typical estimate is off by about KES 2.1M. It is an estimate to support a decision, **not a formal valuation**.
- The charts in the dashboard were produced in the project notebook and saved as images; this app displays them.
"""
    )
