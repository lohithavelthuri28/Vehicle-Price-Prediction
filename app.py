# app.py  – Vehicle Price Prediction Streamlit App (with imputation-aware model)

import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model(path: str = "vehicle_price_model.joblib"):
    """
    Load the trained vehicle price model.
    Make sure this file is present in the same folder as app.py.
    """
    model = joblib.load(path)
    return model


st.title("🚗 Vehicle Price Prediction Demo")

st.write(
    "Upload a CSV file with vehicle details (same feature columns used during training). "
    "The app will predict the **price** for each row and let you download the "
    "results as a CSV.\n\n"
    "Missing values (NaN) in features are handled automatically by the model using "
    "statistical imputation."
)

uploaded = st.file_uploader("Upload vehicle CSV", type=["csv"])

if uploaded is not None:
    # Read uploaded data
    df_raw = pd.read_csv(uploaded)
    st.write("Preview of uploaded data:")
    st.dataframe(df_raw.head())

    # Load model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Could not load model file: {e}")
        st.stop()

    # We rely on the pipeline's imputers to handle NaN in features.
    df_for_pred = df_raw.copy()

    with st.spinner("Predicting vehicle prices..."):
        try:
            preds = model.predict(df_for_pred)
        except Exception as e:
            st.error(
                "Prediction error. This usually happens if the CSV is missing some "
                "columns that were present during training.\n\n"
                f"Details: {e}"
            )
            st.stop()

    # Prepare output
    df_out = df_raw.copy()
    df_out["predicted_price"] = preds

    st.subheader("Predicted Prices (first 20 rows)")
    st.dataframe(df_out.head(20))

    st.download_button(
        "Download full predictions as CSV",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name="vehicle_price_predictions.csv",
        mime="text/csv",
    )
