import h2o
import gradio as gr
import pandas as pd
import os

# ── Initialize H2O ──────────────────────────────────────────────────────────
h2o.init(nthreads=-1, max_mem_size="2G")

# ── Load saved model ─────────────────────────────────────────────────────────
# Replace the folder name below with your actual saved model folder name
MODEL_PATH = os.path.join("saved_model", os.listdir("saved_model")[0])
model = h2o.load_model(MODEL_PATH)

# ── Prediction function ───────────────────────────────────────────────────────
def predict_co2(ship_type, fuel_type, weather_conditions, month,
                distance, fuel_consumption, engine_efficiency):

    input_dict = {
        "ship_type":          ship_type,
        "fuel_type":          fuel_type,
        "weather_conditions": weather_conditions,
        "month":              month,
        "distance":           distance,
        "fuel_consumption":   fuel_consumption,
        "engine_efficiency":  engine_efficiency,
    }

    input_df   = pd.DataFrame([input_dict])
    h2o_frame  = h2o.H2OFrame(input_df)

    # Match column types the model expects
    for col in ["ship_type", "fuel_type", "weather_conditions", "month"]:
        h2o_frame[col] = h2o_frame[col].asfactor()

    prediction = model.predict(h2o_frame)
    result     = prediction.as_data_frame()["predict"][0]

    return f"{result:.4f} tonnes"


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Ocean(), title="Ship CO₂ Emission Predictor") as demo:

    gr.Markdown("""
    # 🚢 Ship CO₂ Emission Predictor
    Predict the **CO₂ emissions** of a ship voyage using an H2O AutoML (XGBoost) model.
    Fill in the voyage parameters below and click **Predict**.
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🏷️ Categorical Inputs")
            ship_type = gr.Dropdown(
                choices=["Cargo", "Tanker", "Passenger", "Container", "Bulk Carrier"],
                label="Ship Type", value="Cargo"
            )
            fuel_type = gr.Dropdown(
                choices=["Diesel", "Heavy Fuel Oil", "LNG", "MDO", "MGO"],
                label="Fuel Type", value="Diesel"
            )
            weather_conditions = gr.Dropdown(
                choices=["Calm", "Moderate", "Rough", "Stormy"],
                label="Weather Conditions", value="Calm"
            )
            month = gr.Dropdown(
                choices=["January","February","March","April","May","June",
                         "July","August","September","October","November","December"],
                label="Month", value="January"
            )

        with gr.Column():
            gr.Markdown("### 🔢 Numerical Inputs")
            distance = gr.Number(
                label="Distance Traveled (nautical miles)", value=500.0, minimum=0
            )
            fuel_consumption = gr.Number(
                label="Fuel Consumption (tonnes)", value=50.0, minimum=0
            )
            engine_efficiency = gr.Number(
                label="Engine Efficiency (0.0 – 1.0)", value=0.85,
                minimum=0.0, maximum=1.0
            )

    predict_btn = gr.Button("⚡ Predict CO₂ Emission", variant="primary", size="lg")

    with gr.Row():
        output = gr.Textbox(
            label="🌿 Predicted CO₂ Emission",
            placeholder="Result will appear here...",
            interactive=False,
            scale=1
        )

    predict_btn.click(
        fn=predict_co2,
        inputs=[ship_type, fuel_type, weather_conditions, month,
                distance, fuel_consumption, engine_efficiency],
        outputs=output
    )

    gr.Markdown("""
    ---
    > **Note:** `ship_id` and `route_id` are identifiers and excluded from prediction.  
    > Model trained with **H2O AutoML** — best model: XGBoost.
    """)

demo.launch()