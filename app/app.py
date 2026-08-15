import streamlit as st
import pandas as pd

from src.classifier import classify_text, load_metrics
from src.decision import get_decision


# Configure the Streamlit application.
st.set_page_config(
    page_title="AI Support Ticket Classifier",
    page_icon="🎯",
    layout="wide"
)


# Store prediction history only for the current browser session.
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# Load the saved model metrics for the dashboard.
metrics = load_metrics()


# Application header.
st.title("🎯 AI Support Ticket Classifier")

st.write(
    "An ML-powered support ticket routing system that classifies "
    "customer messages and recommends whether they should be "
    "automatically routed or reviewed by a human."
)


st.divider()


# Display high-level model performance.
metric_columns = st.columns(4)

with metric_columns[0]:
    st.metric(
        "Model Accuracy",
        f"{metrics['accuracy']:.1%}"
    )

with metric_columns[1]:
    st.metric(
        "Categories",
        len(metrics["categories"])
    )

with metric_columns[2]:
    st.metric(
        "Test Samples",
        metrics["test_samples"]
    )

with metric_columns[3]:
    st.metric(
        "Model",
        "TF-IDF + Logistic Regression"
    )


st.divider()


# Sidebar contains the application's routing policy.
with st.sidebar:
    st.header("Routing Categories")

    st.write(
        """
        **Supported categories**

        - 🚚 Shipping
        - 👤 Account
        - 🛠️ Technical
        - 💬 General Support
        """
    )

    st.divider()

    st.header("Routing Policy")

    st.write("🟢 **80%+** → Automatic routing")
    st.write("🟡 **50–79%** → Human review")
    st.write("🔴 **Below 50%** → Request clarification")

    st.divider()

    st.caption(
        "The classifier uses TF-IDF text features and "
        "Logistic Regression to predict the most likely "
        "support category."
    )


st.subheader("Classify a Support Ticket")


# Let the user enter a realistic customer complaint.
message = st.text_area(
    "Customer message",
    placeholder=(
        "Example: My package says delivered but I haven't received it."
    ),
    height=160
)


# Provide realistic examples for quick application testing.
examples = {
    "Shipping": "My package says delivered but I haven't received it.",
    "Account": "I forgot my password and cannot log into my account.",
    "Technical": "The website keeps crashing whenever I try to use it.",
    "General Support": "I have a problem and need help from customer support."
}


st.caption("Try an example")

example_columns = st.columns(4)

for column, (label, example) in zip(example_columns, examples.items()):
    if column.button(
        label,
        use_container_width=True
    ):
        st.session_state.example_message = example


# Use the selected quick example as the current message.
if "example_message" in st.session_state:
    message = st.session_state.example_message


if st.button(
    "Classify Ticket",
    type="primary",
    use_container_width=True
):

    if not message.strip():
        st.warning(
            "Please enter a customer message first."
        )

    else:
        # Run the trained ML classifier.
        result = classify_text(message)

        category = result["category"]
        confidence = result["confidence"]

        # Convert model confidence into an operational routing decision.
        decision = get_decision(confidence)


        # Save this prediction to the current session history.
        st.session_state.prediction_history.append(
            {
                "Message": message,
                "Category": category.replace("_", " ").title(),
                "Confidence": f"{confidence:.1%}",
                "Decision": decision["status"].replace(
                    "_", " "
                ).title()
            }
        )


        st.divider()

        st.subheader("Prediction")


        result_columns = st.columns(2)

        with result_columns[0]:
            st.metric(
                "Predicted Category",
                category.replace(
                    "_", " "
                ).title()
            )

        with result_columns[1]:
            st.metric(
                "Model Confidence",
                f"{confidence:.1%}"
            )


        # Visualize how confident the model is in its prediction.
        st.progress(
            min(max(confidence, 0.0), 1.0),
            text=f"Confidence: {confidence:.1%}"
        )


        st.subheader("Routing Decision")


        # Display a different message depending on the confidence level.
        if decision["status"] == "HIGH_CONFIDENCE":

            st.success(
                f"✅ {decision['message']}"
            )

        elif decision["status"] == "REVIEW":

            st.warning(
                f"⚠️ {decision['message']}"
            )

        else:

            st.error(
                f"❓ {decision['message']}"
            )


        st.write(
            f"**Recommended action:** "
            f"{decision['action'].replace('_', ' ').title()}"
        )


        st.divider()

        st.subheader("Model Explanation")


        st.write(
            f"The model classified this message as "
            f"**{category.replace('_', ' ').title()}** "
            f"with **{confidence:.1%} confidence**."
        )


        if decision["status"] == "HIGH_CONFIDENCE":

            st.write(
                "The prediction meets the automatic-routing "
                "threshold."
            )

        elif decision["status"] == "REVIEW":

            st.write(
                "The prediction is useful, but the confidence "
                "does not meet the automatic-routing threshold. "
                "A human should review the ticket."
            )

        else:

            st.write(
                "The model is uncertain about the category. "
                "The customer should provide additional "
                "information before the ticket is routed."
            )


# Show predictions made during the current session.
if st.session_state.prediction_history:

    st.divider()

    st.subheader("Prediction History")

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )


    if st.button("Clear Prediction History"):

        st.session_state.prediction_history = []

        st.rerun()


st.divider()


# Footer.
st.caption(
    "AI Support Ticket Classifier • "
    "TF-IDF + Logistic Regression • "
    "Machine Learning Application"
)