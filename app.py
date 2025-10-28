import streamlit as st
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json, re

# 🧠 Configure Gemini (use valid key for testing)
genai.configure(api_key="AIzaSyBS0gcokMgWfQPJgqhPlBFW1AxQrNdlTC8")

# ✅ Using latest available Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")

# 🎨 Streamlit config
st.set_page_config(page_title="Toxicity Analyzer", page_icon="🤖", layout="wide")

st.title("Toxic Comment Detection System")
st.write("Analyze any Hindi/English/Hinglish text across six toxicity dimensions.")

# 📥 Input box
content = st.text_area("Enter text to analyze:", height=150, placeholder="Type or paste text here...")

# Labels
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Verdict logic
def get_verdict(avg_score):
    if avg_score > 0.75:
        return "🚨 Highly Toxic"
    elif avg_score > 0.5:
        return "⚠️ Moderately Toxic"
    elif avg_score > 0.25:
        return "🟡 Mildly Toxic"
    else:
        return "✅ Safe / Non-toxic"

if st.button("🔍 Analyze"):
    if not content.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing ..."):
            try:
                # Gemini prompt
                prompt = f"""
                Analyze the following text for toxicity across these 6 categories:
                toxic, severe_toxic, obscene, threat, insult, identity_hate.
                Respond ONLY with a JSON object where keys are these labels 
                and values are numbers between 0 and 1 (e.g., 0.67).
                Text: ```{content}```
                """

                response = model.generate_content(prompt)
                raw_text = response.text.strip()

                # Extract JSON safely
                json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                data = json.loads(json_match.group()) if json_match else {}

                if not data:
                    st.error("Could not parse Gemini response. Please retry.")
                else:
                    st.success("✅ Analysis complete!")

                    # Prepare DataFrame
                    df = pd.DataFrame(list(data.items()), columns=["Category", "Score"])
                    df["Score (%)"] = df["Score"] * 100

                    # Overall Toxicity
                    avg_score = df["Score"].mean()
                    verdict = get_verdict(avg_score)

                    st.markdown(f"### 🧭 Overall Verdict: {verdict}")
                    st.progress(float(avg_score))

                    # --- First row: Bar + Pie ---
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("📊 Bar Chart")
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.barplot(data=df, x="Category", y="Score", palette="magma", ax=ax)
                        ax.set_title("Toxicity by Category")
                        ax.set_ylim(0, 1)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)

                    with col2:
                        st.subheader("🥧 Pie Chart")
                        fig2, ax2 = plt.subplots()
                        ax2.pie(df["Score"], labels=df["Category"], autopct='%1.1f%%',
                                startangle=140, colors=sns.color_palette("cool", 6))
                        ax2.set_title("Proportion of Toxicity Types")
                        st.pyplot(fig2)

                    # --- Second row: Heatmap ---
                    st.markdown("---")
                    st.subheader("🔥 Heatmap of Toxicity Intensities")
                    fig3, ax3 = plt.subplots(figsize=(6, 3))
                    sns.heatmap(df.set_index("Category")[["Score"]], cmap="Reds",
                                annot=True, fmt=".2f", linewidths=0.5, cbar=False, ax=ax3)
                    ax3.set_title("Toxicity Intensity Heatmap")
                    st.pyplot(fig3)

                    # --- Detailed table ---
                    st.markdown("---")
                    st.subheader("📋 Detailed Scores")
                    st.dataframe(df.style.background_gradient(cmap="Reds"))

            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Enter text above and click **Analyze** to start.")
