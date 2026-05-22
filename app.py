import re
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


st.set_page_config(
    page_title="Personality Prediction System",
    layout="wide"
)

st.title("Personality Prediction from Writing")
st.write(
    "This project predicts MBTI personality type from written text using "
    "natural language processing and machine learning."
)


@st.cache_data
def load_dataset():
    data = pd.read_csv("mbti_1.csv")
    data = data.dropna()
    return data


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\|\|\|", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_basic_text_features(text):
    words = text.split()
    sentences = re.split(r"[.!?]", text)

    word_count = len(words)
    sentence_count = len([s for s in sentences if s.strip()])
    unique_words = len(set(words))

    avg_word_length = 0
    if word_count > 0:
        avg_word_length = sum(len(word) for word in words) / word_count

    return {
        "Word Count": word_count,
        "Sentence Count": sentence_count,
        "Unique Words": unique_words,
        "Average Word Length": round(avg_word_length, 2)
    }


def get_personality_group(personality_type):
    if personality_type.startswith("I"):
        energy = "Introversion"
    else:
        energy = "Extraversion"

    if personality_type[1] == "N":
        perception = "Intuition"
    else:
        perception = "Sensing"

    if personality_type[2] == "T":
        decision = "Thinking"
    else:
        decision = "Feeling"

    if personality_type[3] == "J":
        lifestyle = "Judging"
    else:
        lifestyle = "Perceiving"

    return energy, perception, decision, lifestyle


def describe_personality(personality_type):
    descriptions = {
        "INTJ": "Strategic, independent, analytical, and long-term focused.",
        "INTP": "Curious, logical, theoretical, and problem-solving oriented.",
        "ENTJ": "Confident, structured, goal-oriented, and leadership focused.",
        "ENTP": "Innovative, energetic, argumentative, and idea-driven.",
        "INFJ": "Insightful, idealistic, thoughtful, and values-driven.",
        "INFP": "Creative, empathetic, reflective, and emotionally aware.",
        "ENFJ": "Supportive, expressive, people-oriented, and motivating.",
        "ENFP": "Enthusiastic, imaginative, social, and flexible.",
        "ISTJ": "Responsible, practical, disciplined, and detail-oriented.",
        "ISFJ": "Loyal, caring, organized, and dependable.",
        "ESTJ": "Efficient, direct, organized, and decision-focused.",
        "ESFJ": "Friendly, cooperative, supportive, and community-oriented.",
        "ISTP": "Practical, calm, logical, and action-oriented.",
        "ISFP": "Gentle, artistic, adaptable, and observant.",
        "ESTP": "Bold, active, spontaneous, and practical.",
        "ESFP": "Social, expressive, energetic, and experience-focused."
    }

    return descriptions.get(personality_type, "No description available.")


data = load_dataset()
data["clean_posts"] = data["posts"].apply(clean_text)

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", data.shape[0])

with col2:
    st.metric("Total Columns", data.shape[1])

with col3:
    st.metric("Personality Classes", data["type"].nunique())

st.subheader("Dataset Preview")
st.dataframe(data[["type", "posts"]].head())

st.subheader("Class Distribution")

class_counts = data["type"].value_counts()

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(class_counts.index, class_counts.values)
ax1.set_xlabel("Personality Type")
ax1.set_ylabel("Number of Samples")
ax1.set_title("MBTI Class Distribution")
plt.xticks(rotation=45)
st.pyplot(fig1)

X = data["clean_posts"]
y = data["type"]

vectorizer = TfidfVectorizer(
    max_features=7000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_vectors = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectors,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = MultinomialNB()
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, test_predictions)

st.subheader("Model Details")

model_col1, model_col2 = st.columns(2)

with model_col1:
    st.write("Dataset: MBTI Personality Type Dataset")
    st.write("Feature Extraction: TF-IDF Vectorization")
    st.write("Algorithm: Multinomial Naive Bayes")
    st.write("N-gram Range: Unigrams and Bigrams")

with model_col2:
    st.write(f"Training Samples: {X_train.shape[0]}")
    st.write(f"Testing Samples: {X_test.shape[0]}")
    st.write(f"TF-IDF Features: {X_train.shape[1]}")
    st.write(f"Test Accuracy: {accuracy * 100:.2f}%")

with st.expander("View Classification Report"):
    report = classification_report(y_test, test_predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

st.subheader("Enter Writing Sample")

sample_option = st.selectbox(
    "Choose input method",
    ["Write my own text", "Use sample text"]
)

sample_texts = {
    "Analytical sample":
        "I enjoy solving complex problems and understanding how systems work. "
        "I prefer logical discussions, independent learning, and structured thinking. "
        "I like researching technical topics and improving my skills through practice.",

    "Social sample":
        "I enjoy meeting new people and working in teams. "
        "I feel motivated when I communicate with others, share ideas, and participate in group activities. "
        "I like helping people and creating a positive environment.",

    "Creative sample":
        "I like exploring new ideas and imagining different possibilities. "
        "I enjoy creative projects, writing, design, and learning about human behavior. "
        "I prefer flexible environments where I can express my thoughts freely."
}

if sample_option == "Use sample text":
    selected_sample = st.selectbox("Select a sample", list(sample_texts.keys()))
    user_text = st.text_area(
        "Writing sample",
        value=sample_texts[selected_sample],
        height=220
    )
else:
    user_text = st.text_area(
        "Write a few lines about your thoughts, interests, habits, or communication style:",
        height=220
    )

if st.button("Predict Personality Type"):
    if not user_text.strip():
        st.warning("Please enter a writing sample.")
    elif len(user_text.split()) < 20:
        st.warning("Please enter at least 20 words for better prediction.")
    else:
        cleaned_input = clean_text(user_text)
        input_vector = vectorizer.transform([cleaned_input])

        predicted_type = model.predict(input_vector)[0]
        probabilities = model.predict_proba(input_vector)[0]

        result_table = pd.DataFrame({
            "Personality Type": model.classes_,
            "Probability": probabilities
        }).sort_values(by="Probability", ascending=False)

        st.subheader("Prediction Result")
        st.success(f"Predicted Personality Type: {predicted_type}")

        energy, perception, decision, lifestyle = get_personality_group(predicted_type)

        group_col1, group_col2, group_col3, group_col4 = st.columns(4)

        with group_col1:
            st.metric("Energy", energy)

        with group_col2:
            st.metric("Perception", perception)

        with group_col3:
            st.metric("Decision Style", decision)

        with group_col4:
            st.metric("Lifestyle", lifestyle)

        st.subheader("Personality Interpretation")
        st.write(describe_personality(predicted_type))

        st.subheader("Top Prediction Probabilities")
        st.dataframe(result_table.head(8))

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.bar(
            result_table["Personality Type"].head(8),
            result_table["Probability"].head(8)
        )
        ax2.set_xlabel("Personality Type")
        ax2.set_ylabel("Probability")
        ax2.set_title("Top Personality Prediction Probabilities")
        plt.xticks(rotation=30)
        st.pyplot(fig2)

        st.subheader("Input Text Analysis")
        features = get_basic_text_features(cleaned_input)
        st.table(pd.DataFrame(features.items(), columns=["Feature", "Value"]))

        st.subheader("Project Explanation")
        st.write(
            "The input text is cleaned by removing URLs, symbols, and unnecessary spaces. "
            "Then the text is converted into numerical features using TF-IDF vectorization. "
            "A Multinomial Naive Bayes classifier predicts the MBTI personality type based on "
            "patterns learned from the dataset."
        )

st.caption(
    "This application is built for educational purposes. It demonstrates NLP-based text "
    "classification and should not be treated as a psychological diagnosis."
)