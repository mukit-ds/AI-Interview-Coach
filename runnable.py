import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
import os

# HuggingFace LLM Setup
llm = HuggingFaceEndpoint(
    repo_id='HuggingFaceH4/zephyr-7b-beta',
    task='text-generation',
    huggingfacehub_api_token="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"  # Secure with env in production
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# Prompt Templates
prompt1 = PromptTemplate(
    template='Generate 5 interview questions for a beginner in {topic} with increasing difficulty.',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Answer all the following questions:\n{questions}',
    input_variables=['questions']
)

feedback_prompt = PromptTemplate(
    template=(
        "Evaluate this user's answer to the following question. "
        "Rate it out of 10 and give suggestions to improve:\n\n"
        "Question: {question}\n"
        "User Answer: {user_answer}"
    ),
    input_variables=['question', 'user_answer']
)

recommendation_prompt = PromptTemplate(
    template=(
        "Based on the following user answer, recommend 3 topics the user should improve upon:\n\n"
        "Question: {question}\n"
        "User Answer: {user_answer}"
    ),
    input_variables=['question', 'user_answer']
)

# Chains
question_chain = RunnableSequence(prompt1, model, parser)
answer_chain = RunnableSequence(prompt2, model, parser)
feedback_chain = RunnableSequence(feedback_prompt, model, parser)
recommend_chain = RunnableSequence(recommendation_prompt, model, parser)

# Streamlit UI
st.set_page_config(page_title="AI Interview QA", layout="centered")
st.title("🎤 AI Interview Question Generator & Scorer")

topic = st.text_input("Enter a topic (e.g., AI, Machine Learning, CNNs):")

if st.button("Generate Questions & Answers"):
    if topic:
        with st.spinner("Generating questions..."):
            questions = question_chain.invoke({"topic": topic})
        st.subheader("🤖 Interview Questions")
        st.markdown(questions)

        with st.spinner("Generating answers..."):
            answers = answer_chain.invoke({"questions": questions})
        st.subheader("✅ Sample AI Answers")
        st.markdown(answers)

        st.session_state['questions'] = questions.split('\n')  # Store questions for user answer
    else:
        st.warning("Please enter a topic.")

# Feature 3: User Answer Feedback
if 'questions' in st.session_state:
    st.subheader("✍️ Try Answering Yourself")
    selected_q = st.selectbox("Choose a question to answer:", st.session_state['questions'])
    user_answer = st.text_area("Your Answer:", height=150)

    if st.button("🧠 Get Feedback & Suggestions"):
        if selected_q and user_answer.strip():
            with st.spinner("Evaluating your answer..."):
                feedback = feedback_chain.invoke({
                    "question": selected_q,
                    "user_answer": user_answer
                })
                st.success("✅ Feedback:")
                st.markdown(feedback)

            with st.spinner("Recommending topics to improve..."):
                recommendations = recommend_chain.invoke({
                    "question": selected_q,
                    "user_answer": user_answer
                })
                st.info("📚 Recommended Topics to Study:")
                st.markdown(recommendations)
        else:
            st.warning("Please write an answer before submitting.")
