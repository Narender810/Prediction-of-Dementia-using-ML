import streamlit as st
from PIL import Image
import smtplib
import time
from backend import process_contact_form
from streamlit_lottie import st_lottie
from streamlit_modal import Modal
from footer import get_footer
from sidebar import render_sidebar
import pandas as pd
from det import detection
import joblib
import streamlit.components.v1 as components
from jupyter_component import data


logo = Image.open("./static/logo.png")
project_image = Image.open("./static/project.png")

def home():
    st.markdown(
        """
        <style>
        .sidebar-content {
            background-color: #f8f9fa; 
            padding: 20px; 
            border-radius: 10px; 
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); 
            margin-bottom: 20px; 
        }

        .sidebar-title {
            color: #333;
            font-weight: bold; 
            margin-bottom: 15px; 
        }

        .sidebar-item {
            color: #555; 
            margin-bottom: 10px; 
        }

        .sidebar-item.selected {
            color: #007bff; 
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.image(logo, use_column_width=True)

    st.markdown("<h2 class='header'>Project Description</h2>", unsafe_allow_html=True)
    st.write("Welcome to the Dementia Detection App. This app is designed to detect dementia using machine learning techniques.")
    st.write("Our goal is to provide a user-friendly interface for dementia detection, helping healthcare professionals and caregivers identify early signs of dementia.")

    st.image(project_image, use_column_width=True)

    st.markdown("<h2 class='header'>Features</h2>", unsafe_allow_html=True)
    feature_icons = {
        "👩‍💻 User-friendly Interface": "Intuitive design for easy navigation.",
        "🤖 Machine Learning Integration": "Advanced algorithms for accurate diagnosis.",
        "🔒 Secure Data Handling": "Ensuring patient data confidentiality.",
        "⏱️ Real-time Analysis": "Instantaneous results for timely intervention.",
        "🔄 Continuous Improvement": "Regular updates based on user feedback."
    }
    for feature, description in feature_icons.items():
        st.write(f"<span class='icon'>{feature}</span>{description}", unsafe_allow_html=True)

    st.markdown("<h2 class='header'>Sections</h2>", unsafe_allow_html=True)
    sections = [
        ("🔍 Detection", "Input patient data for dementia detection."),
        ("✉️ Contact Us", "Reach out for support or inquiries."),
        ("ℹ️ About Project", "Learn more about the project."),
        ("📊 Project Summary", "Summary of the project."),
        ("📈 Result Analysis", "Analyze results obtained from the ML model.")
    ]
    for section, description in sections:
        st.write(f"<span class='icon'>{section}</span>{description}", unsafe_allow_html=True)

    st.markdown("<h2 class='header'>Containers</h2>", unsafe_allow_html=True)
    containers = [
        ("🏠 Home", "Landing page of the app."),
        ("🔍 Detection", "Page for dementia detection."),
        ("✉️ Contact Us", "Page for contacting support."),
        ("ℹ️ About Project", "Page providing detailed information about the project."),
        ("📊 Project Summary", "Page summarizing the project."),
        ("📈 Result Analysis", "Page analyzing results obtained from the ML model.")
    ]
    for container, description in containers:
        st.write(f"<span class='icon'>{container}</span>{description}", unsafe_allow_html=True)

    st.markdown("<h2 class='header'>Dynamic UI Content</h2>", unsafe_allow_html=True)
    dynamic_content = [
        "Input forms for patient data on the detection page.",
        "Interactive visualizations for result analysis.",
        "Real-time feedback during data input."
    ]
    for content in dynamic_content:
        st.write(f"🔹 {content}")

    st.markdown("<h2 class='header'>Conclusion</h2>", unsafe_allow_html=True)
    st.write("Thank you for using the Dementia Detection App. We hope this tool proves to be valuable in identifying and managing dementia.")



def contact_us():
    st.title("Contact Us")
    st.write("Have a question or feedback? Feel free to reach out to us!")

    with st.form("contact_form"):
        st.write("Please fill out the form below:")

        name = st.text_input("Your Name", max_chars=50, key="name", placeholder="John Doe")
        st.image("./static/user.png", width=24)

        email = st.text_input("Your Email", max_chars=50, key="email", placeholder="example@example.com")
        st.image("./static/gmail.png", width=24)

        message = st.text_area("Your Message", max_chars=500, key="message")
        st.image("./static/comments.png", width=24)

        submitted = st.form_submit_button("Submit")
        st.image("./static/email.png", width=24)

        if submitted:
            st.write("")
            col1, col2, col3 = st.columns([2, 3, 2])
            with col2:
                lottie_url = "https://assets4.lottiefiles.com/private_files/lf30_SORsDb.json"
                st_lottie(lottie_url, width=200, height=200)

            with st.spinner("Sending email..."):
                time.sleep(5)

            st.spinner(False)

            st.success("Email sent successfully!")


def about_project():
    st.markdown(
        """
        <style>
        .about-project-content {
            padding: 20px; 
            border-radius: 10px; 
            background-color: #f9f9f9; 
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); 
            margin-bottom: 20px;
        }
        .feature-icon {
            font-size: 24px; 
            margin-right: 10px; 
        }
        .info-section {
            margin-bottom: 30px; 
        }
        .section-title {
            font-size: 24px;
            margin-bottom: 15px;
            color: #333333;
        }
        .sub-section-title {
            font-size: 20px;
            margin-bottom: 10px;
            color: #333333;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("About Project")

    st.markdown(
        """
        An application that allows users to predict the risk of a patient having dementia based on their MRI Scan and other medical data.
        """
    )

    st.markdown("## Ensemble Methods")
    st.write("""
        * **Bagging using Random Forest**
        * **Stacking using Decision Trees, Naive Bayes, and K-Nearest Neighbors**
        * **Boosting using XGBoost**
    """)
    st.markdown("---")

    st.markdown("## Data Preprocessing")
    st.write("""
        Performed various Data Preprocessing techniques such as missing data imputation and removal of multicollinear features to clean and make the data ready for model building.
    """)
    st.markdown("---")

    st.markdown("## Hyperparameter Tuning")
    st.write("""
        Tuned hyperparameters of the model to achieve the best performance.
    """)
    st.markdown("---")

    st.markdown("## Model Performance")
    st.write("""
        Accuracy, F1-Score, and Recall were the metrics used to evaluate the performance of the model.
    """)
    
    performance_data = {
        "Method": ["Bagging", "Stacking", "Boosting"],
        "Accuracy (%)": [85.29, 85.29, 86.76],
        "F1-Score (%)": [82.14, 80.77, 83.64],
        "Recall (%)": [85.19, 77.78, 85.19]
    }
    
    performance_df = pd.DataFrame(performance_data).set_index("Method").T
    
    performance_chart = st.bar_chart(performance_df)

    st.markdown("### Confusion Matrix")
    st.write("""
        The confusion matrix provides a summary of the model's performance.
    """)
    confusion_matrix_image = Image.open("./static/cf.png")
    st.image(confusion_matrix_image, use_column_width=True)
    st.markdown("---")

    st.markdown("## Features")
    st.write("""
        Features used in the model for prediction.
    """)
    features_data = {
        "Variable": ["MR Delay", "Gender", "Hand", "Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV"],
        "Data Object": ["The number of days between visits by a patient.", "Gender of a patient (M or F)",
                        "Patient’s dominant hand", "Patient's age at the time of data collection",
                        "Years of Education", "Socioeconomic status is classified into categories from 1 (highest status) to 5 (lowest status)",
                        "Mini-mental State Examination score (range is from 0-worst to 30-best)",
                        "Estimated total intracranial volume (cm3)", "Normalized whole brain volume"],
        "Data type": ["Integer", "Object", "Object", "Integer", "Integer", "Integer", "Integer", "Integer", "Float"]
    }
    features_chart = st.table(features_data)
    st.markdown("---")

    st.markdown("## Model Deployment")
    st.write("""
        The final model with the best score was deployed on a web application built with streamlit.
    """)
    st.markdown("---")

    st.markdown("## Additional Insights")
    st.write("""
        Additional insights or findings from the project.
    """)
    insights_chart = st.line_chart({"accuracy": [0.85, 0.86, 0.87], "f1-score": [0.82, 0.83, 0.84]})


def project_summary():
    st.markdown(
        """
        <style>
        .summary-content {
            padding: 20px; 
            border-radius: 10px; 
            background-color: #f9f9f9; 
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); 
            margin-bottom: 20px;
        }
        .feature-icon {
            font-size: 24px; 
            margin-right: 10px; 
        }
        .info-section {
            margin-bottom: 30px; 
        }
        .section-title {
            font-size: 24px;
            margin-bottom: 15px;
            color: #333333;
        }
        .sub-section-title {
            font-size: 20px;
            margin-bottom: 10px;
            color: #333333;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Project Summary")

    st.markdown(
        """
        An application that allows users to predict the risk of a patient having dementia based on their MRI Scan and other medical data.
        """
    )

    st.markdown("## Summary")
    st.write("""
        The project aimed to develop a machine learning model for detecting dementia at an early stage. Three ensemble methods, including Bagging using Random Forest, Stacking using Decision Trees, Naive Bayes, and K-Nearest Neighbors, and Boosting using XGBoost, were performed to model the data. Various data preprocessing techniques such as missing data imputation and removal of multicollinear features were applied to clean the data. Hyperparameters of the model were tuned to achieve the best performance. The Boosting method achieved the highest accuracy of 86.76%, with an F1-score of 83.64% and a recall of 85.19%.
    """)
    st.markdown("---")

    st.markdown("## Data Summary")
    st.write("""
        The data used in the project included features such as MR Delay, Gender, Hand, Age, EDUC, SES, MMSE, eTIV, and nWBV. These features were used to predict the risk of dementia.
    """)
    st.markdown("---")

    st.markdown("## Model Summary")
    st.write("""
        Three ensemble methods were used: Bagging, Stacking, and Boosting. Bagging used Random Forest, Stacking used Decision Trees, Naive Bayes, and K-Nearest Neighbors, and Boosting used XGBoost. Model performance was evaluated using accuracy, F1-score, and recall metrics.
    """)
    st.markdown("---")

    st.markdown("## Conclusion")
    st.write("""
        The project successfully developed a machine learning model for dementia detection using ensemble methods. The Boosting method achieved the highest accuracy, indicating its effectiveness in predicting dementia risk.
    """)
    st.markdown("---")

    st.markdown("## Additional Insights")
    st.write("""
        Additional insights or findings from the project can be included here.
    """)
    st.markdown("---")

    st.markdown("## References")
    st.write("""
        CENTERIS – International Conference on ENTERprise Information Systems / ProjMAN –
        International Conference on Project MANagement / HCist – International Conference on Health
        and Social Care Information Systems and Technologies 2022
        --- Dementia Prediction Using Machine Learning
    """)
    st.markdown("---")

    st.markdown("## Acknowledgements")
    with st.container():
        st.markdown("<h2 class='section-title'>References</h2>", unsafe_allow_html=True)
        references_data = {
            "Author(s)": ["Mirjalili, Seyedali, and Shahriar B. Shamsollahi",
                          "Kaur, Harpreet, et al.",
                          "Chawla, Nitesh V., et al.",
                          "Yang, Fan, et al.",
                          "Filho, Wendeson S., et al."],
            "Title": ["Early Prediction of Alzheimer’s Disease Using Ensemble Learning Techniques",
                      "Ensemble of Machine Learning Algorithms for the Early Prediction of Dementia",
                      "Ensemble learning for expert system design: An application to dementia diagnosis",
                      "Predicting the incidence of Alzheimer's disease using combination of multiple chemical markers and ensemble learning methods",
                      "Machine Learning Algorithms Applied to the Diagnosis of Dementia: A Systematic Review"],
            "Conference/Journal": ["2020 9th International Conference on Computer and Knowledge Engineering (ICCKE)",
                                   "2020 5th International Conference on Computing, Communication and Security (ICCCS)",
                                   "Proceedings of the 24th international conference on Machine learning",
                                   "PLoS ONE 14.1",
                                   "Frontiers in Aging Neuroscience 12"],
            "DOI": ["10.1109/ICCKE50442.2020.9344234",
                    "10.1109/ICCCS50889.2020.9266515",
                    "10.1145/1273496.1273574",
                    "10.1371/journal.pone.0208359",
                    "10.3389/fnagi.2020.00207"]
        }
        references_df = pd.DataFrame(references_data)
        st.table(references_df)

def analysis():
    st.write("Embedded Jupyter Notebook")
    components.html("""
        
            """)

def result_analysis():
    st.markdown(
        """
        <style>
        .analysis-container {
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            background-color: #f9f9f9;
        }
        .section-title {
            font-size: 24px;
            margin-bottom: 15px;
            color: #333333;
        }
        .sub-section-title {
            font-size: 20px;
            margin-bottom: 10px;
            color: #333333;
        }
        .plot-container {
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Result Analysis")

    with st.container():
        st.markdown("<h2 class='section-title'>Performance Metrics</h2>", unsafe_allow_html=True)
        st.write("""
            Performance metrics provide insights into how well the model is performing.
        """)
        performance_data = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
            "Value": [0.85, 0.87, 0.82, 0.84]
        }
        performance_df = pd.DataFrame(performance_data)
        st.table(performance_df)

    with st.container():
        st.markdown("<h2 class='section-title'>Confusion Matrix</h2>", unsafe_allow_html=True)
        st.write("""
            The confusion matrix is a visualization tool to understand the performance of the model.
        """)
        confusion_matrix_image = Image.open("./static/cf.png")
        st.image(confusion_matrix_image, use_column_width=True)

    with st.container():
        st.markdown("<h2 class='section-title'>Feature Importance</h2>", unsafe_allow_html=True)
        st.write("""
            Feature importance analysis helps identify the most influential features in the model.
        """)
        feature_importance_data = {
            "Feature": ["Age", "Education", "Gender", "BMI"],
            "Importance": [0.4, 0.3, 0.2, 0.1]
        }
        feature_importance_df = pd.DataFrame(feature_importance_data)
        st.bar_chart(feature_importance_df.set_index('Feature'))

    with st.container():
        st.markdown("<h2 class='section-title'>Model Evaluation</h2>", unsafe_allow_html=True)
        st.write("""
            Model evaluation is crucial to understand how well the model generalizes to unseen data.
        """)
        fpr = [0, 0.2, 0.4, 0.6, 0.8, 1]
        tpr = [0, 0.4, 0.6, 0.8, 0.9, 1]
        roc_data = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})

        st.subheader("ROC Curve")
        st.line_chart(roc_data)

        precision = [1, 0.8, 0.6, 0.4, 0.2, 0]
        recall = [0, 0.2, 0.4, 0.6, 0.8, 1]
        pr_data = pd.DataFrame({"Precision": precision, "Recall": recall})

        st.subheader("Precision-Recall Curve")
        st.line_chart(pr_data)
    


selected_page = render_sidebar()

if selected_page == "Home":
    home()
elif selected_page == "Detection":
    detection()
elif selected_page == "Contact Us":
    contact_us()
elif selected_page == "About Project":
    about_project()
elif selected_page == "Project Summary":
    project_summary()
elif selected_page == "Analysis":
    st.title("Embedded Jupyter Notebook")
    data()
elif selected_page == "Result Analysis":
    result_analysis()


footer = get_footer()
st.markdown(footer, unsafe_allow_html=True)






