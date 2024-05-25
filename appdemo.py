import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from io import BytesIO

st.set_option('deprecation.showPyplotGlobalUse', False)

# Custom CSS to set the background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/free-vector/gradient-black-background-with-wavy-lines_23-2149151738.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to load the data from a file
def load_data(file):
    if file.name.endswith('csv'):
        return pd.read_csv(file)
    elif file.name.endswith('xlsx'):
        return pd.read_excel(BytesIO(file.read()), engine='openpyxl')
    elif file.name.endswith('xls'):
        return pd.read_excel(BytesIO(file.read()), engine='xlrd')
    else:
        st.error("Unsupported file format")
        return None

# Function to do some cool 2D plotting
def plot_2d(data, labels, algorithm='PCA'):
    # Select only numeric columns
    data_numeric = data.select_dtypes(include=['number'])

    if data_numeric.empty:
        st.error("No numeric data available for visualization.")
        return

    if algorithm == 'PCA':
        model = PCA(n_components=2)
    elif algorithm == 't-SNE':
        model = TSNE(n_components=2)
    else:
        st.error("Unsupported algorithm")
        return None

    transformed = model.fit_transform(data_numeric)
    df = pd.DataFrame(transformed, columns=['Component 1', 'Component 2'])
    df['Label'] = labels
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Component 1', y='Component 2', hue='Label', data=df, palette='viridis', ax=ax)
    ax.set_title(f'2D Visualization using {algorithm}')
    st.pyplot(fig)


# Main app title
st.title('Data Visualization and Machine Learning with Streamlit')

# Setting up the tabs
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Data Visualization", "Machine Learning", "Info"])

with tab1:
    st.header("Home")
    st.write("Welcome to the Data Visualization and Machine Learning App!")

# File uploader on the sidebar
uploaded_file = st.sidebar.file_uploader("Choose a file (CSV or Excel)", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    data = load_data(uploaded_file)

    if data is not None:
        # Splitting features and labels
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        with tab1:
            st.write("Here's a preview of your data:")
            st.dataframe(data)  # Display all the data

            st.write("General Distribution Plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=y, kde=True, ax=ax)
            ax.set_title('General Distribution of the Labels')
            st.pyplot(fig)

        with tab2:
            # 2D Visualization Tab
            st.sidebar.header("2D Visualization")
            vis_algo = st.sidebar.selectbox("Pick your algorithm", ['PCA', 't-SNE'])
            
            st.subheader("2D Visualization")
            st.write("Choose the algorithm and see your data in 2D")
            plot_2d(X, y, vis_algo)
            
            # Exploratory Data Analysis (EDA)
            st.sidebar.header("Exploratory Data Analysis")
            st.subheader("Exploratory Data Analysis")
            st.write("Select a chart to explore your data")
            chart_type = st.sidebar.selectbox("Pick a chart type", ['Correlation Heatmap', 'Pairplot', 'Distribution Plot'])

            if chart_type == 'Correlation Heatmap':
                st.write("Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(X.corr(), annot=True, cmap='coolwarm', ax=ax)
                ax.set_title('Correlation Heatmap')
                st.pyplot(fig)

            elif chart_type == 'Pairplot':
                st.write("Pairplot")
                sns.pairplot(data, hue=data.columns[-1])
                st.pyplot()

            elif chart_type == 'Distribution Plot':
                st.write("Distribution Plot")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=y, kde=True, ax=ax)
                ax.set_title('Distribution of the Labels')
                st.pyplot(fig)

        with tab3:
            # Machine Learning Tabs
            st.sidebar.header("Machine Learning")
            ml_task = st.sidebar.selectbox("Pick your ML task", ['Classification', 'Clustering'])

            if ml_task == 'Classification':
                st.subheader("Classification")
                test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
                random_state = st.sidebar.slider("Random State", 0, 100, 42)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

                classifiers = {
                    'Random Forest': RandomForestClassifier(random_state=random_state),
                    'SVM': SVC(random_state=random_state),
                    'Logistic Regression': LogisticRegression(random_state=random_state)
                }

                best_accuracy = 0
                best_model_name = ""
                best_model = None

                for name, model in classifiers.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)

                    st.write(f"## {name}")
                    st.write(f"Accuracy: {accuracy:.2f}")

                    # Classification Report
                    st.write("Classification Report:")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    
                    # Keep only relevant columns and format
                    report_df = report_df[['precision', 'recall', 'f1-score', 'support']]
                    report_df = report_df.round(2)
                    st.table(report_df)

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model_name = name
                        best_model = model

                    # Confusion Matrix
                    st.subheader(f"Confusion Matrix - {name}")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
                    plt.title(f'Confusion Matrix - {name}')
                    st.pyplot()

                st.write(f"### Best Model: {best_model_name} with accuracy of {best_accuracy:.2f}")

            elif ml_task == 'Clustering':
                st.subheader("Clustering")
                n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(X)

                st.write("Cluster Visualization")
                plot_2d(X, clusters, 'PCA')

                st.write("Cluster Centers")
                st.dataframe(kmeans.cluster_centers_)

                st.write("Inertia (Sum of squared distances):")
                st.write(kmeans.inertia_)

                st.write("Silhouette Score:")
                from sklearn.metrics import silhouette_score
                score = silhouette_score(X, clusters)
                st.write(score)
with tab4:
    # Info tab to provide detailed information about the app and its development process
    st.header("Info")
    st.write("""
        ## About this App
        This application is designed to empower users to visualize and analyze their data using a variety of machine learning techniques. Whether you're exploring patterns, classifying data, or clustering similar instances, this app offers a user-friendly interface for insightful analysis.

        ### Features:
        - **Data Visualization**: Leverage principal component analysis (PCA) and t-distributed stochastic neighbor embedding (t-SNE) to project high-dimensional data into 2D space, enabling intuitive visualization.
        - **Exploratory Data Analysis (EDA)**: Generate correlation heatmaps, pairplots, and distribution plots to gain deeper insights into the underlying structure of your dataset.
        - **Machine Learning**: Conduct classification tasks using algorithms such as Random Forest, Support Vector Machine (SVM), and Logistic Regression. Explore unsupervised learning with KMeans clustering.

        ### Team Contributions:
        - **Stergios Moutzikos**:
            - **Data Loading**: Implemented efficient handling and loading of CSV and Excel files to ensure seamless data integration.
            - **Table Specifications**: Designed and implemented features to ensure the correct display and management of data tables.
            - **2D Visualization Tab**: Developed functionality to visualize data using PCA and t-SNE, enhancing data exploration capabilities.
            - **Machine Learning Tabs**: Structured and implemented the machine learning workflow, enabling users to perform classification and clustering tasks effortlessly.
            - **Results and Comparison**: Implemented features to compare the performance of different machine learning models, aiding users in selecting the most suitable algorithm for their data.
            - **Info Tab**: Created this information section to provide users with comprehensive details about the application and its development process.
            - **Docker and GitHub**: Orchestrated the deployment of the application using Docker and maintained the codebase on GitHub, facilitating collaboration and version control.

        - **Nikolaos Nikolaou**:
            - **Info Tab**: Dedicated efforts to create a comprehensive information section, detailing the application's functionality, development team, and individual contributions.
            - **Docker and GitHub**: Led the development and distribution of the application through Docker, fostering effective team communication, collaboration, and code management via GitHub.
            - **Report**: Played a pivotal role in compiling a detailed report using LaTeX, encompassing the design, implementation, analysis results, conclusions, and individual contributions of each team member.
            - **UML Diagram**: Illustrated the architecture of the application and user interface through a UML diagram, providing a clear overview of the system's structure.
            - **Software Version Lifecycle**: Established a model for the software version lifecycle, tailored to the Agile methodology and optimized for widespread deployment to a diverse audience.

        - **Ioannis Savoulidis**:
            - **Data Loading**: Ensured efficient handling and loading of CSV and Excel files, streamlining the data import process for users.
            - **Table Specifications**: Implemented features to guarantee the correct display and management of data tables, enhancing the user experience and data accessibility.
            - **2D Visualization Tab**: Developed functionality for PCA and t-SNE-based data visualization, enhancing users' ability to explore and understand their data.
            - **Machine Learning Tabs**: Contributed to structuring and implementing the machine learning workflow, facilitating seamless execution of classification and clustering tasks.
            - **Results and Comparison**: Played a crucial role in comparing the results of various machine learning models, enabling informed decision-making regarding algorithm selection.
        
        """)