import os
import io
import spacy
from textblob import TextBlob
from collections import defaultdict
from flask import Flask, render_template, request, jsonify, send_file, session
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from docx import Document
from PyPDF2 import PdfReader
from reportlab.pdfgen import canvas
import google.generativeai as genai
# Load NLP Model
nlp = spacy.load("en_core_web_sm")

team_members = []
genai.configure(api_key="AIzaSyA6axcDK2vv3GLQef2NFG64g_jW8I3m2Ag")
model = genai.GenerativeModel("gemini-1.5-flash")

app = Flask(__name__)
app.secret_key = '0fdd675e2c6f513deb04c79bd7ddb7e0'

app.config['UPLOAD_FOLDER'] = 'uploads'

# In-memory storage for data
placements_data = []
uploaded_csv_data = None

CATEGORY_KEYWORDS = {
    "Training and Development": [
        "training", "skill", "workshop", "seminar", "certification",
        "online course", "bootcamp", "hands-on training", "e-learning",
        "faculty development", "technical training", "career development"
    ],
    "Industry Collaborations": [
        "industry", "collaboration", "partnership", "internship",
        "real-world experience", "memorandum of understanding",
        "MoU", "industry tie-up", "corporate relations", "joint ventures"
    ],
    "Placement Activities": [
        "placement", "campus recruitment", "career fair", "job fair",
        "employment drive", "interview preparation", "resume building",
        "mock interviews", "campus drive", "hiring event", "job placement"
    ],
    "Research and Development": [
        "research", "development", "innovation", "project",
        "prototype", "publication", "patent", "experiment",
        "discovery", "R&D", "technical paper", "scientific study",
        "academic research"
    ],
    "Community Engagement": [
        "community service", "social responsibility", "volunteering",
        "outreach program", "awareness campaign", "environmental initiative",
        "social work", "CSR", "cleanliness drive", "tree plantation"
    ],
    "Infrastructure Development": [
        "new lab", "technology upgrade", "facility improvement",
        "smart classroom", "digital infrastructure", "equipment purchase",
        "campus renovation", "infrastructure expansion", "amenities development"
    ],
    "Other Initiatives": []  # Catch-all category for uncategorized initiatives
}

QUESTION_WEIGHTS = {
    # General Information (15%)
    'team_structure': 0.05,
    'goals_objectives': 0.05,
    'tools_software': 0.05,
    
    # Student Engagement (15%)
    'communication_system': 0.05,
    'data_collection': 0.05,
    'feedback_sessions': 0.05,
    
    # Employer Engagement (15%)
    'recruiter_strategy': 0.05,
    'onboarding_process': 0.05,
    'agreement_template': 0.05,
    
    # Placement and Training (20%)
    'training_programs': 0.05,
    'training_types': 0.05,
    'effectiveness_measure': 0.05,
    'placement_rate': 0.05,
    
    # Data Management (10%)
    'record_maintenance': 0.04,
    'trend_analysis': 0.03,
    'data_security': 0.03,
    
    # Other Categories (25%)
    'challenges': 0.05,
    'partnerships': 0.05,
    'innovation_plans': 0.05,
    'feedback_system': 0.05,
    'additional_support': 0.05
}

@app.route('/analyze_initiatives', methods=['GET', 'POST'])
def analyze_initiatives():
    """
    Analyze activities or initiatives from uploaded PDF or DOCX files.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part in the request.", 400

        file = request.files['file']

        if file and (file.filename.endswith('.pdf') or file.filename.endswith('.docx')):
            try:
                # Extract text from the file
                if file.filename.endswith('.pdf'):
                    text = extract_text_from_pdf(file)
                elif file.filename.endswith('.docx'):
                    text = extract_text_from_docx(file)

                # Analyze extracted text
                insights = generate_initiatives_insights(text)

                return render_template(
                    'initiatives_analysis.html',
                    insights=insights,
                    text=text
                )
            except Exception as e:
                return f"Error processing file: {e}", 400
        else:
            return "Invalid file format. Please upload a PDF or DOCX file.", 400

    return render_template('upload_initiatives.html')

def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file):
    """
    Extract text from a DOCX file.
    """
    doc = Document(docx_file)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

def generate_initiatives_insights(initiatives):
    """
    Generate structured insights from the extracted initiatives with extended functionality.
    """
    categorized_insights = defaultdict(list)

    for initiative in initiatives:
        matched = False

        # Match initiative to a category based on keywords
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(keyword in initiative.lower() for keyword in keywords):
                categorized_insights[category].append(initiative)
                matched = True
                break

        if not matched:
            categorized_insights["Other Initiatives"].append(initiative)

    # Perform sentiment analysis and highlight entities
    insights_html = "<ul>"
    for category, items in categorized_insights.items():
        insights_html += f"<li><strong>{category}</strong> ({len(items)} initiatives)<ul>"
        for item in items:
            # Perform sentiment analysis
            sentiment = TextBlob(item).sentiment.polarity
            sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

            # Extract entities using spaCy
            doc = nlp(item)
            entities = ", ".join([f"{ent.text} ({ent.label_})" for ent in doc.ents])

            insights_html += (
                f"<li>{item} "
                f"<ul>"
                f"<li><strong>Sentiment:</strong> {sentiment_label}</li>"
                f"<li><strong>Entities:</strong> {entities if entities else 'None'}</li>"
                f"</ul></li>"
            )
        insights_html += "</ul></li>"
    insights_html += "</ul>"

    return insights_html

def analyze_mou_projects(data):
    graphs = []

    # Total MoUs and projects per department
    mou_counts = data["Department"].value_counts().reset_index()
    mou_counts.columns = ["Department", "Count"]
    graphs.append(px.bar(mou_counts, x="Department", y="Count",
                         title="Total MoUs and Projects per Department",
                         labels={"Count": "Number of Projects"}).to_html(full_html=False))

    # Funding distribution by department
    funding_by_department = data.groupby("Department")["Funding (in Lakh)"].sum().reset_index()
    graphs.append(px.bar(funding_by_department, x="Department", y="Funding (in Lakh)",
                         title="Total Funding by Department",
                         labels={"Funding (in Lakh)": "Funding (in Lakh)"}).to_html(full_html=False))

    # Funding over the years
    funding_by_year = data.groupby("Year")["Funding (in Lakh)"].sum().reset_index()
    graphs.append(px.line(funding_by_year, x="Year", y="Funding (in Lakh)",
                          title="Funding Trends Over the Years",
                          labels={"Funding (in Lakh)": "Funding (in Lakh)"}).to_html(full_html=False))

    # Outcomes analysis
    outcomes = data["Outcome"].value_counts().reset_index()
    outcomes.columns = ["Outcome", "Count"]
    graphs.append(px.bar(outcomes, x="Outcome", y="Count",
                         title="Distribution of Project Outcomes",
                         labels={"Count": "Number of Outcomes"}).to_html(full_html=False))

    return graphs

def extract_text_from_url(url):
    """
    Extracts meaningful text from a given URL and formats it.
    """
    try:
        response = requests.get(url, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        raw_text = soup.get_text(separator="\n", strip=True)
        paragraphs = [para.strip() for para in raw_text.split("\n") if para.strip()]

        # Combine and format paragraphs
        text_to_format = "\n".join(paragraphs[:100])  # Limit to 100 paragraphs
        prompt = (
            "You are a professional content extractor and formatter. Your goal is to extract all meaningful information from the provided webpage text while excluding irrelevant content. Follow these instructions carefully: "
            "1. Capture all the main content from the webpage, including introductions, descriptions, key points, headers, and any important textual elements."
            "2. Remove irrelevant content, such as navigation menus, disclaimers, repetitive lines, advertisements, copyrights, and unstructured data (e.g., footer links or table of contents)."
            "3. Filter out special characters like '*', '©', and similar symbols that are not part of meaningful text."
            "4. Format the extracted content into structured, concise bullet points using <ul> and <li> HTML tags."
            "5. Retain the logical flow of the webpage's main content, grouping related points together."
            "6. Highlight critical keywords or phrases within each bullet point using <strong> tags to emphasize important details."
            "7. Ensure the output is comprehensive, readable, and user-friendly without redundancy or unnecessary repetition."
            "8. Avoid truncating or leaving out valuable information while maintaining clarity and brevity.\n\n"
            f"Extracted Text:\n{text_to_format}"

        )

        try:
            response = model.generate_content(prompt)
            formatted_text = response.text.strip()
            return formatted_text.replace("*", "")  # Clean up unwanted symbols
        except Exception as gemini_error:
            return f"<p>Error formatting text with Generative AI: {str(gemini_error)}</p>"

    except requests.exceptions.RequestException as e:
        return f"<p>Error fetching the website: {str(e)}</p>"

def generate_graphs(data):
    """
    Generate comprehensive graphs based on the provided placement data.
    """
    graphs = []

    # 1. Simple Distribution Analysis
    if 'Department' in data.columns:
        department_counts = data['Department'].value_counts().reset_index()
        department_counts.columns = ['Department', 'Count']
        graphs.append(px.bar(department_counts,
                             x='Department', y='Count',
                             title="Count of Placements per Department",
                             labels={'Department': 'Department', 'Count': 'Count'}).to_html(full_html=False))

    if 'Company' in data.columns:
        company_counts = data['Company'].value_counts().reset_index()
        company_counts.columns = ['Company', 'Count']
        graphs.append(px.bar(company_counts,
                             x='Company', y='Count',
                             title="Count of Placements per Company",
                             labels={'Company': 'Company', 'Count': 'Count'}).to_html(full_html=False))

    if 'Post' in data.columns:
        graphs.append(px.pie(data, names='Post',
                             title="Proportion of Placements by Post").to_html(full_html=False))

    if 'Department' in data.columns:
        graphs.append(px.pie(data, names='Department',
                             title="Share of Placements Across Departments").to_html(full_html=False))

    if 'Package' in data.columns:
        graphs.append(px.histogram(data, x='Package',
                                   title="Distribution of Package Values").to_html(full_html=False))

    if 'Year of Placement' in data.columns:
        graphs.append(px.histogram(data, x='Year of Placement',
                                   title="Distribution of Year of Placement").to_html(full_html=False))

    # 2. Comparative Analysis
    if 'Department' in data.columns and 'Package' in data.columns:
        avg_package_dept = data.groupby('Department')['Package'].mean().reset_index()
        graphs.append(px.bar(avg_package_dept, x='Department', y='Package',
                             title="Average Package by Department").to_html(full_html=False))

    if 'Post' in data.columns and 'Package' in data.columns:
        avg_package_post = data.groupby('Post')['Package'].mean().reset_index()
        graphs.append(px.bar(avg_package_post, x='Post', y='Package',
                             title="Average Package by Post").to_html(full_html=False))

    if 'Department' in data.columns and 'Post' in data.columns:
        dept_post_counts = data.groupby(['Department', 'Post']).size().reset_index(name='Count')
        graphs.append(px.bar(dept_post_counts, x='Department', y='Count', color='Post',
                             title="Distribution of Posts Within Each Department").to_html(full_html=False))

    # 3. Relationship Analysis
    if 'Package' in data.columns and 'Year of Placement' in data.columns:
        graphs.append(px.scatter(data, x='Year of Placement', y='Package',
                                 title="Package vs. Year of Placement").to_html(full_html=False))

    if 'Graduation Year' in data.columns and 'Package' in data.columns:
        graphs.append(px.scatter(data, x='Graduation Year', y='Package',
                                 title="Graduation Year vs. Package").to_html(full_html=False))

    if 'Year of Placement' in data.columns and 'Package' in data.columns and 'Company' in data.columns:
        company_counts = data['Company'].value_counts().to_dict()
        data['Company Size'] = data['Company'].map(company_counts)
        graphs.append(px.scatter(data, x='Year of Placement', y='Package', size='Company Size', color='Company',
                                 title="Year of Placement vs. Package (Bubble Size by Company)").to_html(full_html=False))

    # 4. Trend Analysis
    if 'Year of Placement' in data.columns and 'Package' in data.columns:
        avg_package_year = data.groupby('Year of Placement')['Package'].mean().reset_index()
        graphs.append(px.line(avg_package_year, x='Year of Placement', y='Package',
                              title="Trends in Average Package Over Years of Placement").to_html(full_html=False))

    if 'Year of Placement' in data.columns:
        placement_counts = data['Year of Placement'].value_counts().reset_index()
        placement_counts.columns = ['Year of Placement', 'Count']
        graphs.append(px.line(placement_counts, x='Year of Placement', y='Count',
                              title="Count of Placements Over Years of Placement").to_html(full_html=False))

    # 5. Hierarchical Analysis
    if 'Company' in data.columns and 'Post' in data.columns:
        graphs.append(px.treemap(data, path=['Company', 'Post'],
                                 title="Breakdown of Placements by Company and Post").to_html(full_html=False))

    # 6. Heatmap Analysis
    if {'Package', 'Graduation Year', 'Year of Placement'}.issubset(data.columns):
        correlation_matrix = data[['Package', 'Graduation Year', 'Year of Placement']].corr()
        graphs.append(px.imshow(correlation_matrix, text_auto=True,
                                title="Correlation Heatmap").to_html(full_html=False))

    if 'Department' in data.columns and 'Company' in data.columns:
        dept_company_counts = data.groupby(['Department', 'Company']).size().reset_index(name='Count')
        heatmap_data = dept_company_counts.pivot(index='Department', columns='Company', values='Count').fillna(0)
        graphs.append(px.imshow(heatmap_data, text_auto=True,
                                title="Frequency of Placements by Department and Company").to_html(full_html=False))

    # 7. Advanced and Custom Graphs
    if 'Department' in data.columns and 'Package' in data.columns:
        graphs.append(px.box(data, x='Department', y='Package',
                             title="Variation in Package Across Departments").to_html(full_html=False))

    if 'Post' in data.columns and 'Package' in data.columns:
        graphs.append(px.box(data, x='Post', y='Package',
                             title="Variation in Package by Post").to_html(full_html=False))

    if 'Department' in data.columns and 'Package' in data.columns:
        graphs.append(px.violin(data, x='Department', y='Package', box=True, points="all",
                                title="Distribution and Density of Package Values Across Departments").to_html(full_html=False))

    if 'Department' in data.columns and 'Package' in data.columns:
        dept_package_range = data.groupby('Department')['Package'].agg(['min', 'max']).reset_index()
        dept_package_range.columns = ['Department', 'Min Package', 'Max Package']
        graphs.append(px.line(dept_package_range, x='Department', y=['Min Package', 'Max Package'],
                              title="Minimum and Maximum Package Within Each Department").to_html(full_html=False))

    if 'Department' in data.columns and 'Post' in data.columns:
        sankey_data = data.groupby(['Department', 'Post']).size().reset_index(name='Count')
        graphs.append(go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=list(set(sankey_data['Department']) | set(sankey_data['Post']))
            ),
            link=dict(
                source=[list(set(sankey_data['Department'])).index(dep) for dep in sankey_data['Department']],
                target=[list(set(sankey_data['Post'])).index(post) + len(set(sankey_data['Department'])) for post in sankey_data['Post']],
                value=sankey_data['Count']
            )
        )).to_html(full_html=False))

    return graphs

def generate_insights(data):
    """
    Generate comprehensive insights based on the provided placement data.
    """
    insights = []

    # Helper to generate insights using Gemini
    def generate_gemini_insight(prompt):
        try:
            response = model.generate_content(prompt)
            raw_text = response.text.strip()

            # Truncate text to 100-150 words
            words = raw_text.split()
            truncated_text = " ".join(words[:150])  # Keep only the first 150 words

            # Remove unwanted '*' symbols from the raw text
            truncated_text = truncated_text.replace('*', '')

            # Split the text into sentences or points for better readability
            sentences = truncated_text.split('.')
            formatted_text = "<ul>"

            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    formatted_text += f"<li><strong>{sentence}</strong></li>"

            formatted_text += "</ul>"
            return formatted_text
        except Exception as e:
            return f"<p>Error generating insights: {str(e)}</p>"

    # Common prompt structure with enhanced formatting instructions
    base_prompt = (
        "You are an expert data analyst. Provide insights in the following format:\n"
        "1. Structure the insights into bullet points.\n"
        "2. Highlight important statistics or trends using bold text.\n"
        "3. Keep the language concise and user-friendly.\n"
        "4. Focus on the most important details and remove unnecessary repetition.\n"
    )

    # 1. Simple Distribution Analysis
    if 'Department' in data.columns:
        prompt = (
            base_prompt +
            f"Analyze the count of placements per department:\n{data['Department'].value_counts().to_dict()}.\n"
        )
        insights.append(generate_gemini_insight(prompt))

    if 'Company' in data.columns:
        prompt = (
            base_prompt +
            f"Analyze the count of placements per company:\n{data['Company'].value_counts().to_dict()}.\n"
        )
        insights.append(generate_gemini_insight(prompt))

    if 'Post' in data.columns:
        prompt = (
            base_prompt +
            f"Analyze the proportion of placements by post:\n{data['Post'].value_counts().to_dict()}.\n"
        )
        insights.append(generate_gemini_insight(prompt))

    if 'Department' in data.columns:
        prompt = (
            base_prompt +
            f"Analyze the share of placements across departments:\n{data['Department'].value_counts().to_dict()}.\n"
        )
        insights.append(generate_gemini_insight(prompt))

    if 'Package' in data.columns:
        prompt = (
            base_prompt +
            f"Analyze the distribution of package values:\n{data['Package'].describe().to_dict()}.\n"
        )
        insights.append(generate_gemini_insight(prompt))

    if 'Year of Placement' in data.columns:
        prompt = (
            base_prompt +
            f"Analyze the distribution of year of placement:\n{data['Year of Placement'].value_counts().to_dict()}.\n"
        )
        insights.append(generate_gemini_insight(prompt))

    # 2. Comparative Analysis
    if 'Department' in data.columns and 'Package' in data.columns:
        avg_package_dept = data.groupby('Department')['Package'].mean().reset_index()
        prompt = (
            base_prompt +
            f"Analyze the average package by department:\n{avg_package_dept.to_dict()}.\n"
        )
        insights.append(generate_gemini_insight(prompt))

    if 'Post' in data.columns and 'Package' in data.columns:
        avg_package_post = data.groupby('Post')['Package'].mean().reset_index()
        prompt = (
            base_prompt +
            f"Analyze the average package by post:\n{avg_package_post.to_dict()}.\n"
        )
        insights.append(generate_gemini_insight(prompt))

    if 'Department' in data.columns and 'Post' in data.columns:
        dept_post_counts = data.groupby(['Department', 'Post']).size().reset_index(name='Count').to_dict()
        prompt = (
            base_prompt +
            f"Analyze the distribution of posts within each department:\n{dept_post_counts}.\n"
        )
        insights.append(generate_gemini_insight(prompt))

    # 3. Relationship Analysis
    if 'Package' in data.columns and 'Year of Placement' in data.columns:
        correlation = data[['Package', 'Year of Placement']].corr().to_dict()
        prompt = (
            base_prompt +
            f"Analyze the relationship between Package and Year of Placement based on the correlation:\n{correlation}.\n"
        )
        insights.append(generate_gemini_insight(prompt))

    if 'Graduation Year' in data.columns and 'Package' in data.columns:
        correlation = data[['Graduation Year', 'Package']].corr().to_dict()
        prompt = (
            base_prompt +
            f"Analyze the relationship between Graduation Year and Package based on the correlation:\n{correlation}.\n"
        )
        insights.append(generate_gemini_insight(prompt))

    if 'Year of Placement' in data.columns and 'Package' in data.columns and 'Company' in data.columns:
        company_package_counts = data.groupby(['Year of Placement', 'Company'])['Package'].count().reset_index().to_dict()
        prompt = (
            base_prompt +
            f"Analyze the Year of Placement vs Package with bubble size indicating placements per company:\n{company_package_counts}.\n"
        )
        insights.append(generate_gemini_insight(prompt))

    # 4. Trend Analysis
    if 'Year of Placement' in data.columns and 'Package' in data.columns:
        avg_package_year = data.groupby('Year of Placement')['Package'].mean().reset_index().to_dict()
        prompt = (
            base_prompt +
            f"Analyze trends in average package over years of placement:\n{avg_package_year}.\n"
        )
        insights.append(generate_gemini_insight(prompt))

    if 'Year of Placement' in data.columns:
        yearly_counts = data['Year of Placement'].value_counts().reset_index()
        yearly_counts.columns = ['Year of Placement', 'Count']
        prompt = (
            base_prompt +
            f"Analyze trends in placement counts over years:\n{yearly_counts.to_dict()}.\n"
        )
        insights.append(generate_gemini_insight(prompt))

    # 5. Hierarchical Analysis
    if 'Company' in data.columns and 'Post' in data.columns:
        hierarchical_data = data.groupby(['Company', 'Post']).size().reset_index(name='Count').to_dict()
        prompt = (
            base_prompt +
            f"Analyze hierarchical data of placements by company and post:\n{hierarchical_data}.\n"
        )
        insights.append(generate_gemini_insight(prompt))

    # 6. Heatmap Analysis
    if {'Package', 'Graduation Year', 'Year of Placement'}.issubset(data.columns):
        correlation_matrix = data[['Package', 'Graduation Year', 'Year of Placement']].corr().to_dict()
        prompt = (
            base_prompt +
            f"Analyze the correlation matrix between Package, Graduation Year, and Year of Placement:\n{correlation_matrix}.\n"
        )
        insights.append(generate_gemini_insight(prompt))

    if 'Department' in data.columns and 'Company' in data.columns:
        dept_company_counts = data.groupby(['Department', 'Company']).size().reset_index(name='Count').to_dict()
        prompt = (
            base_prompt +
            f"Analyze the frequency of placements by department and company:\n{dept_company_counts}.\n"
        )
        insights.append(generate_gemini_insight(prompt))

    # 7. Advanced and Custom Graphs
    if 'Department' in data.columns and 'Package' in data.columns:
        dept_package_range = data.groupby('Department')['Package'].agg(['min', 'max']).reset_index().to_dict()
        prompt = (
            base_prompt +
            f"Analyze the variation in Package across departments:\n{dept_package_range}.\n"
        )
        insights.append(generate_gemini_insight(prompt))

    if 'Post' in data.columns and 'Package' in data.columns:
        post_package_range = data.groupby('Post')['Package'].agg(['min', 'max']).reset_index().to_dict()
        prompt = (
            base_prompt +
            f"Analyze the variation in Package by post:\n{post_package_range}.\n"
        )
        insights.append(generate_gemini_insight(prompt))

    if 'Department' in data.columns and 'Package' in data.columns:
        violin_data = data.groupby('Department')['Package'].describe().to_dict()
        prompt = (
            base_prompt +
            f"Analyze the distribution and density of Package values across departments:\n{violin_data}.\n"
        )
        insights.append(generate_gemini_insight(prompt))

    if 'Department' in data.columns and 'Post' in data.columns:
        sankey_data = data.groupby(['Department', 'Post']).size().reset_index(name='Count').to_dict()
        prompt = (
            base_prompt +
            f"Analyze the flow of students from Departments to Posts:\n{sankey_data}.\n"
        )
        insights.append(generate_gemini_insight(prompt))

    return insights


@app.route('/download_analysis', methods=['GET'])
def download_analysis():
    """
    Generate and download a PDF report of the analysis.
    """
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.setFont("Helvetica", 12)
    pdf.drawString(100, 800, "Placement Analysis Report")

    y_position = 760
    if uploaded_csv_data is not None:
        for _, row in uploaded_csv_data.iterrows():
            pdf.drawString(100, y_position, f"Department: {row['Department']}, Package: {row['Package']}, Grade: {row.get('Grade', 'N/A')}")
            y_position -= 20
            if y_position < 50:  # Create a new page if content overflows
                pdf.showPage()
                y_position = 800

    pdf.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="analysis_report.pdf", mimetype='application/pdf')

COMPANY_REPUTATION_SCORES = {
    "Google": 10,
    "Microsoft": 9,
    "Amazon": 8,
    "TCS": 6,
    "Wipro": 5
}

# Define post scores
POST_SCORES = {
    "Manager": 10,
    "Team Lead": 8,
    "Senior Developer": 6,
    "Intern": 4
}

def calculate_detailed_grade(row, dept_avg_package, year_trend_scores):
    """
    Allocate a detailed grade based on multiple weighted parameters.
    """
    # Placement Package Grade (30%)
    if row['Package'] >= 2 * dept_avg_package[row['Department']]:
        package_score = 10
    elif row['Package'] >= 1.5 * dept_avg_package[row['Department']]:
        package_score = 8
    elif row['Package'] >= dept_avg_package[row['Department']]:
        package_score = 6
    elif row['Package'] >= 0.75 * dept_avg_package[row['Department']]:
        package_score = 4
    else:
        package_score = 2

    # Company Reputation Grade (20%)
    company_score = COMPANY_REPUTATION_SCORES.get(row['Company'], 5)

    # Year of Placement Trend Grade (20%)
    year_score = year_trend_scores.get(row['Year of Placement'], 5)

    # Graduation Year Grade (15%)
    if row['Graduation Year'] >= 2023:
        grad_year_score = 10
    elif row['Graduation Year'] >= 2020:
        grad_year_score = 8
    else:
        grad_year_score = 6

    # Post or Role Grade (15%)
    post_score = POST_SCORES.get(row['Post'], 5)

    # Weighted Total Score
    total_score = (
        0.3 * package_score +
        0.2 * company_score +
        0.2 * year_score +
        0.15 * grad_year_score +
        0.15 * post_score
    )

    # Convert Total Score to Final Grade
    if total_score >= 9:
        return "A+"
    elif total_score >= 8:
        return "A"
    elif total_score >= 7:
        return "B"
    elif total_score >= 6:
        return "C"
    else:
        return "D"

def calculate_questionnaire_grade(responses):
    """
    Calculate grade based on questionnaire responses
    """
    total_score = 0
    
    # Calculate scores for Yes/No questions
    yes_no_scoring = {
        'communication_system': responses.get('q2.1', 'No') == 'Yes',
        'feedback_sessions': responses.get('q2.3', 'No') == 'Yes',
        'recruiter_strategy': responses.get('q3.1', 'No') == 'Yes',
        'agreement_template': responses.get('q3.3', 'No') == 'Yes',
        'training_programs': responses.get('q4.1', 'No') == 'Yes',
        'record_maintenance': responses.get('q5.1', 'No') == 'Yes',
        'trend_analysis': responses.get('q5.2', 'No') == 'Yes',
    }
    
    # Score text responses based on completeness and detail
    text_response_scoring = {
        'team_structure': len(responses.get('q1.1', '')) > 50,
        'goals_objectives': len(responses.get('q1.2', '')) > 50,
        'tools_software': len(responses.get('q1.3', '')) > 30,
        'data_collection': len(responses.get('q2.2', '')) > 50,
        'onboarding_process': len(responses.get('q3.2', '')) > 50,
    }
    
    # Score placement rate
    placement_rate = float(responses.get('q4.4', 0))
    placement_score = 0
    if placement_rate >= 90:
        placement_score = 1
    elif placement_rate >= 75:
        placement_score = 0.8
    elif placement_rate >= 60:
        placement_score = 0.6
    elif placement_rate >= 45:
        placement_score = 0.4
    else:
        placement_score = 0.2
    
    # Calculate total score
    for key, weight in QUESTION_WEIGHTS.items():
        if key in yes_no_scoring:
            total_score += weight * (1 if yes_no_scoring[key] else 0)
        elif key in text_response_scoring:
            total_score += weight * (1 if text_response_scoring[key] else 0)
        elif key == 'placement_rate':
            total_score += weight * placement_score
    
    # Convert score to grade
    if total_score >= 0.9:
        return "A+"
    elif total_score >= 0.8:
        return "A"
    elif total_score >= 0.7:
        return "B"
    elif total_score >= 0.6:
        return "C"
    else:
        return "D"

def calculate_final_grade(placement_grade, questionnaire_grade):
    """
    Calculate final grade combining placement data and questionnaire responses
    """
    # Grade to numeric conversion
    grade_values = {"A+": 10, "A": 9, "B": 8, "C": 7, "D": 6}
    
    # Calculate weighted average (70% placement grade, 30% questionnaire grade)
    placement_score = grade_values.get(placement_grade, 0)
    questionnaire_score = grade_values.get(questionnaire_grade, 0)
    
    final_score = (0.7 * placement_score) + (0.3 * questionnaire_score)
    
    # Convert back to letter grade
    if final_score >= 9.5:
        return "A+"
    elif final_score >= 8.5:
        return "A"
    elif final_score >= 7.5:
        return "B"
    elif final_score >= 6.5:
        return "C"
    else:
        return "D"

# Add this function to generate SWOT analysis
def generate_swot_analysis(questionnaire_responses):
    """
    Generate SWOT analysis using Gemini API based on questionnaire responses
    """
    # Combine all responses into a context string
    context = "\n".join([f"{k}: {v}" for k, v in questionnaire_responses.items()])
    
    prompt = f"""
    Based on the following TPO (Training and Placement Office) questionnaire responses, perform a detailed SWOT analysis:

    {context}

    Generate a comprehensive SWOT analysis in the following JSON format:
    {{
        "strengths": [
            {{"point": "Main strength point", "icon": "suggested-icon-name", "description": "Detailed description"}},
            ...
        ],
        "weaknesses": [
            {{"point": "Main weakness point", "icon": "suggested-icon-name", "description": "Detailed description"}},
            ...
        ],
        "opportunities": [
            {{"point": "Main opportunity point", "icon": "suggested-icon-name", "description": "Detailed description"}},
            ...
        ],
        "threats": [
            {{"point": "Main threat point", "icon": "suggested-icon-name", "description": "Detailed description"}},
            ...
        ]
    }}

    For icons, suggest appropriate icon names from Font Awesome (e.g., "fa-chart-line", "fa-users", "fa-building", etc.).
    Provide 4-5 points for each category.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return str(e)

@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Home route to handle team member management, text extraction, CSV upload, initiative analysis, and MoU analysis.
    """
    global team_members
    global uploaded_csv_data  # Global variables for team members and uploaded data
    placements = []  # Store placement data for rendering
    initiatives_text = None  # Store analyzed initiatives text
    initiatives_insights = None  # Store generated insights for initiatives
    graphs_and_insights = []  # Store graphs and insights for CSV analysis
    mou_graphs = []  # Store graphs for MoU analysis
    grades_calculated = False  # Flag for grade calculation

    # Initialize session data if not already set
    if 'placements' not in session:
        session['placements'] = []
    if 'team_members' not in session:
        session['team_members'] = []
    if 'initiatives_text' not in session:
        session['initiatives_text'] = None
    if 'initiatives_insights' not in session:
        session['initiatives_insights'] = None
    if 'uploaded_csv_data' not in session:
        session['uploaded_csv_data'] = None
    if 'mou_graphs' not in session:
        session['mou_graphs'] = []

    if request.method == 'POST':
        if 'name' in request.form:  # Handle team member addition
            # Add to session and global list
            team_members = session['team_members']
            new_member = {
                'name': request.form['name'],
                'post': request.form['post'],
                'education': request.form['education'],
                'email': request.form['email'],
                'phone': request.form['phone']
            }
            team_members.append(new_member)
            session['team_members'] = team_members

        elif 'extract_text' in request.form:  # Handle text extraction
            college_name = request.form.get('college_name', 'Unknown College')
            website_link = request.form.get('website_link', '').strip()

            if not website_link:
                return "Website URL is required for text extraction.", 400

            # Extract text from URL
            extracted_text = extract_text_from_url(website_link)
            placements = session['placements']
            new_placement = {
                "college_name": college_name,
                "website_link": website_link,
                "text": extracted_text
            }
            placements.append(new_placement)
            session['placements'] = placements

        elif 'upload_csv' in request.form:  # Handle CSV upload
            if 'file' not in request.files:
                return "No file part in the request.", 400

            file = request.files['file']
            if file and file.filename.endswith('.csv'):
                try:
                    # Save the file to the upload folder
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                    file.save(file_path)

                    # Read CSV data
                    uploaded_csv_data = pd.read_csv(file_path)

                    # Check if required columns exist for grading
                    required_columns = {'Package', 'Department', 'Graduation Year', 'Company', 'Post',
                                        'Year of Placement'}
                    if required_columns.issubset(uploaded_csv_data.columns):
                        # Perform grading logic
                        dept_avg_package = uploaded_csv_data.groupby('Department')['Package'].mean().to_dict()

                        year_placement_counts = uploaded_csv_data['Year of Placement'].value_counts()
                        year_trend_scores = {
                            year: 10 if count > year_placement_counts.mean() else 5
                            for year, count in year_placement_counts.items()
                        }

                        # Calculate detailed grades
                        uploaded_csv_data['Grade'] = uploaded_csv_data.apply(
                            lambda row: calculate_detailed_grade(row, dept_avg_package, year_trend_scores),
                            axis=1
                        )
                        grades_calculated = True

                    # Store in session
                    session['uploaded_csv_data'] = uploaded_csv_data.to_dict(orient='records')
                except Exception as e:
                    return f"Error processing file: {e}", 400

        elif 'analyze_initiatives' in request.form:  # Handle initiative analysis
            if 'file' not in request.files:
                return "No file part in the request.", 400

            file = request.files['file']
            if file and (file.filename.endswith('.pdf') or file.filename.endswith('.docx')):
                try:
                    # Extract text from the file
                    if file.filename.endswith('.pdf'):
                        initiatives_text = extract_text_from_pdf(file)
                    elif file.filename.endswith('.docx'):
                        initiatives_text = extract_text_from_docx(file)

                    # Analyze extracted text for insights
                    initiatives_insights = generate_initiatives_insights(initiatives_text.split('\n'))

                    # Store in session
                    session['initiatives_text'] = initiatives_text
                    session['initiatives_insights'] = initiatives_insights
                except Exception as e:
                    return f"Error processing file: {e}", 400

        elif 'analyze_mou' in request.form:  # Handle MoU analysis
            if 'file' not in request.files:
                return "No file part in the request.", 400

            file = request.files['file']
            if file and file.filename.endswith('.csv'):
                try:
                    # Save and read the CSV file
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                    file.save(file_path)

                    # Analyze the MoU data
                    uploaded_csv_data = pd.read_csv(file_path)
                    mou_graphs = analyze_mou_projects(uploaded_csv_data)

                    # Store MoU graphs in session
                    session['mou_graphs'] = mou_graphs
                except Exception as e:
                    return f"Error processing file: {e}", 400

        # Handle questionnaire submission
        if 'submit_questionnaire' in request.form:
            questionnaire_responses = {
                'q1.1': request.form.get('q1.1'),
                'q1.2': request.form.get('q1.2'),
                'q1.3': request.form.get('q1.3'),
                'q2.1': request.form.get('q2.1'),
                'q2.2': request.form.get('q2.2'),
                'q2.3': request.form.get('q2.3'),
                # ... (add all question responses)
            }
            
            # Calculate questionnaire grade
            questionnaire_grade = calculate_questionnaire_grade(questionnaire_responses)
            
            # If we have placement data, calculate final grade
            if uploaded_csv_data is not None and 'Grade' in uploaded_csv_data.columns:
                placement_grade = uploaded_csv_data['Grade'].mode()[0]  # Most common grade
                final_grade = calculate_final_grade(placement_grade, questionnaire_grade)
                
                # Update grades in the data
                uploaded_csv_data['Final_Grade'] = final_grade
                
                # Update session data
                session['uploaded_csv_data'] = uploaded_csv_data.to_dict(orient='records')
                session['questionnaire_grade'] = questionnaire_grade
                session['final_grade'] = final_grade

            # Generate SWOT analysis
            swot_analysis = generate_swot_analysis(questionnaire_responses)
            session['swot_analysis'] = swot_analysis

    # Load session data for rendering
    graphs = generate_graphs(uploaded_csv_data) if uploaded_csv_data is not None else []
    insights = generate_insights(uploaded_csv_data) if uploaded_csv_data is not None else []
    graphs_and_insights = [{"graph": g, "insight": i} for g, i in zip(graphs, insights)]

    overall_average_grade = None
    department_grades = {}

    if uploaded_csv_data is not None:
        # Grade mapping
        grade_mapping = {'A+': 10, 'A': 9, 'B': 8, 'C': 7, 'D': 6}
        reverse_grade_mapping = {v: k for k, v in grade_mapping.items()}
        department_grades_numeric = defaultdict(list)

        # Calculate numeric grades for department-wise grading
        for _, row in uploaded_csv_data.iterrows():
            grade_value = grade_mapping.get(row['Grade'], 0)
            department_grades_numeric[row['Department']].append(grade_value)

        # Calculate department-wise average grades
        department_grades = {
            dept: reverse_grade_mapping[round(sum(grades) / len(grades))]
            for dept, grades in department_grades_numeric.items()
        }

        # Calculate overall average grade
        all_grades = [grade for grades in department_grades_numeric.values() for grade in grades]
        if all_grades:
            overall_average_grade_numeric = round(sum(all_grades) / len(all_grades))
            overall_average_grade = reverse_grade_mapping.get(overall_average_grade_numeric, "N/A")

    return render_template(
        'index.html',
        graphs_and_insights=graphs_and_insights,
        placements=session.get('placements', []),
        initiatives_text=session.get('initiatives_text'),
        initiatives_insights=session.get('initiatives_insights'),
        grades_calculated=grades_calculated,
        mou_graphs=session.get('mou_graphs', []),
        team_members=session.get('team_members', []),
        data=session.get('uploaded_csv_data', []),
        overall_average_grade=overall_average_grade,  # Pass overall average grade
        department_grades=department_grades,  # Pass department-wise grades
        questionnaire_grade=session.get('questionnaire_grade'),
        final_grade=session.get('final_grade'),
        swot_analysis=session.get('swot_analysis')
    )

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)