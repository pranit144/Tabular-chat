

# TPO Analytics and Initiative Analysis Platform

This repository contains a Flask-based web application designed to assist Training and Placement Offices (TPO) in managing various tasks, such as:

- **Team Management:** Add and manage team member details.
- **Text Extraction:** Extract text from college websites for further analysis.
- **Placement Data Analysis:** Upload and analyze placement CSV data to generate grades and visual insights.
- **Initiative Analysis:** Upload PDF or DOCX files to analyze initiatives using Natural Language Processing (NLP) techniques.
- **MoU Analysis:** Process CSV files containing Memorandum of Understanding (MoU) data and generate corresponding graphs.
- **Questionnaire Grading and SWOT Analysis:** Process TPO questionnaire responses to calculate grades, generate a final grade combining placement data, and perform a detailed SWOT analysis using Google’s Gemini API.

The project leverages several powerful libraries and tools, including:
- **Flask:** For web routing and session management.
- **pandas:** For CSV data processing.
- **plotly:** For generating interactive graphs and visual insights.
- **spaCy & TextBlob:** For NLP tasks, including sentiment analysis and entity recognition.
- **PyPDF2 & python-docx:** For extracting text from PDFs and DOCX files.
- **google.generativeai:** For generating a SWOT analysis via a generative model (e.g., Gemini API).

## Features

- **Team Member Management:** Add and display team members with details like name, post, education, email, and phone.
- **Web Text Extraction:** Input a college website URL to extract text content for further processing.
- **CSV Upload for Placement Analysis:** Upload CSV files with placement details (Package, Department, Graduation Year, Company, Post, Year of Placement) and perform detailed grade calculation using weighted parameters.
- **Initiative Analysis:** Upload PDF or DOCX files containing TPO initiatives and receive categorized insights (e.g., Training & Development, Industry Collaborations, etc.) along with sentiment analysis and entity recognition.
- **MoU Analysis:** Upload CSV files containing MoU data and generate graphical representations.
- **Questionnaire Evaluation:** Submit a questionnaire, calculate a grade based on weighted responses, and generate a SWOT analysis in JSON format using the Gemini API.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/tpo-analytics-platform.git
   cd tpo-analytics-platform
Create and Activate a Virtual Environment (Optional but Recommended):

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
Install Dependencies:

Ensure you have Python 3.7+ installed. Then install the required packages:

bash
Copy
Edit
pip install -r requirements.txt
If you don't have a requirements.txt file yet, you can create one with the following content:

txt
Copy
Edit
Flask
pandas
plotly
spacy
textblob
beautifulsoup4
requests
python-docx
PyPDF2
reportlab
google-generativeai
Download NLP Models:

The application uses spaCy's en_core_web_sm model. Install it by running:

bash
Copy
Edit
python -m spacy download en_core_web_sm
Set Up Google Generative AI API:

Obtain an API key for the Gemini API.

In the source code (typically near the top of your app.py file), configure your API key:

python
Copy
Edit
genai.configure(api_key="YOUR_API_KEY")
Usage
Run the Application:

bash
Copy
Edit
python app.py
Access the Web Interface:

Open your browser and navigate to http://127.0.0.1:5000 to interact with the application.

Features Overview:

Home Page: Manage team members, extract website text, and view placement data graphs and insights.
Initiative Analysis: Use the provided form to upload PDF or DOCX files for analyzing TPO initiatives.
CSV Upload: Upload placement and MoU CSV files to process and generate graphs and grade insights.
Questionnaire Submission: Fill in the questionnaire form to calculate grades and generate a SWOT analysis.
Project Structure
php
Copy
Edit
├── app.py                   # Main Flask application
├── templates/               # HTML templates (index.html, upload_initiatives.html, etc.)
├── uploads/                 # Directory to store uploaded files
├── static/                  # Static assets (CSS, JS, images)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
Contributing
Contributions are welcome! Feel free to open issues or submit pull requests for improvements and bug fixes.

Fork the repository.
Create a new branch for your feature or bugfix.
Commit your changes and push your branch.
Open a pull request with a detailed description of your changes.
License
This project is open-source and available under the MIT License.

Acknowledgements
Thanks to the developers of Flask, spaCy, TextBlob, Plotly, and other open-source libraries used in this project.
Special thanks to the contributors of the Google Generative AI API for enabling advanced SWOT analysis.
Happy analyzing!

yaml
Copy
Edit

---

This file is now ready to be used as your project's `README.md`. Enjoy building your TPO Analytics and Initiativ
