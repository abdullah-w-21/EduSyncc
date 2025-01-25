# EduSyncc 📚

EduSyncc is an intelligent offline learning content optimizer a project made for LabLab.ai (Ai for connectivity Hackathon) 


EduSyncc is an intelligent educational content management system that helps educators optimize content delivery through smart network management and curriculum alignment analysis. The application uses AI to analyze educational materials, match them with curriculum requirements, and intelligently schedule downloads based on network conditions.

## 🌟 Features

### Curriculum Management
- Create and manage structured curriculum content
- Define learning objectives and prerequisites
- Organize content by grade level and subject
- Track curriculum alignment and coverage

### Content Analysis
- AI-powered content analysis using Groq's LLaMA 3 model
- Automatic extraction of learning objectives
- Grade level and subject identification
- Curriculum alignment scoring
- Content priority assessment
- Support for multiple file formats:
  - PDF documents
  - Word documents (.doc, .docx)
  - Text files
  - Web content
  - Google Drive links

### Smart Network Management
- Real-time network speed monitoring
- Predictive network performance analysis
- Intelligent download scheduling
- Download queue prioritization
- Automatic retry mechanism for failed downloads
- Network usage optimization during peak hours

### Analytics Dashboard
- Comprehensive content overview
- Network performance metrics
- Download status tracking
- Curriculum alignment visualization
- Resource optimization insights

## 📋 Prerequisites

- Python 3.11
- Streamlit
- SQLite3
- Groq API key
- Required Python packages (see requirements.txt)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/abdullah-w-21/EduSyncc.git
cd EduSyncc
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up your Groq API key:
   - Create a `.streamlit/secrets.toml` file
   - Add your API key:
     ```toml
     GROQ_API_KEY = "your-api-key-here"
     ```

5. Initialize the database:
```bash
python -c "from app import init_db; init_db()"
```

## 💻 Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Navigate to the different sections using the sidebar:
   - **Dashboard**: Overview of system metrics and insights
   - **Curriculum Management**: Set up and manage curriculum content
   - **Content Analysis**: Upload and analyze educational materials
   - **Network Monitor**: Track network performance and manage downloads

### Setting Up Curriculum

1. Navigate to "Curriculum Management"
2. Use the "Add New Curriculum Content" form to input:
   - Grade Level
   - Subject
   - Topic
   - Learning Objectives
   - Prerequisites
3. Click "Add to Curriculum" to save

### Analyzing Content

1. Go to "Content Analysis"
2. Choose between:
   - Upload files directly
   - Provide content URLs
3. The system will automatically:
   - Extract content
   - Analyze alignment with curriculum
   - Generate learning objectives
   - Schedule downloads if needed

### Monitoring Network Performance

1. Access "Network Monitor"
2. Features available:
   - Run speed tests
   - View network history
   - Check download queue
   - See network predictions
   - Process download queue

## 🔧 Configuration

### Network Optimization Settings
```python
MIN_DOWNLOAD_SPEED = 5.0  # Minimum acceptable speed in Mbps
PEAK_HOURS = range(9, 18)  # 9 AM to 6 PM
PEAK_MIN_SPEED = 10.0  # Minimum speed during peak hours
```

### Database Schema

The system uses SQLite with the following tables:

- `speed_tests`: Network performance data
- `content_analysis`: Analyzed content information
- `download_schedule`: Download queue management
- `curriculum`: Curriculum structure and requirements

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


## 🔍 Technical Details

### AI Analysis

The system uses Groq's LLaMA 3 model for content analysis with the following capabilities:
- Content summarization
- Learning objective extraction
- Grade level assessment
- Curriculum alignment scoring
- Priority calculation

### Network Prediction

The system implements a Random Forest model for network performance prediction:
- Features historical speed test data
- Accounts for time-of-day patterns
- Considers peak usage periods
- Adapts to changing network conditions

### File Processing

Supports multiple content sources and formats:
- Direct file uploads
- URL content extraction
- Google Drive integration
- PDF text extraction
- Word document processing
- Web content scraping

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## ✨ Acknowledgments

- [Streamlit](https://streamlit.io/) for the web framework
- [Groq](https://groq.com/) for AI capabilities
- [Plotly](https://plotly.com/) for data visualization
- All contributors and users of EduSyncc
