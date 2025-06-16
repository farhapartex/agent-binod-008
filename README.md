# 🤖 Agent Binod v1.0.0

**A Comprehensive LangChain AI Agent for Learning and Practical Applications**

Agent Binod is a powerful, feature-rich AI assistant built with LangChain that demonstrates modern AI agent capabilities. Named after the legendary "Binod" meme, this agent brings intelligence, versatility, and a touch of humor to your AI learning journey.

## ✨ What is Agent Binod?

Agent Binod is a comprehensive LangChain-powered AI agent that showcases the full spectrum of modern AI capabilities:

- 🌤️ **Real-time Weather Information** - Get current weather for any city worldwide
- 🧮 **Smart Calculator** - Perform complex mathematical calculations
- 🔍 **Web Search** - Search the internet for current information
- 📚 **Wikipedia Integration** - Access detailed encyclopedia information
- 🧠 **Knowledge Base Search** - Semantic search through LangChain documentation
- 🎯 **Expert Analysis** - Get specialized responses from science, history, and technology experts
- 📖 **Learning Plans** - Generate comprehensive study plans for any topic
- 💡 **Topic Analysis** - Get summaries and thought-provoking questions simultaneously
- 📊 **CSV Data Analysis** - Load, analyze, and get insights from CSV files


## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- OpenWeatherMap API key (optional for weather features)

### 1. Clone or Download

```bash
git clone https://github.com/farhapartex/agent-binod-008
cd agent-binod
```

### 2. Install Dependencies (without docker)

```bash
pip install -r requirements.txt
source venv/bin/activate
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
WEATHER_API_KEY=your_openweathermap_api_key_here
```

**Getting API Keys:**

**OpenAI API Key:**
1. Visit [OpenAI Platform](https://platform.openai.com/)

**OpenWeatherMap API Key (Optional):**
1. Visit [OpenWeatherMap](https://openweathermap.org/api)
2. *Note: Without this key, weather tool will use mock data*

### 4. Project Structure

```
agent-binod/
├── agent.py              # Main agent implementation
├── agent_libs/
│   ├── tools.py          # Calculator and custom output parser
│   └── weather.py        # Weather tool implementation
├── .dockerignore         # docker ignore items file
├── .env                  # Environment variables
├── .gitignore            # git ignore items file
├── Dockerfile            # Main dockerfile file
├── requirements.txt      # Dependencies
├── main.py
└── README.md            # This file
```

## 🎮 How to Use Agent Binod

### Quick Start (without docker)

```bash
python main.py
```

### Quick Start (with docker)

```bash
docker build -t agent-binod:latest . # one time build
docker run -it --rm --env-file .env agent-binod:latest
```

### Interactive Commands

Once running, you can use these special commands:

- **`quit`** - Exit the agent
- **`demo`** - Run comprehensive feature demonstration
- **`memory`** - View conversation history

### Example Conversations

#### 📊 CSV Data Analysis
```
You: Load the CSV file at /home/user/data/sales.csv

Agent: ✅ CSV Loaded Successfully!
📁 File: sales.csv
📊 Shape: 1,000 rows × 8 columns
📋 Columns: date, product, sales, region, price...
🔧 Delimiter: ',' | Encoding: utf-8

You: Do a general analysis of the data

Agent: 📈 General Data Analysis
- Total Records: 1,000
- Numeric Columns: 3
- Categorical Columns: 4
- Missing Data: 25 cells
- Complete Rows: 975 (97.5%)
...

You: Show correlation analysis

Agent: 🔗 Correlation Analysis
Strong Correlations (|r| > 0.7):
- sales ↔ price: 0.823 (positive)
- region ↔ sales: -0.745 (negative)
...
```

#### 🌤️ Weather Queries
```
You: What's the weather in London?
Agent: 🌤️ Weather for London
Current: 15.3°C (feels like 14.1°C)
Condition: Light Rain
Humidity: 78% | Wind: 4.2 m/s
...
```

#### 🧮 Calculations
```
You: Calculate the area of a circle with radius 7.5
Agent: ✅ Result: 3.14159 * 7.5 * 7.5 = 176.71
```

#### 🎯 Expert Analysis
```
You: Explain photosynthesis scientifically
Agent: *Uses science expert chain*
As a science expert, photosynthesis is the complex biochemical process...
```

#### 📖 Learning Plans
```
You: I want to learn machine learning completely
Agent: 📚 Learning Plan for Machine Learning

📝 Summary:
Machine learning is a subset of artificial intelligence...

❓ Key Questions to Explore:
1. What are the different types of ML algorithms?
2. How do neural networks learn from data?
3. What are common real-world applications?

🎯 Next Steps:
1. Start with Python and key libraries (NumPy, Pandas, Scikit-learn)
2. Take an online course covering supervised/unsupervised learning
3. Practice with real datasets on Kaggle
```

### Advanced Features

#### Chain Demonstrations
```python
# Run individual chain demos
agent.run_simple_chain("artificial intelligence")
agent.run_parallel_chain("quantum computing")
agent.run_sequential_chain("blockchain")
agent.run_router_chain("How does DNA work?")
```

#### Vector Search
```python
# Search internal knowledge base
agent.vector_search_demo("What are LangChain agents?")
```

## 🔧 Technical Features

### Modern LangChain Implementation
- **LCEL (LangChain Expression Language)** with pipe operators
- **Tool-calling agents** with automatic tool selection
- **Vector stores** with semantic search capabilities
- **Memory management** for conversation context
- **Multiple chain types** for different use cases

### Chain Types Demonstrated

1. **Simple Chain** - Basic prompt → LLM → response
2. **Structured Chain** - Parsed output with metadata
3. **Parallel Chain** - Simultaneous processing of multiple tasks
4. **Sequential Chain** - Multi-step workflows
5. **Routing Chain** - Intelligent expert selection

### Tools Available

| Tool | Description | Example Usage |
|------|-------------|---------------|
| Weather | Real-time weather data | "Weather in Tokyo?" |
| Calculator | Mathematical calculations | "Calculate 15 * 23 + 100" |
| Search | Internet search | "Recent news about AI" |
| Wikipedia | Encyclopedia lookup | "Tell me about Python language" |
| Knowledge Search | LangChain documentation | "What are agents in LangChain?" |
| Expert Analysis | Specialized expert responses | "Explain quantum physics scientifically" |
| Learning Plan | Complete study plans | "I want to learn blockchain" |
| Topic Analysis | Summary + questions | "Analyze machine learning" |

## 🎯 Use Cases

### Educational
- Learn LangChain concepts through hands-on experience
- Understand modern AI agent architecture
- Study prompt engineering and chain composition

### Development
- Template for building production AI agents
- Reference implementation for LangChain best practices
- Starting point for custom AI applications

### Daily Assistant
- Get weather updates for travel planning
- Perform quick calculations and research
- Generate learning plans for new topics
- Get expert-level explanations on various subjects

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Weather data from [OpenWeatherMap](https://openweathermap.org/)
- Search capabilities powered by DuckDuckGo
- Encyclopedia data from Wikipedia

---

**Happy Learning with Agent Binod! 🤖✨**

*"Just like the legendary Binod comment that took the internet by storm, Agent Binod is here to make AI learning simple, powerful, and memorable!"*