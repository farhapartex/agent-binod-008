from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, Any
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.memory import ChatMessageHistory


from agent_libs.tools import CalculatorTool, CustomOutputParser, CSVAnalysisTool
from agent_libs.weather import WeatherTool

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

class ComprehensiveLangChainAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-3.5-turbo",
            max_tokens=1000,
            api_key=OPENAI_API_KEY
        )
        self.embedding = OpenAIEmbeddings()
        self.memory = ChatMessageHistory()
        self.setup_vector_store()
        self.tools = self.setup_tools()
        self.setup_chains()
        self.output_parse = CustomOutputParser()
        self.setup_agent()


    def setup_vector_store(self):
        """
        * Vector stores enable semantic search - find meaning, not just keywords
        * Text splitting is crucial - right chunk size affects search quality
        * Embeddings capture meaning - similar concepts have similar vectors
        * These are your raw text documents
        * In a real app, this might be company docs, PDFs, web pages, etc.
        :return:
        """
        # TODO: will make it dynamic later
        documents = [
            "LangChain is a powerful framework for developing applications powered by language models. It provides modular components and chains for building complex AI applications.",
            "Agents in LangChain can use tools to interact with the outside world. They can search the internet, perform calculations, access databases, and execute custom functions.",
            "Memory in LangChain allows agents to remember previous conversations and maintain context across multiple interactions.",
            "Chains in LangChain allow you to combine multiple components together using the new LCEL (LangChain Expression Language) syntax.",
            "Vector stores enable semantic search and retrieval of relevant information using embeddings and similarity search algorithms.",
            "LCEL (LangChain Expression Language) is the modern way to build chains using the pipe operator: prompt | llm | output_parser",
            "Tools are functions that agents can call to perform specific tasks like web search, calculations, or database queries.",
            "Output parsers help structure the LLM responses into specific formats like JSON, lists, or custom objects."
        ]
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50, separator=". ")
        texts = text_splitter.create_documents(documents)
        for i, doc in enumerate(texts):
            doc.metadata = {"chunk_id": i, "source": "langchain_docs", "created_at": datetime.now().isoformat()}

        self.vector_store = FAISS.from_documents(texts, self.embedding)

    def setup_tools(self):
        weather_tool =WeatherTool(api_key=WEATHER_API_KEY)
        calculator_tool = CalculatorTool()
        search_tool = DuckDuckGoSearchRun()
        wikipedia_tool = WikipediaAPIWrapper()
        csv_analyzer = CSVAnalysisTool()

        def vector_search(query: str):
            try:
                docs = self.vector_store.similarity_search(query, k=2)
                return "\n".join([doc.page_content for doc in docs])
            except Exception as e:
                return f"Error searching knowledge base: {str(e)}"

        # TODO: will make it dynamic later
        tools = [
            Tool(
                name="Weather",
                func=weather_tool.run,
                description="Get current weather information for a city. Input should be a city name."
            ),
            Tool(
                name="Calculator",
                func=calculator_tool.run,
                description="Perform mathematical calculations. Input should be a mathematical expression."
            ),
            Tool(
                name="Search",
                func=search_tool.run,
                description="Search the internet for current information."
            ),
            Tool(
                name="Wikipedia",
                func=wikipedia_tool.run,
                description="Search Wikipedia for information about a topic."
            ),
            Tool(
                name="VectorSearch",
                func=vector_search,
                description="Search internal knowledge base for LangChain-related information."
            ),
            Tool(
                name="expert_analysis",
                description="Get specialized expert analysis for science, history, technology, or general topics. Input should be a question or topic you want analyzed by the appropriate expert.",
                func=lambda query: self.routing_chain.invoke(query),
            ),
            Tool(
                name="learning_plan",
                description="Create a comprehensive learning plan with summary, questions, and next steps for any topic. Input should be a topic you want to study.",
                func=lambda topic: self._format_learning_plan(self.sequential_chain.invoke({"topic": topic})),
            ),
            Tool(
                name="topic_analysis",
                description="Get both a summary and interesting questions about a topic simultaneously. Input should be any topic you want analyzed.",
                func=lambda topic: self._format_topic_analysis(self.parallel_chain.invoke({"topic": topic})),
            ),
            Tool(
                name="csv_loader",
                description="Load a CSV file and get basic information about its structure. Input should be the file path.",
                func=csv_analyzer.load_csv
            ),
            Tool(
                name="csv_structure",
                description="Get detailed data structure information of the loaded CSV file. Input can be anything (will be ignored).",
                func=lambda x: csv_analyzer.get_data_structure()
            ),
            Tool(
                name="csv_analysis",
                description="Perform data analysis. Options: 'general', 'numeric', 'categorical', 'missing', 'correlation'. Input should be the analysis type.",
                func=csv_analyzer.analyze_data
            ),
            Tool(
                name="csv_column_info",
                description="Get detailed information about a specific column. Input should be the column name.",
                func=csv_analyzer.get_column_info
            )
        ]

        return tools

    def _format_learning_plan(self, result: Dict) -> str:
        """Format learning plan result for better readability"""
        try:
            formatted = f"""**Learning Plan for {result.get('topic', 'Unknown Topic').title()}**
                **Summary:**
                {result.get('summary', 'No summary available')}
            
                ❓ **Key Questions to Explore:**
                {result.get('questions', 'No questions available')}
            
                **Next Steps:**
                {result.get('next_steps', 'No next steps available')}
                """
            return formatted.strip()
        except Exception as e:
            return f"Error formatting learning plan: {str(e)}"

    def _format_topic_analysis(self, result: Dict) -> str:
        """Format topic analysis result for better readability"""
        try:
            formatted = f"""🔍 **Topic Analysis**
                **Summary:**
                {result.get('summary', 'No summary available')}
                ❓ **Interesting Questions:**
                {result.get('questions', 'No questions available')}
                """
            return formatted.strip()
        except Exception as e:
            return f"Error formatting topic analysis: {str(e)}"


    def setup_chains(self):
        """
        * this function demonstrates three different types of chains - the building blocks of LangChain applications
        What it does (Sequential Chain (Simple LLM Chain):
        * Takes a topic as input
        * Plugs it into the prompt template
        * Sends to LLM for processing
        * Parses the output
        * Flow: Input → Prompt → LLM → Output Parser → Result
        Example:
        ```python
        result = self.simple_chain.run(topic="machine learning")
        # Output: "Machine learning is a subset of artificial intelligence..."
        ```
        What it does (Sequential Chain (Multi-Step Processing):
        * Step 1: Takes topic → generates summary
        * Step 2: Takes summary → generates questions
        * Returns both summary and questions
        * Flow: Topic → Summary Chain → Questions Chain → {summary, questions}
        Example:
        ```python
        result = self.sequential_chain({"topic": "artificial intelligence"})
        print(result["summary"])    # "AI is the simulation of human intelligence..."
        print(result["questions"])  # "1. How does AI learn? 2. What are the risks?..."
        ```

        What it does (Router Chain (Smart Routing):
        * This is the most sophisticated! It automatically chooses which specialized chain to use based on the input.
        * Step A: Define Specialized Prompts
        * Step B: Create Chain Descriptions
        * Step C: Build Destination Chains
        * Step D: Create the Router
        Flow:
        1. Input: "What is photosynthesis?"
        2. Router thinks: "This is a science question" → routes to science chain
        3. Science chain: Uses science expert prompt
        4. Output: Scientific explanation
        5. Input: "Tell me about World War II"
        6. Router thinks: "This is a history question" → routes to history chain
        7. History chain: Uses history expert prompt
        8. Output: Historical explanation
        :return:
        """
        simple_prompt = PromptTemplate.from_template(
            "You are a helpful AI assistant. Your name is Agent Binod. During first chat, inform user your name. Provide a clear and informative explanation about {topic}."
        )
        self.simple_chain = simple_prompt | self.llm | StrOutputParser()

        structured_prompt = PromptTemplate.from_template(
            """Analyze the following topic and provide a structured response: {topic}

            Please format your response as:
            Response: [Your detailed explanation here]
            Category: [Choose: Technology, Science, History, General, or Other]
            """
        )

        def parse_structured_output(text: str) -> Dict[str, Any]:
            """Parse structured output"""
            lines = text.strip().split('\n')
            result = {
                "response": text.strip(),
                "word_count": len(text.split()),
                "timestamp": datetime.now().isoformat(),
                "category": "General"
            }

            # Extract category if present
            for line in lines:
                if line.startswith("Category:"):
                    result["category"] = line.replace("Category:", "").strip()
                    break

            return result

        self.structured_chain = (
                structured_prompt
                | self.llm
                | StrOutputParser()
                | RunnableLambda(parse_structured_output)
        )

        summary_prompt = PromptTemplate.from_template("Provide a brief summary of: {topic}")
        questions_prompt = PromptTemplate.from_template("Generate 3 interesting questions about: {topic}")

        self.parallel_chain = RunnableParallel(
            summary=summary_prompt | self.llm | StrOutputParser(),
            questions=questions_prompt | self.llm | StrOutputParser()
        )

        def create_learning_plan(inputs: Dict) -> Dict:
            topic = inputs.get("topic", "")
            parallel_result = self.parallel_chain.invoke({"topic": topic})
            next_steps_prompt = PromptTemplate.from_template(
                "For the topic '{topic}', provide 3 specific and actionable next steps for learning more about it."
            )
            next_steps = (next_steps_prompt | self.llm | StrOutputParser()).invoke({"topic": topic})

            return {
                "topic": topic,
                "summary": parallel_result["summary"],
                "questions": parallel_result["questions"],
                "next_steps": next_steps
            }

        self.sequential_chain = RunnableLambda(create_learning_plan)

        def route_question(question: str) -> str:
            """Route questions to appropriate templates"""
            question_lower = question.lower()

            if any(word in question_lower for word in
                   ["science", "physics", "chemistry", "biology"]):
                return "science"
            elif any(word in question_lower for word in
                     ["history", "war", "ancient", "civilization"]):
                return "history"
            elif any(word in question_lower for word in
                     ["technology", "programming", "computer", "ai"]):
                return "technology"
            else:
                return "general"

        expert_prompts = {
            "science": PromptTemplate.from_template(
                "As a science expert, provide a detailed scientific explanation for: {input}"
            ),
            "history": PromptTemplate.from_template(
                "As a history expert, provide historical context and detailed information about: {input}"
            ),
            "technology": PromptTemplate.from_template(
                "As a technology expert, explain the technical aspects and implications of: {input}"
            ),
            "general": PromptTemplate.from_template(
                "Provide a comprehensive and helpful answer to: {input}"
            )
        }

        def route_and_process(question: str) -> str:
            """Route question and process with appropriate expert"""
            route = route_question(question)
            prompt = expert_prompts[route]
            chain = prompt | self.llm | StrOutputParser()
            return chain.invoke({"input": question})

        self.routing_chain = RunnableLambda(route_and_process)

    def setup_agent(self):
        # Create agent prompt
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """Your name is Agent Binod. You are a helpful AI assistant with access to various tools. During first chat, inform your name to the user. Use the tools when needed to provide accurate and helpful responses.

                    Available tools:
                    - weather: Get weather information for cities
                    - calculator: Perform mathematical calculations  
                    - search: Search the internet for current information
                    - wikipedia: Get detailed information from Wikipedia
                    - knowledge_search: Search LangChain knowledge base

                    Always be helpful, accurate, and provide clear explanations."""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        self.agent = create_tool_calling_agent(self.llm, self.tools, agent_prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=3,
            early_stopping_method="generate"
        )

    def run_simple_chain(self, topic: str):
        """Demonstrate simple LLM chain"""
        print(f"\n=== Simple Chain Demo ===")
        result = self.simple_chain.invoke(topic=topic)
        print(f"Result: {result}")
        return result

    def run_structured_chain(self, topic: str):
        """Demonstrate structured output chain"""
        print(f"\nStructured Chain Demo")
        print(f"Topic: {topic}")
        try:
            result = self.structured_chain.invoke({"topic": topic})
            print(f"Structured Result:")
            for key, value in result.items():
                print(f"  {key}: {value}")
            return result
        except Exception as e:
            print(f"Error: {e}")
            return None

    def run_parallel_chain(self, topic: str):
        """Demonstrate parallel processing"""
        print(f"\nParallel Chain Demo")
        print(f"Topic: {topic}")
        try:
            result = self.parallel_chain.invoke({"topic": topic})
            print(f"Summary: {result['summary']}")
            print(f"Questions: {result['questions']}")
            return result
        except Exception as e:
            print(f"Error: {e}")
            return None

    def run_sequential_chain(self, topic: str):
        """Demonstrate sequential chain"""
        print(f"\n=== Sequential Chain Demo ===")
        result = self.sequential_chain.invoke({"topic": topic})
        print(f"Summary: {result['summary']}")
        print(f"Questions: {result['questions']}")
        return result

    def run_router_chain(self, question: str):
        """Demonstrate router chain"""
        print(f"\n=== Router Chain Demo ===")
        result = self.routing_chain.invoke(question)
        print(f"Result: {result}")
        return result

    def vector_search_demo(self, query: str):
        """Demonstrate vector store search"""
        try:
            print(f"\n=== Vector Search Demo ===")
            docs = self.vector_store.similarity_search(query, k=3)
            print(f"Query: {query}")
            print("Relevant documents:")
            for i, doc in enumerate(docs, 1):
                print(f"{i}. {doc.page_content}")
                print(f"     Metadata: {doc.metadata}")
            return docs
        except Exception as e:
            print(f"Error: {e}")
            return None

    def agent_conversation(self, message: str):
        """Main agent conversation method"""
        print(f"\n=== Agent Conversation ===")
        print(f"User: {message}")

        try:
            response = self.agent.run(message)
            print(f"Agent: {response}")
            return response
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"Agent: {error_msg}")
            return error_msg

    def chat_with_agent(self, message: str):
        """Chat with the main agent"""

        try:
            # Add to memory
            self.memory.add_user_message(message)

            # Get agent response
            response = self.agent_executor.invoke({"input": message})
            agent_response = response.get("output", "No response generated")

            # Add to memory
            self.memory.add_ai_message(agent_response)

            print(f"Agent: {agent_response}")
            return agent_response

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"Agent: {error_msg}")
            return error_msg

    def demonstrate_all_features(self):
        """Run comprehensive demonstration"""
        print("LangChain Agent Demonstration")
        print("=" * 60)

        # 1. Simple Chain
        self.run_simple_chain("artificial intelligence")

        # 2. Structured Chain
        self.run_structured_chain("machine learning")

        # 3. Parallel Chain
        self.run_parallel_chain("quantum computing")

        # 4. Sequential Chain
        self.run_sequential_chain("blockchain technology")

        # 5. Routing Chain
        self.run_router_chain("How does photosynthesis work?")
        self.run_router_chain("What caused World War II?")
        self.run_router_chain("Explain neural networks")

        # 6. Vector Search
        self.vector_search_demo("What are LangChain agents?")

        # 7. Agent Conversations
        test_conversations = [
            "What's the weather like in Tokyo?",
            "Calculate the area of a circle with radius 5",
            "Search for recent news about artificial intelligence",
            "Tell me about Python programming language from Wikipedia",
            "What do you know about LCEL from your knowledge base?"
        ]

        for conversation in test_conversations:
            self.chat_with_agent(conversation)

        # 8. Memory Demo
        print(f"\n💭 Memory Demo")
        print("Recent conversation history:")
        for i, message in enumerate(self.memory.messages[-6:], 1):
            role = "Human" if isinstance(message, HumanMessage) else "AI"
            content = message.content[:100] + "..." if len(
                message.content) > 100 else message.content
            print(f"  {i}. {role}: {content}")
