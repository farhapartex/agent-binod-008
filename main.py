import os
from dotenv import load_dotenv
from agent.core import ComprehensiveLangChainAgent
from langchain_core.messages import HumanMessage

load_dotenv()

OPEN_AI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
PDF_FILE_PATH = os.getenv("PDF_FILE_PATH", None)


def main():
    try:
        agent = ComprehensiveLangChainAgent(OPEN_AI_MODEL, PDF_FILE_PATH)

        # Interactive mode
        print(f"\n{'=' * 60}")
        print("💬 Interactive Mode - Chat with the agent!")
        print("Commands: 'quit' to exit, 'demo' to run demos, 'memory' to see chat history")
        print("=" * 60)

        while True:
            user_input = input("\n🧑 You: ").strip()

            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'demo':
                agent.demonstrate_all_features()
            elif user_input.lower() == 'memory':
                print("\n💭 Chat History:")
                for msg in agent.memory.messages:
                    role = "You" if isinstance(msg, HumanMessage) else "Agent"
                    print(f"  {role}: {msg.content[:100]}...")
            elif user_input:
                agent.chat_with_agent(user_input)

    except Exception as e:
        print(f"Error initializing agent: {e}")


if __name__ == "__main__":
    main()