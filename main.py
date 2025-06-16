from agent import ComprehensiveLangChainAgent
from langchain_core.messages import HumanMessage

from agent_libs.tools import CSVAnalysisTool


def main():
    try:
        agent = ComprehensiveLangChainAgent()

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