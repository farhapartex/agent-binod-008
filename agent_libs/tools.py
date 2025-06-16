from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class CalculatorTool:
    """Custom calculator tool"""

    def __init__(self):
        self.name = "Calculator"
        self.description = "Perform basic mathematical calculations"

    def run(self, expression: str) -> str:
        """Safely evaluate mathematical expressions"""
        try:
            # Only allow safe mathematical operations
            expression = expression.strip()
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"

            if any(dangerous in expression for dangerous in ['__', 'import', 'exec', 'eval']):
                return "Error: Invalid expression"

            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"


class ParsedResponse(BaseModel):
    """Structured response model for output parsing"""
    response: str = Field(description="The main response text")
    word_count: int = Field(description="Number of words in response")
    timestamp: str = Field(description="When the response was generated")
    category: str = Field(description="Category of the response")

class CustomOutputParser(PydanticOutputParser):
    """Custom output parser to demonstrate LangChain parsing capabilities"""

    def __init__(self):
        super().__init__(pydantic_object=ParsedResponse)