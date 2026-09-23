import re
from langchain_core.tools import tool
from config.setting import config

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression securely using regex and isolated eval.
    
    Args:
        expression: A valid mathematical expression (e.g. '2 * (3 + 4)').
    """
    expression = expression.strip()
    expression = expression.replace('^', '**')
    
    if len(expression) > config.CALCULATOR_MAX_LENGTH:
        return f"Error: Expression exceeds maximum allowed length of {config.CALCULATOR_MAX_LENGTH}."

    # Whitelist check: numbers, basic math symbols, and spaces only
    if not re.match(r'^[\d+\-*/().% ]+$', expression):
        return "Error: Invalid characters detected. Only numbers and basic math operators are allowed."

    if expression.count('**') > 1:
        return "Error: Nested exponentiation is blocked to prevent CPU exhaustion."

    try:
        # Isolated eval with stripped built-ins
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except ZeroDivisionError:
        return "Math Error: Division by zero."
    except Exception as e:
        return f"Error evaluating expression: {e}"


# @tool
# def calculator(first_num: float, second_num: float, operation: str) -> dict:
#     """
#     Perform a basic arithmetic operation on two numbers.
#     Supported operations: add, sub, mul, div
#     """
#     try:
#         if operation == "add":
#             result = first_num + second_num
#         elif operation == "sub":
#             result = first_num - second_num
#         elif operation == "mul":
#             result = first_num * second_num
#         elif operation == "div":
#             if second_num == 0:
#                 return {"error": "Division by zero is not allowed"}
#             result = first_num / second_num
#         else:
#             return {"error": f"Unsupported operation '{operation}'"}
        
#         return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
#     except Exception as e:
#         return {"error": str(e)}