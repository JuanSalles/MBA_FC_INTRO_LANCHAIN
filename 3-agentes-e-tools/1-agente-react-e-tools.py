from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from dotenv import load_dotenv

load_dotenv()

# ==================== DEFINIÇÃO DE TOOLS ====================

@tool
def calculator(expression: str) -> str:
    """Evaluate a simple mathematical expression and return the result as a string."""
    try:
        result = eval(expression)
    except Exception as e:
        return f"Error: {e}"
    return str(result)

@tool
def web_search_mock(query: str) -> str:
    """Return the capital of a given country if it exists in the mock data."""
    data = {
        "Brazil": "Brasília",
        "France": "Paris",
        "Germany": "Berlin",
        "Italy": "Rome",
        "Spain": "Madrid",
        "United States": "Washington, D.C."
        
    }
    for country, capital in data.items():
        if country.lower() in query.lower():
            return f"The capital of {country} is {capital}."
    return "I don't know the capital of that country."

# ==================== INICIALIZAÇÃO ====================

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
tools = [calculator, web_search_mock]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""Answer the following questions as best you can. 
    Only use the information you get from the tools, even if you know the answer.
    If the information is not provided by the tools, say you don't know."""
)

# ==================== FUNÇÕES DE VERIFICAÇÃO ====================

def verify_tool_usage(result):
    """Verifica e exibe todas as tools chamadas com detalhes completos."""
    messages = result.get("messages", [])
    
    print("\n" + "="*70)
    print("📊 ANÁLISE COMPLETA DA EXECUÇÃO DO AGENTE")
    print("="*70)
    
    tools_used = []
    tool_results = []
    
    for i, message in enumerate(messages):
        message_type = type(message).__name__
        print(f"\n[Etapa {i}] {message_type}")
        print("-" * 70)
        
        if isinstance(message, HumanMessage):
            print(f"  👤 ENTRADA DO USUÁRIO:")
            print(f"     {message.content}")
        
        elif isinstance(message, AIMessage):
            # Verifica se tem tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    tool_id = tool_call.get('id', 'N/A')
                    
                    tools_used.append({
                        'name': tool_name,
                        'args': tool_args,
                        'id': tool_id
                    })
                    
                    print(f"  🔧 TOOL CHAMADA:")
                    print(f"     Nome: {tool_name}")
                    print(f"     ID: {tool_id}")
                    print(f"     Argumentos: {tool_args}")
            else:
                # Resposta do agente
                print(f"  🤖 RESPOSTA DO AGENTE:")
                print(f"     {message.content}")
        
        elif isinstance(message, ToolMessage):
            # Resultado da tool
            tool_result = {
                'tool': message.name,
                'tool_id': message.tool_call_id,
                'result': message.content
            }
            tool_results.append(tool_result)
            
            print(f"  📤 RESULTADO DA TOOL:")
            print(f"     Tool: {message.name}")
            print(f"     ID: {message.tool_call_id}")
            print(f"     Resultado: {message.content}")
    
    # ==================== RESUMO FINAL ====================
    print("\n" + "="*70)
    print("📋 RESUMO FINAL")
    print("="*70)
    
    if tools_used:
        print(f"\n✅ FERRAMENTAS UTILIZADAS: {len(tools_used)}")
        for i, tool in enumerate(tools_used, 1):
            print(f"\n   {i}. {tool['name']}")
            print(f"      Argumentos: {tool['args']}")
            print(f"      Resultado: {tool_results[i-1]['result'] if i-1 < len(tool_results) else 'N/A'}")
    else:
        print(f"\n❌ NENHUMA FERRAMENTA FOI UTILIZADA")
        print("   O agente respondeu apenas com conhecimento próprio")
    
    # Resposta final
    print(f"\n📝 RESPOSTA FINAL:")
    final_message = messages[-1]
    if isinstance(final_message, AIMessage) and final_message.content:
        print(f"   {final_message.content}")
    
    print("\n" + "="*70 + "\n")
    
    return len(tools_used) > 0, tools_used, tool_results

# ==================== WRAPPER PARA COMPATIBILIDADE ====================

class AgentExecutorWrapper:
    def __init__(self, agent):
        self.agent = agent
    
    def invoke(self, input_dict):
        user_input = input_dict["input"]
        result = self.agent.invoke({
            "messages": [{"role": "user", "content": user_input}]
        })
        
        # Extrai resposta final
        final_message = result["messages"][-1].content
        
        return {"output": final_message}

# ==================== MAIN ====================

if __name__ == "__main__":
    agent_executor = AgentExecutorWrapper(agent)
    
    # Teste 1: Pergunta que usa tool
    print("\n🧪 TESTE 1: Pergunta sobre a capital do Irã")
    print("=" * 70)
    
    result1 = agent.invoke({
        "messages": [{"role": "user", "content": "What is the capital of Iran?"}]
    })
    
    tool_foi_usada, tools_info, results = verify_tool_usage(result1)
    output1 = agent_executor.invoke({"input": "What is the capital of Iran?"})
    print(f"📌 Resposta Final (Wrapper): {output1['output']}\n")
    
    # Teste 2: Pergunta que usa tool
    print("\n🧪 TESTE 2: Pergunta sobre operação matemática")
    print("=" * 70)
    
    result2 = agent.invoke({
        "messages": [{"role": "user", "content": "How much is 10 + 10?"}]
    })
    
    tool_foi_usada, tools_info, results = verify_tool_usage(result2)
    output2 = agent_executor.invoke({"input": "How much is 10 + 10?"})
    print(f"📌 Resposta Final (Wrapper): {output2['output']}\n")
    
    # Teste 3: Pergunta que pode não usar tool
    print("\n🧪 TESTE 3: Pergunta genérica")
    print("=" * 70)
    
    result3 = agent.invoke({
        "messages": [{"role": "user", "content": "What is machine learning?"}]
    })
    
    tool_foi_usada, tools_info, results = verify_tool_usage(result3)
    output3 = agent_executor.invoke({"input": "What is machine learning?"})
    print(f"📌 Resposta Final (Wrapper): {output3['output']}\n")