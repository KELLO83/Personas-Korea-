import sys
import logging

sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO)

from src.rag.llm import create_llm

def test_nvidia_llm():
    print("NVIDIA LLM (DeepSeek-V4-Pro) Init...")
    try:
        llm = create_llm(temperature=0.7)
        
        prompt = "안녕하세요! 당신은 어떤 모델인가요? 짧고 친절하게 한국어로 대답해주세요."
        print(f"User Prompt: {prompt}")
        print("Waiting for response...\n")
        
        response = llm.invoke(prompt)
        
        print("="*40)
        print("[NVIDIA DeepSeek-V4-Pro Response]")
        print("="*40)
        print(response.content)
        print("="*40)
        print("\nLLM Test Success!")
        
    except Exception as e:
        print(f"\nLLM Test Failed: {e}")

if __name__ == "__main__":
    test_nvidia_llm()
