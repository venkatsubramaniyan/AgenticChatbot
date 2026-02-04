#!/usr/bin/env python3
"""
Run the real AI News pipeline using Tavily and Groq.
Usage:
  - Ensure TAVILY_API_KEY and GROQ_API_KEY are set in the environment.
  - Optionally, set SELECTED_GROQ_MODEL env var to choose model (defaults to first option if not provided).

This script avoids printing API keys.
"""
import logging
import os
from types import SimpleNamespace

from langchain_core.messages import HumanMessage
from src.langgraphagenticai.nodes.ai_news_node import AINewsNode
from src.langgraphagenticai.LLMS.groqllm import GroqLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    tavily_key = os.environ.get('TAVILY_API_KEY')
    groq_key = os.environ.get('GROQ_API_KEY')
    selected_model = os.environ.get('SELECTED_GROQ_MODEL', 'llama-3.1-8b-instant')

    if not tavily_key or not groq_key:
        logger.error('TAVILY_API_KEY and/or GROQ_API_KEY missing in environment. Aborting test.')
        return

    # Instantiate LLM
    user_controls = {'GROQ_API_KEY': groq_key, 'selected_groq_model': selected_model}
    llm_config = GroqLLM(user_controls_input=user_controls)

    try:
        llm = llm_config.get_llm_model()
    except Exception as e:
        logger.exception('Failed to initialize Groq LLM')
        return

    node = AINewsNode(llm=llm)

    # Build test state
    state = {'messages':[HumanMessage(content='Daily')]}

    try:
        state = node.fetch_news(state)
        logger.info(f"Fetched {len(state.get('news_data', []))} news items")
    except Exception as e:
        logger.exception('Error fetching news')
        return

    try:
        node.summarize_news(state)
        summary_len = len(node.state.get('summary',''))
        logger.info(f"Summary length: {summary_len}")
    except Exception as e:
        logger.exception('Error summarizing news')
        return

    try:
        result = node.save_result(state)
        logger.info(f"Result file: {result.get('filename')}")
    except Exception as e:
        logger.exception('Error saving result')
        return

if __name__ == '__main__':
    main()