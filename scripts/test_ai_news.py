#!/usr/bin/env python3
"""
Simple test for AINewsNode using a fake LLM and sample news data.
Run: python scripts/test_ai_news.py
"""

import logging
import os
from types import SimpleNamespace

from src.langgraphagenticai.nodes.ai_news_node import AINewsNode

os.makedirs("AINews", exist_ok=True)
logging.basicConfig(level=logging.INFO)

class FakeLLM:
    def invoke(self, prompt):
        # mimic a response object with a 'content' attribute
        return SimpleNamespace(content=(
            "### 2026-02-04\n"
            "- Breakthrough in AI model efficiency (http://example.com)\n"
        ))

print("Running AINewsNode test with FakeLLM...")
node = AINewsNode(llm=FakeLLM())
node.state['frequency'] = 'daily'
node.state['news_data'] = [
    {
        'content': 'Significant improvement in AI model efficiency reported',
        'url': 'http://example.com',
        'published_date': '2026-02-04'
    }
]

# Summarize and save - should create a valid markdown file
node.summarize_news({})
result_state = node.save_result({})
print('\nSaved file:', result_state.get('filename'))
with open(result_state.get('filename'), 'r') as f:
    print('\n--- File contents ---\n')
    print(f.read())

# Now simulate placeholder/empty summary to test guard
print('\nTesting guard for placeholder/empty summaries...')
node.state['summary'] = 'safe'
node.state['frequency'] = 'daily'
result_state2 = node.save_result({})
print('\nSaved file after placeholder:', result_state2.get('filename'))
with open(result_state2.get('filename'), 'r') as f:
    print('\n--- File contents after placeholder guard ---\n')
    print(f.read())

print('\nTest script finished.')