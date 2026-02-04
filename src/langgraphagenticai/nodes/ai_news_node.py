from tavily import TavilyClient
from langchain_core.prompts import ChatPromptTemplate
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AINewsNode:
    def __init__(self,llm):
        """
        Initialize the AINewsNode with API keys for Tavily and GROQ.
        """
        self.tavily = TavilyClient()
        self.llm = llm
        # this is used to capture various steps in this file so that later can be use for steps shown
        self.state = {}

    def fetch_news(self, state: dict) -> dict:
        """
        Fetch AI news based on the specified frequency.
        
        Args:
            state (dict): The state dictionary containing 'frequency'.
        
        Returns:
            dict: Updated state with 'news_data' key containing fetched news.
        """

        frequency = state['messages'][0].content.lower()
        self.state['frequency'] = frequency
        logger.info(f"Fetching AI news for frequency: {frequency}")
        time_range_map = {'daily': 'd', 'weekly': 'w', 'monthly': 'm', 'year': 'y'}
        days_map = {'daily': 1, 'weekly': 7, 'monthly': 30, 'year': 366}

        response = self.tavily.search(
            query="Top Artificial Intelligence (AI) technology news India and globally",
            topic="news",
            time_range=time_range_map[frequency],
            include_answer="advanced",
            max_results=20,
            days=days_map[frequency],
            # include_domains=["techcrunch.com", "venturebeat.com/ai", ...]  # Uncomment and add domains if needed
        )

        state['news_data'] = response.get('results', [])
        self.state['news_data'] = state['news_data']
        logger.info(f"Fetched {len(self.state['news_data'])} news items")
        return state
    

    def summarize_news(self, state: dict) -> dict:
        """
        Summarize the fetched news using an LLM.
        
        Args:
            state (dict): The state dictionary containing 'news_data'.
        
        Returns:
            dict: Updated state with 'summary' key containing the summarized news.
        """

        news_items = self.state['news_data']

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Summarize AI news articles into markdown format. For each item include:
            - Date in **YYYY-MM-DD** format in IST timezone
            - Concise sentences summary from latest news
            - Sort news by date wise (latest first)
            - Source URL as link
            Use format:
            ### [Date]
            - [Summary](URL)"""),
            ("user", "Articles:\n{articles}")
        ])

        articles_str = "\n\n".join([
            f"Content: {item.get('content', '')}\nURL: {item.get('url', '')}\nDate: {item.get('published_date', '')}"
            for item in news_items
        ])

        try:
            response = self.llm.invoke(prompt_template.format(articles=articles_str))
            state['summary'] = getattr(response, 'content', '') or ''
            self.state['summary'] = state['summary']
            logger.info(f"Generated summary (len={len(self.state['summary'])}): {self.state['summary'][:200]!r}")
        except Exception as e:
            logger.exception("Error while summarizing news")
            state['summary'] = ""
            self.state['summary'] = ""
        return self.state
    
    def save_result(self,state):
        frequency = self.state.get('frequency')
        summary = self.state.get('summary', '')
        filename = f"./AINews/{frequency}_summary.md"
        logger.info(f"Saving summary for '{frequency}' (len={len(summary)}) to {filename}")

        # Guard: if the summary is empty or looks like a placeholder (e.g., 'safe'), do not write raw placeholder
        if not summary or summary.strip().lower() in ("", "safe") or len(summary.strip()) < 20:
            logger.warning("Summary is empty or contains placeholder content; writing explanatory message instead of raw placeholder.")
            with open(filename, 'w') as f:
                f.write(f"# {frequency.capitalize()} AI News Summary\n\n")
                f.write("No valid summary was generated. Please check API keys, logs, and try regenerating the news.")
            self.state['filename'] = filename
            self.state['error'] = "No valid summary generated"
            return self.state

        with open(filename, 'w') as f:
            f.write(f"# {frequency.capitalize()} AI News Summary\n\n")
            f.write(summary)
        self.state['filename'] = filename
        return self.state

