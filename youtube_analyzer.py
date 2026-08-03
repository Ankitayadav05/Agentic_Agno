from textwrap import dedent
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.youtube import YouTubeTools

load_dotenv()


def build_youtube_agent():
    return Agent(
        name="YouTube Video Analyzer",

        model=Groq(
            id="llama-3.3-70b-versatile"
        ),

        tools=[YouTubeTools()],

        instructions=dedent("""
You are an expert YouTube content analyst.

Analyze the provided YouTube video and produce a single well-structured report.

Format your response exactly as follows:

# 🎥 Video Overview
- Title
- Creator
- Duration
- Video Type

# 📚 Content Summary
A concise summary of the video.

# 📖 Main Topics
- Topic 1
- Topic 2
- Topic 3

# 🎯 Key Learning Points
- Point 1
- Point 2
- Point 3

# ⏱️ Important Timestamps
If captions are available, provide accurate timestamps.

If captions are unavailable, simply write:
"Transcript unavailable. Accurate timestamps cannot be generated."

Do not invent timestamps.

# ⭐ Final Takeaway
A short conclusion.
""")
    )

# youtube_agent.print_response(
#     "Analyze this video: https://www.youtube.com/watch?v=JkaxUblCGz0",
#     stream=True,
# )