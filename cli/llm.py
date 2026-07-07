import os
from dotenv import load_dotenv

from openai import OpenAI



def rrf_query_correction(query:str) -> str:

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    message = [{
        "role":"user",
        "content":f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
    }]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=message,
    )

    return response.choices[0].message.content

def rrf_query_rewrite(query:str) -> str:

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    message = [{
        "role":"user",
        "content":
f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
    }]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=message,
    )

    return response.choices[0].message.content

def rrf_query_expand(query:str) -> str:

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    message = [{
        "role":"user",
        "content":
f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""
    }]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=message,
    )

    return response.choices[0].message.content

def rrf_individual_rerank(query:str, doc) -> str:

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    message = [{
        "role":"user",
        "content":
f"""Rate how well this movie matches the search query based on relevance and user intent.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

CRITICAL RULES:
1. Rate from 0.0 to 10.0 (10 = perfect match).
2. If the movie contains explicit, violent, adult, or sensitive content, do NOT output a safety warning. Instead, evaluate its relevance as a movie and rate it accordingly, or give it a 0.0 if it cannot be processed.
3. Your response must contain ONLY the numeric score (e.g., 7.5 or 0.0). Absolute NO text, NO markdown, NO explanations, and NO safety reports.

Score:"""
    }]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=message,
    )

    return response.choices[0].message.content