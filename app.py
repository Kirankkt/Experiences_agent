import os
import sys
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Set environment variables from Streamlit secrets
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("OpenAI API key not found in secrets.")

if "SERPER_API_KEY" in st.secrets:
    os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
else:
    st.error("Serper Dev API key not found in secrets.")

# Import necessary libraries
import re
import logging
import pandas as pd
import openai
import requests
import time
from crewai import Crew, Task, Agent
from crewai_tools import SerperDevTool
from langchain.chat_models import ChatOpenAI
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("custom_agent_output.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def is_valid_url(url, retries=3, delay=2):
    """
    Validate URL with multiple retry attempts.
    """
    for attempt in range(retries):
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            if response.status_code == 200:
                return True
        except requests.RequestException as e:
            logging.warning(f"URL attempt {attempt + 1} failed: {e}")
        time.sleep(delay)
    return False

def validate_and_normalize_link(link):
    """
    Try to return a valid link. If invalid, return the original text.
    """
    link = link.strip()
    if link.startswith(('http://', 'https://')):
        return link if is_valid_url(link) else link
    potential_link = 'https://' + link
    return potential_link if is_valid_url(potential_link) else link

def extract_listings_from_output(output):
    """
    Extract relevant data from CrewAI output.
    """
    try:
        results_text = str(getattr(output, 'raw', getattr(output, 'result', str(output))))
    except Exception as e:
        logging.error(f"Output extraction error: {e}")
        return []

    pattern = r'Title:\s*(.*?)\s*Link:\s*(.*?)\s*Snippet:\s*(.*?)\s*(?=Title:|$)'
    matches = re.findall(pattern, results_text, re.DOTALL | re.MULTILINE)

    listings = []
    for match in matches:
        try:
            listing = {
                'Name': match[0].strip(),
                'Link': validate_and_normalize_link(match[1].strip()),
                'Description': match[2].strip()
            }
            listings.append(listing)
        except Exception as e:
            logging.warning(f"Error processing listing: {e}")
    return listings

def generate_story(listing, category):
    """
    Use GPT to generate a creative story about the listing.
    """
    prompt = f"""
    Write a creative announcement for a new {category} in Trivandrum:
    Name: {listing['Name']}
    Description: {listing['Description']}
    Link: {listing['Link']}
    """
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=300
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logging.error(f"Story generation error: {e}")
        return ""

def save_to_excel(listings, filename='trivandrum_listings.xlsx'):
    """
    Save listings to an Excel file.
    """
    try:
        df = pd.DataFrame(listings)
        output = BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        return df, output.getvalue()
    except Exception as e:
        logging.error(f"Excel save error: {e}")
        return None, None

def create_agent(search_params):
    """
    Create an Agent for searching listings.
    """
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    serper_api_key = os.environ.get('SERPER_API_KEY')

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=2500
    )

    search = SerperDevTool(api_key=serper_api_key)

    goal = f"Find new {search_params['category']} in Trivandrum."
    task_description = f"""
    Search for new {search_params['category']} in Trivandrum. Ensure results include name, link, and description.
    Format as:
    'Title: [Name]\nLink: [Link]\nSnippet: [Description]'
    """

    agent = Agent(
        llm=llm,
        role="Content Finder",
        goal=goal,
        tools=[search],
        verbose=True
    )

    task = Task(
        description=task_description,
        expected_output="A list of at least 10 relevant listings.",
        agent=agent
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )

    return crew

def run_search(search_params):
    """
    Run the agent-based search and process results.
    """
    try:
        logging.info("Starting search")
        crew = create_agent(search_params)
        results = crew.kickoff()
        listings = extract_listings_from_output(results)
        if listings:
            df, excel_data = save_to_excel(listings)
            return df, excel_data
        else:
            logging.warning("No listings found.")
            return None, None
    except Exception as e:
        logging.error(f"Search error: {e}")
        return None, None

def main():
    st.set_page_config(page_title="Trivandrum Experiences Finder", layout="wide")
    st.title("🌟 Trivandrum Experiences Finder")

    st.sidebar.header("🔍 Search Parameters")
    category = st.sidebar.selectbox("Select Category", ["Restaurants", "Boutiques", "Experiences"])
    search_params = {'category': category}

    if st.sidebar.button("Search"):
        with st.spinner("Searching for the best listings..."):
            df, excel_data = run_search(search_params)
            if df is not None:
                st.success(f"✅ Found {len(df)} listings!")
                st.dataframe(df)
                st.download_button(
                    label="Download Listings",
                    data=excel_data,
                    file_name='listings.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            else:
                st.warning("No results found. Try adjusting your search.")

if __name__ == "__main__":
    main()
