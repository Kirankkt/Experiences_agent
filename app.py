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
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents import Tool, initialize_agent, AgentType
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

##############################################################################
#                         HELPER FUNCTIONS                                   #
##############################################################################

def is_valid_url(url, retries=2, delay=1):
    """
    Validate URL with multiple retry attempts.
    """
    for attempt in range(retries):
        try:
            response = requests.head(url, allow_redirects=True, timeout=3)
            if response.status_code == 200:
                return True
        except requests.RequestException as e:
            logging.warning(f"URL attempt {attempt + 1} failed: {e}")
        time.sleep(delay)
    return False

def validate_and_normalize_link(link):
    """
    Try to return a valid link. If invalid, return the original text.
    Skip validation for certain aggregator or marketplace domains if desired.
    """
    skip_validation_domains = ["olx.in", "quikr.com"]
    link = link.strip()
    if any(domain in link for domain in skip_validation_domains):
        return link  # Skip validation for these domains
    if link.startswith(('http://', 'https://')):
        return link if is_valid_url(link) else link
    potential_link = 'https://' + link
    return potential_link if is_valid_url(potential_link) else link

def extract_listings_from_output(output):
    """
    Extract relevant data from the agent's output.
    """
    try:
        results_text = str(output)
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
    Use GPT to generate a creative and engaging story about the listing.
    Uses openai.ChatCompletion.create(...) for openai>=1.0.0
    """
    prompt = f"""
    You are a local content creator in Trivandrum. Write a captivating and creative announcement for a new {category[:-1]} in Trivandrum. 
    Highlight its unique features, ambiance, specialties, and why locals and visitors should visit. 
    Include a friendly invitation to check it out.

    Name: {listing['Name']}
    Description: {listing['Description']}
    Link: {listing['Link']}

    Story:
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=200
        )
        story = response.choices[0].message.content.strip()
        logging.info(f"Generated story for {listing['Name']}: {story}")
        return story
    except Exception as e:
        logging.error(f"Story generation error for {listing['Name']}: {e}")
        return "Could not generate story at this time."

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

def perform_search(category):
    """
    Perform a web search using Serper API and return the raw results.
    We refine the query to skip aggregator sites and "top 10" style pages.
    """
    # We specifically tell it: official websites, skip aggregator keywords
    base_queries = {
        "Restaurants": "restaurants in Trivandrum official site OR 'official website' -tripadvisor -zomato -justdial -wanderlog -yelp -timesofindia -thehindu -blog -list -ranking -youtube -facebook -instagram -quora -reddit -twitter",
        "Boutiques": "boutiques in Trivandrum official site OR 'official website' -tripadvisor -zomato -justdial -wanderlog -yelp -blog -list -ranking -youtube -facebook -instagram -quora -reddit -twitter",
        "Experiences": "experiences in Trivandrum official site OR 'official website' -tripadvisor -zomato -justdial -wanderlog -yelp -blog -list -ranking -youtube -facebook -instagram -quora -reddit -twitter"
    }
    
    # If the category is not in the base_queries, do a fallback
    search_query = base_queries.get(
        category, 
        f"{category} in Trivandrum official site OR 'official website' -tripadvisor -zomato -justdial -wanderlog -yelp -blog -list -ranking -youtube -facebook -instagram -quora -reddit -twitter"
    )

    logging.info(f"Performing search with query: {search_query}")
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        logging.error("Serper API key not found.")
        return ""

    try:
        response = requests.get(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key},
            params={"q": search_query, "num": 20}  # 20 results
        )
        response.raise_for_status()
        results = response.json()
        logging.info(f"Raw search results: {results}")
        return results
    except Exception as e:
        logging.error(f"Search API error: {e}")
        return ""

def parse_search_results(results):
    """
    Parse Serper API results to extract listings.
    Exclude aggregator or big directories for the final display.
    """
    # Expand the aggregator or unwanted domains you want to skip
    unwanted_domains = [
        "reddit.com", "quora.com", "instagram.com", "facebook.com", "twitter.com", 
        "tripadvisor.com", "tripadvisor.in", "zomato.com", "justdial.com", 
        "wanderlog.com", "youtube.com", "yelp.com", "blogspot.com", "wordpress.com",
        "thehindu.com", "timesofindia.com", "indiatimes.com"
    ]

    try:
        organic_results = results.get("organic", [])
    except Exception as e:
        logging.error(f"Parsing error: {e}")
        return []

    listings = []
    initial_count = len(organic_results)
    for item in organic_results:
        try:
            title = item.get("title", "").strip()
            link = item.get("link", "").strip()
            snippet = item.get("snippet", "").strip()

            if not (title and link and snippet):
                continue

            # If domain is in the unwanted list, skip
            # e.g. if the link is https://www.tripadvisor.com/....
            skip_it = any(domain in link for domain in unwanted_domains)
            if skip_it:
                continue

            # Optionally also skip if the title/snippet looks like "top 10" or "best X"
            # That might help further reduce aggregator style results
            aggregator_indicators = ["top 10", "top 5", "best 10", "list", "ranking", "guide", "guide to"]
            if any(phrase.lower() in title.lower() for phrase in aggregator_indicators):
                continue

            listing = {
                'Name': title,
                'Link': validate_and_normalize_link(link),
                'Description': snippet
            }
            listings.append(listing)
        except Exception as e:
            logging.warning(f"Error processing item: {e}")

    logging.info(f"Total organic results fetched: {initial_count}")
    logging.info(f"Parsed {len(listings)} listings after filtering.")
    return listings

def create_vector_store(listings):
    """
    Create a FAISS vector store from listings.
    """
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        texts = [f"{listing['Name']} {listing['Description']}" for listing in listings]
        vector_store = FAISS.from_texts(texts, embeddings)
        return vector_store
    except ImportError:
        logging.error("Could not import tiktoken python package. Needed for OpenAIEmbeddings. pip install tiktoken.")
        return None
    except Exception as e:
        logging.error(f"Vector store creation error: {e}")
        return None

def create_agent(vector_store):
    """
    Create a LangChain agent with tools.
    """
    try:
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=2500
        )

        # Define tools
        tools = [
            Tool(
                name="Search",
                func=lambda query: perform_search(query),
                description="Searches the web for information."
            )
        ]

        # Initialize agent
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        return agent
    except Exception as e:
        logging.error(f"Agent creation error: {e}")
        return None

def run_search(category):
    """
    Orchestrate the search process.
    """
    try:
        logging.info("Starting search")
        raw_results = perform_search(category)
        if not raw_results:
            logging.warning("No raw results obtained from search.")
            return None, None

        listings = parse_search_results(raw_results)
        if not listings:
            logging.warning("No listings found in search results.")
            return None, None

        vector_store = create_vector_store(listings)
        if not vector_store:
            logging.warning("Failed to create vector store.")
            return None, None

        agent = create_agent(vector_store)
        if not agent:
            logging.warning("Failed to create agent.")
            return None, None

        # Generate stories for each listing
        for listing in listings:
            story = generate_story(listing, category)
            listing['Story'] = story

        df, excel_data = save_to_excel(listings)
        return df, excel_data
    except Exception as e:
        logging.error(f"Search orchestration error: {e}")
        return None, None

##############################################################################
#                               STREAMLIT APP                                #
##############################################################################

def main():
    # Make sure to set the API key for openai
    openai.api_key = os.getenv("OPENAI_API_KEY")

    st.set_page_config(page_title="Trivandrum Experiences Finder", layout="wide")
    st.title("🌟 Trivandrum Experiences Finder")

    st.sidebar.header("🔍 Search Parameters")
    category = st.sidebar.selectbox("Select Category", ["Restaurants", "Boutiques", "Experiences"])
    search_params = {'category': category}

    if st.sidebar.button("Search"):
        with st.spinner("Searching for the best listings..."):
            df, excel_data = run_search(search_params['category'])
            if df is not None and not df.empty:
                st.success(f"✅ Found {len(df)} listings!")
                
                for idx, row in df.iterrows():
                    with st.expander(f"**{row['Name']}**"):
                        st.markdown(f"**Description:** {row['Description']}")
                        st.markdown(f"**[Visit Website]({row['Link']})**")
                        st.markdown(f"**Story:** {row['Story']}")

                # Provide option to download all listings
                st.download_button(
                    label="Download Listings",
                    data=excel_data,
                    file_name='trivandrum_listings.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            else:
                st.warning("No results found. Try adjusting your search by selecting a different category or using broader search terms.")


if __name__ == "__main__":
    main()
