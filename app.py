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

def parse_search_results(results):
    """
    Parse Serper API results to extract listings.
    Exclude aggregator or big directories for the final display
    but keep it less aggressive for restaurants.
    """
    # Slightly loosened approach to skip fewer aggregator sites
    unwanted_domains = [
        "reddit.com", "quora.com", "instagram.com", "facebook.com", "twitter.com",
        "wanderlog.com", "youtube.com", "yelp.com",
        "thehindu.com", "timesofindia.com", "indiatimes.com"
        # Notice we commented out tripadvisor, zomato, justdial, etc.
        # because sometimes these sites contain official site references.
        # "tripadvisor.com", "tripadvisor.in", "zomato.com", "justdial.com",
    ]

    aggregator_indicators = ["top 10", "top 5", "best 10", "list", "ranking", "guide", "guide to"]

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
            if any(domain in link for domain in unwanted_domains):
                continue

            # Optionally also skip if the title/snippet looks like "top 10" or "best X"
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
        logging.error("Could not import tiktoken python package. Needed for OpenAIEmbeddings. `pip install tiktoken`.")
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

def perform_search(query):
    """
    Actually call the Serper dev API with the given query.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        logging.error("Serper API key not found.")
        return {}

    try:
        response = requests.get(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key},
            params={"q": query, "num": 20}  # 20 results
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Search API error: {e}")
        return {}

def orchestrate_search(category):
    """
    Orchestrates searching with a more refined approach.
    1) Perform an initial search with moderate filters.
    2) If results are too few, do a fallback search with minimal filters.
    """
    # Step 1: Less restrictive queries for Restaurants, moderate for others
    base_queries = {
        # For restaurants, we allow synonyms and only a few negative keywords
        "Restaurants": "(restaurants OR eateries OR cafes OR 'fine dining') in Trivandrum official site OR 'official website' -tripadvisor -wanderlog -list -ranking -youtube -facebook -instagram -quora -reddit -twitter",
        "Boutiques": "boutiques in Trivandrum official site OR 'official website' -tripadvisor -zomato -justdial -wanderlog -yelp -blog -list -ranking -youtube -facebook -instagram -quora -reddit -twitter",
        "Experiences": "experiences in Trivandrum official site OR 'official website' -tripadvisor -zomato -justdial -wanderlog -yelp -blog -list -ranking -youtube -facebook -instagram -quora -reddit -twitter"
    }

    initial_query = base_queries.get(category, f"{category} in Trivandrum")
    logging.info(f"Performing initial search with query: {initial_query}")
    raw_results = perform_search(initial_query)
    listings = parse_search_results(raw_results)

    # Fallback if listings < 5
    if len(listings) < 5:
        logging.info(f"Fewer than 5 listings found ({len(listings)}). Relaxing constraints...")
        fallback_query = f"{category} in Trivandrum"
        fallback_results = perform_search(fallback_query)
        fallback_listings = parse_search_results(fallback_results)

        # Merge the two sets of listings (unique by Link or something similar)
        combined_links = set([l['Link'] for l in listings])
        for item in fallback_listings:
            if item['Link'] not in combined_links:
                listings.append(item)
                combined_links.add(item['Link'])

    if listings:
        logging.info(f"Total combined listings after fallback: {len(listings)}")
    else:
        logging.warning("No listings found even after fallback.")

    return listings

def run_search(category):
    """
    High-level function that:
    1) Orchestrates the search for the given category.
    2) Creates vector store and agent.
    3) Generates GPT-based stories.
    4) Saves to Excel.
    """
    try:
        logging.info("Starting orchestrated search...")
        listings = orchestrate_search(category)
        if not listings:
            logging.warning("No listings found. Exiting.")
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
    # Set the API key for openai
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
                st.warning("No results found. Try relaxing your search or using a broader category.")


if __name__ == "__main__":
    main()
