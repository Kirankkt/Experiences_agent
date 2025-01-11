import os
import sys
import streamlit as st
import warnings
import re
import logging
import pandas as pd
import openai
import requests
import time
from io import BytesIO
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents import Tool, initialize_agent, AgentType

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO or DEBUG as needed
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

def generate_story(listing, category):
    """
    Use GPT to generate a creative and engaging story about the listing.
    """
    # Prepare data, handle missing fields
    name = listing.get('Name', 'The Restaurant')
    address = listing.get('address', 'a prime location in Trivandrum')
    primary_cuisine = listing.get('Primary_Cuisine', 'delicious cuisine')
    website = listing.get('website', 'our website')
    customer_rating = listing.get('Customer_Rating', 'excellent')
    ranking_description = listing.get('Ranking_Description', '')
    
    # Handle missing website
    if not website or pd.isna(website):
        website = 'our website'
    
    # Handle missing Customer Rating
    if not customer_rating or pd.isna(customer_rating):
        customer_rating = 'excellent'
    
    prompt = f"""
    You are a creative content writer based in Trivandrum. Write a captivating and engaging announcement for the opening of a new {category[:-1].lower()} in Trivandrum.
    
    The announcement should include:
    - **Introduction**: Introduce the {category[:-1].lower()} with its name.
    - **Unique Features**: Highlight what makes it unique (e.g., special dishes, ambiance, services).
    - **Location**: Mention its location and any notable nearby landmarks.
    - **Invitation**: Encourage locals and visitors to visit.
    - **Call to Action**: Provide a link or contact information for more details.
    
    Here are the details:
    
    - **Name**: {name}
    - **Address**: {address}
    - **Primary Cuisine**: {primary_cuisine}
    - **Website**: {website}
    - **Customer Rating**: {customer_rating}
    - **Ranking Description**: {ranking_description}
    
    Story:
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a creative and engaging content writer specializing in crafting appealing restaurant announcements."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=300  # Increased tokens for more detailed stories
        )
        story = response.choices[0].message.content.strip()
        logging.info(f"Generated story for {name}: {story}")
        return story
    except Exception as e:
        logging.error(f"Story generation error for {name}: {e}")
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
    # Define base queries with more specific keywords
    base_queries = {
        "Restaurants": "intitle:restaurant in Trivandrum official site OR 'official website' menu OR dining OR 'book a table' -tripadvisor -zomato -justdial -wanderlog -yelp -timesofindia -thehindu -blog -list -ranking -youtube -facebook -instagram -quora -reddit -twitter",
        "Boutiques": "intitle:boutique in Trivandrum official site OR 'official website' -tripadvisor -zomato -justdial -wanderlog -yelp -blog -list -ranking -youtube -facebook -instagram -quora -reddit -twitter",
        "Experiences": "intitle:experience in Trivandrum official site OR 'official website' -tripadvisor -zomato -justdial -wanderlog -yelp -blog -list -ranking -youtube -facebook -instagram -quora -reddit -twitter"
    }

    # Fallback for categories not explicitly defined
    search_query = base_queries.get(
        category, 
        f"intitle:{category} in Trivandrum official site OR 'official website' -tripadvisor -zomato -justdial -wanderlog -yelp -blog -list -ranking -youtube -facebook -instagram -quora -reddit -twitter"
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
        logging.debug(f"Raw search results: {results}")
        return results
    except Exception as e:
        logging.error(f"Search API error: {e}")
        return ""

def parse_search_results(results, category):
    """
    Parse Serper API results to extract listings.
    Exclude aggregator or big directories for the final display.
    """
    # Expand the aggregator or unwanted domains you want to skip
    unwanted_domains = [
        "reddit.com", "quora.com", "instagram.com", "facebook.com", "twitter.com", 
        "tripadvisor.com", "tripadvisor.in", "zomato.com", "justdial.com", 
        "wanderlog.com", "youtube.com", "yelp.com", "blogspot.com", "wordpress.com",
        "thehindu.com", "timesofindia.com", "indiatimes.com",
        "sasa.kerala.gov.in", "startups.startupmission.in", "serper.dev"  # Added based on sample results
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
            skip_it = any(domain in link for domain in unwanted_domains)
            if skip_it:
                logging.debug(f"Skipping '{title}' as it belongs to an unwanted domain.")
                continue

            # Ensure 'restaurant' is in title or description for Restaurants category
            if category == "Restaurants":
                if 'restaurant' not in title.lower() and 'restaurant' not in snippet.lower():
                    logging.debug(f"Skipping '{title}' as it does not mention 'restaurant'.")
                    continue

            # Skip aggregator indicators
            aggregator_indicators = ["top 10", "top 5", "best 10", "list", "ranking", "guide", "guide to"]
            if any(phrase.lower() in title.lower() for phrase in aggregator_indicators):
                logging.debug(f"Skipping '{title}' as it matches aggregator indicators.")
                continue

            # Additional content-based filtering: ensure relevance
            if category == "Restaurants":
                if not re.search(r'\brestaurant\b', title, re.IGNORECASE) and not re.search(r'\brestaurant\b', snippet, re.IGNORECASE):
                    logging.debug(f"Skipping '{title}' as it lacks 'restaurant' keyword in content.")
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

def load_scraped_data(file_path):
    """
    Load scraped restaurant data from an Excel file.
    """
    try:
        df = pd.read_excel(file_path)
        logging.info(f"Loaded scraped data with {len(df)} entries.")
        return df
    except Exception as e:
        logging.error(f"Error loading scraped data: {e}")
        return pd.DataFrame()

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

        listings = parse_search_results(raw_results, category)
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

def generate_story_from_scraped(row):
    """
    Generate a story using data from a scraped restaurant row.
    """
    listing = row.to_dict()
    story = generate_story(listing, category="Restaurants")
    return story

##############################################################################
#                               STREAMLIT APP                                #
##############################################################################

def main():
    # Set OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    st.set_page_config(page_title="Trivandrum Experiences Finder", layout="wide")
    st.title("🌟 Trivandrum Experiences Finder")

    st.sidebar.header("🔍 Search Parameters")

    # Option to choose data source
    data_source = st.sidebar.selectbox("Choose Data Source", ["Live Search", "Browse Scraped Data"])

    if data_source == "Live Search":
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

    elif data_source == "Browse Scraped Data":
        # Load scraped data from the Excel file
        excel_file_path = "final_700_Restaraunt_Trivandrum.csv" # Ensure this file is in the same directory as the app
        scraped_df = load_scraped_data(excel_file_path)

        if not scraped_df.empty:
            # Clean and preprocess data
            scraped_df['Name'] = scraped_df['name'].fillna('Unknown Restaurant')
            scraped_df['address'] = scraped_df['address'].fillna('Address not available')
            scraped_df['Primary_Cuisine'] = scraped_df['Primary_Cuisine'].fillna('Various Cuisines')
            scraped_df['website'] = scraped_df['website'].fillna('N/A')
            scraped_df['Customer_Rating'] = scraped_df['Customer_Rating'].fillna('N/A')

            # Create a dropdown menu for restaurant selection
            restaurant_names = scraped_df['Name'].tolist()
            selected_restaurant = st.selectbox("Select a Restaurant", restaurant_names)

            if selected_restaurant:
                # Get the selected restaurant's details
                selected_row = scraped_df[scraped_df['Name'] == selected_restaurant].iloc[0]
                listing = selected_row.to_dict()

                # Generate story
                story = generate_story_from_scraped(selected_row)
                listing['Story'] = story

                # Display details and story
                st.subheader(f"**{listing['Name']}**")
                st.markdown(f"**Address:** {listing['address']}")
                st.markdown(f"**Primary Cuisine:** {listing['Primary_Cuisine']}")
                if listing['website'] != 'N/A':
                    st.markdown(f"**Website:** [Visit Website]({listing['website']})")
                else:
                    st.markdown(f"**Website:** Not available")
                st.markdown(f"**Customer Rating:** {listing['Customer_Rating']}")
                st.markdown(f"**Ranking Description:** {listing.get('Ranking_Description', 'N/A')}")
                st.markdown(f"**Story:** {listing['Story']}")

                # Optionally, provide download for the selected listing
                csv = pd.DataFrame([listing]).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Listing",
                    data=csv,
                    file_name=f"{listing['Name'].replace(' ', '_')}.csv",
                    mime='text/csv',
                )
        else:
            st.warning("Failed to load the scraped data. Please ensure the Excel file is correctly formatted and located in the app's directory.")

if __name__ == "__main__":
    main()
