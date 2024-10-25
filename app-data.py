import streamlit as st
import pandas as pd
import duckdb
import os
import pickle
import logging
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(model_name="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))

# Define paths for storing data
DATA_DIR = "data"
VIEWS_FILE = os.path.join(DATA_DIR, "views.pkl")
DB_FILE = os.path.join(DATA_DIR, "app_database.duckdb")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Set page config
st.set_page_config(page_title="Data Playground", layout="wide")

# Load saved views
def load_views():
    views = {}
    if os.path.exists(VIEWS_FILE):
        with open(VIEWS_FILE, 'rb') as f:
            views = pickle.load(f)
            logging.info(f"Loaded views: {views}")
    return views

# Save views
def save_views(views):
    with open(VIEWS_FILE, 'wb') as f:
        pickle.dump(views, f)
    logging.info(f"Saved views: {views}")

# Define the save_data_source function before using it
def save_data_source(file_name, df):
    logging.info(f"Saving data source: {file_name}")
    logging.info(f"Columns before saving: {df.columns.tolist()}")
    logging.info(f"Shape before saving: {df.shape}")

    st.session_state.data_sources[file_name] = df
    
    # Use DuckDB's efficient import method
    st.session_state.db_connection.execute(f"""
        CREATE OR REPLACE TABLE "{file_name}" AS SELECT * FROM df
    """)
    
    # Verify the saved data
    verify_df = st.session_state.db_connection.execute(f'SELECT * FROM "{file_name}" LIMIT 5').df()
    logging.info(f"Columns after saving: {verify_df.columns.tolist()}")
    logging.info(f"Shape after saving: {verify_df.shape}")
    
    if not df.columns.equals(verify_df.columns):
        logging.error(f"Column mismatch for {file_name}!")
        logging.error(f"Original columns: {df.columns.tolist()}")
        logging.error(f"Saved columns: {verify_df.columns.tolist()}")
    
    logging.info(f"Data source saved: {file_name}")

def pandas_type_to_duckdb_type(pandas_type):
    if pandas_type in ['object', 'string', 'string[pyarrow]']:
        return 'VARCHAR'
    elif pandas_type in ['int64', 'int32', 'int16', 'int8']:
        return 'BIGINT'
    elif pandas_type in ['float64', 'float32']:
        return 'DOUBLE'
    elif pandas_type == 'bool':
        return 'BOOLEAN'
    elif pandas_type.startswith('datetime'):
        return 'TIMESTAMP'
    else:
        return 'VARCHAR'  # Default to VARCHAR for unknown types

# Load data sources from DuckDB
def load_data_sources():
    data_sources = {}
    try:
        tables = st.session_state.db_connection.execute("SHOW TABLES").fetchall()
        for table in tables:
            table_name = table[0]
            if table_name != 'dummy':  # Exclude the dummy table
                try:
                    # Check if it's a table, not a view
                    is_table = st.session_state.db_connection.execute(f"SELECT type FROM sqlite_master WHERE name='{table_name}'").fetchone()[0] == 'table'
                    if is_table:
                        df = st.session_state.db_connection.execute(f'SELECT * FROM "{table_name}" LIMIT 5').df()
                        data_sources[table_name] = df
                        logging.info(f"Successfully loaded table: {table_name}")
                except Exception as e:
                    logging.error(f"Error loading table {table_name}: {str(e)}")
    except Exception as e:
        logging.error(f"Error listing tables: {str(e)}")
    return data_sources

# Define this function early in your script, before it's used
def delete_data_source(file_name):
    try:
        # Remove from session state
        if file_name in st.session_state.data_sources:
            del st.session_state.data_sources[file_name]
        
        # Drop the table from DuckDB
        st.session_state.db_connection.execute(f'DROP TABLE IF EXISTS "{file_name}"')
        
        # Remove the CSV file if it exists
        csv_path = os.path.join(DATA_DIR, f"{file_name}.csv")
        if os.path.exists(csv_path):
            os.remove(csv_path)
        
        logging.info(f"Data source deleted: {file_name}")
        st.success(f"Data source '{file_name}' has been deleted.")
    except Exception as e:
        logging.error(f"Error deleting data source {file_name}: {str(e)}")
        st.error(f"Error deleting data source '{file_name}': {str(e)}")

# Initialize database connection
if 'db_connection' not in st.session_state:
    st.session_state.db_connection = duckdb.connect(DB_FILE)
    st.session_state.db_connection.execute("SET TimeZone='America/Sao_Paulo';")
    st.session_state.db_connection.execute("SET memory_limit='4GB';")

# Initialize session state variables
if 'data_sources' not in st.session_state:
    st.session_state.data_sources = load_data_sources()
    logging.info(f"Loaded data sources: {list(st.session_state.data_sources.keys())}")

if 'views' not in st.session_state:
    st.session_state.views = load_views()

# Recreate views
for view_name, query in st.session_state.views.items():
    safe_view_name = view_name.replace('-', '_')
    try:
        st.session_state.db_connection.execute(f'CREATE OR REPLACE VIEW "{safe_view_name}" AS {query}')
    except duckdb.Error as e:
        logging.error(f"Error recreating view {view_name}: {str(e)}")

if 'query_result' not in st.session_state:
    st.session_state.query_result = None

# Add this function
def execute_query(query):
    try:
        # Replace view names in the query with their safe versions
        for view_name in st.session_state.views:
            safe_view_name = view_name.replace('-', '_')
            query = query.replace(f"'{view_name}'", safe_view_name)
            query = query.replace(f'"{view_name}"', safe_view_name)
            query = query.replace(view_name, safe_view_name)
        
        result = st.session_state.db_connection.execute(query).df()
        return result, None
    except Exception as e:
        return None, str(e)

def save_view(view_name, query):
    if view_name:
        safe_view_name = view_name.replace('-', '_')
        st.session_state.views[view_name] = query
        try:
            st.session_state.db_connection.execute(f'CREATE OR REPLACE VIEW "{safe_view_name}" AS {query}')
            st.success(f"View '{view_name}' saved successfully!")
            save_views(st.session_state.views)
            logging.info(f"View saved: {view_name}")
        except duckdb.Error as e:
            st.error(f"DuckDB error: {str(e)}")
            logging.error(f"DuckDB error: {str(e)}")
    else:
        st.warning("Please enter a name for the view.")

st.title("Data Playground")

# Define load_csv function before using it
def load_csv(file_path):
    try:
        # Get the file size
        file_size = os.path.getsize(file_path)
        
        if file_size > 200 * 1024 * 1024:  # If file is larger than 200MB
            st.sidebar.warning("Large file detected. Loading may take some time.")
            
            # Use chunking to read large files
            chunks = pd.read_csv(file_path, chunksize=100000)  # Adjust chunk size as needed
            df = pd.concat(chunks, ignore_index=True)
        else:
            # For smaller files, read normally
            df = pd.read_csv(file_path)
        
        logging.info(f"CSV loaded successfully.")
        logging.info(f"Columns: {df.columns.tolist()}")
        logging.info(f"Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV: {str(e)}")
        raise

# Sidebar
st.sidebar.header("Data Sources")

# Add instructions
st.sidebar.markdown("""
### Instructions:
1. Enter the name of your CSV file (including .csv extension) in the text input below.
2. Click 'Load File' to preview and save the file as a data source.
3. Use the main panel to query your data or ask questions in AI mode.
""")

# Add a refresh button
if st.sidebar.button("Refresh Data Sources"):
    st.session_state.data_sources = load_data_sources()
    st.rerun()

# File name input
file_name = st.sidebar.text_input("Enter the name of your CSV file (e.g., data.csv):")
load_button = st.sidebar.button("Load File")

if load_button:
    if file_name:
        try:
            # Construct the full path
            file_path = os.path.join(os.path.dirname(__file__), file_name)
            
            if not os.path.exists(file_path):
                st.sidebar.error(f"File '{file_name}' not found in the app directory.")
            else:
                df = load_csv(file_path)
                st.sidebar.write(f"Preview of {file_name}:")
                st.sidebar.dataframe(df.head())
                st.sidebar.write(f"Columns: {df.columns.tolist()}")
                st.sidebar.write(f"Shape: {df.shape}")
                
                # Move the save button outside of the load_button condition
                st.session_state.current_df = df
                st.session_state.current_file_name = file_name.split('.')[0]
        except Exception as e:
            st.sidebar.error(f"Error loading {file_name}: {str(e)}")
            logging.error(f"Error loading CSV {file_name}: {str(e)}")
    else:
        st.sidebar.warning("Please enter a file name.")

# Add this outside the load_button condition
if 'current_df' in st.session_state and 'current_file_name' in st.session_state:
    if st.sidebar.button(f"Save {st.session_state.current_file_name}"):
        save_data_source(st.session_state.current_file_name, st.session_state.current_df)
        st.sidebar.success(f"Loaded: {st.session_state.current_file_name}")
        logging.info(f"CSV loaded and saved: {st.session_state.current_file_name}")
        st.rerun()

# Display loaded data sources
st.sidebar.subheader("Raw")
for source in list(st.session_state.data_sources.keys()):
    col1, col2 = st.sidebar.columns([3, 1])
    col1.text(source)
    if col2.button(f"Delete", key=f"delete_source_{source}"):
        delete_data_source(source)
        st.rerun()

# Display saved views
st.sidebar.subheader("Views")
for view_name in list(st.session_state.views.keys()):
    col1, col2 = st.sidebar.columns([3, 1])
    col1.text(view_name)
    if col2.button(f"Delete", key=f"delete_view_{view_name}"):
        safe_view_name = view_name.replace('-', '_')
        del st.session_state.views[view_name]
        st.session_state.db_connection.execute(f'DROP VIEW IF EXISTS "{safe_view_name}"')
        save_views(st.session_state.views)
        logging.info(f"View deleted: {view_name}")
        st.rerun()

# Callback functions
def execute_query_callback():
    query = st.session_state.query_input
    try:
        result, error = execute_query(query)
        if error:
            st.session_state.query_error = error
            st.session_state.query_result = None
            logging.error(f"Query execution error: {error}")
        else:
            st.session_state.query_result = result
            st.session_state.query_error = None
            logging.info("Query executed successfully")
    except duckdb.Error as e:
        st.session_state.query_error = str(e)
        st.session_state.query_result = None
        logging.error(f"DuckDB error: {str(e)}")

# Main panel
ai_mode = st.toggle("AI Mode", value=False)

if ai_mode:
    selected_source = st.selectbox("Select a data source to query:", list(st.session_state.data_sources.keys()))
    query = st.text_area("Ask a question about your data:", height=150, key="query_input")
    
    if st.button("Ask AI"):
        if query and selected_source:
            try:
                # Generate data summary
                df = st.session_state.data_sources[selected_source]
                data_summary = f"Data source: {selected_source}\n"
                data_summary += f"Columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}\n"
                data_summary += f"Number of rows: {len(df)}\n"
                data_summary += "Sample data (first 3 rows, up to 5 columns):\n"
                data_summary += df.head(3).iloc[:, :5].to_string()

                # First AI interaction: Generate SQL query
                sql_generation_prompt = f"""
                Based on the following data summary and user question, generate a SQL query to answer the question.
                Only return the SQL query, without any additional text or explanation.
                IMPORTANT: Always use double quotes around table names and column names in your SQL query.
                The table name to use is: "{selected_source}"

                Data Summary:
                {data_summary}

                User Question: {query}

                SQL Query:
                """
                sql_response = llm.invoke(sql_generation_prompt)
                generated_sql = sql_response.content.strip()

                # Display the generated SQL
                st.write("Generated SQL Query:")
                st.code(generated_sql, language="sql")

                # Check if the generated SQL is valid
                if not generated_sql.lower().startswith("select"):
                    raise ValueError("The generated SQL query is not valid. It should start with SELECT.")

                # Execute the generated SQL query
                result, error = execute_query(generated_sql)

                if error:
                    st.error(f"Error executing query: {error}")
                    st.write("The AI generated an invalid SQL query. Please try rephrasing your question.")
                else:
                    # Limit the result size
                    result_preview = result.head(5).to_string()
                    total_rows = len(result)
                    
                    # Second AI interaction: Interpret the results
                    interpretation_prompt = f"""
                    Based on the user's question and the SQL query results, provide a concise interpretation.

                    User Question: {query}
                    SQL Query: {generated_sql}
                    Query Results (first 5 rows):
                    {result_preview}

                    Total number of rows in the result: {total_rows}

                    Interpretation:
                    """
                    interpretation_response = llm.invoke(interpretation_prompt)
                    
                    st.write("AI Response:")
                    st.write(interpretation_response.content)
                    
                    st.write("Query Result:")
                    st.dataframe(result)
                    
                    # Add download button for query result
                    if not result.empty:
                        csv = result.to_csv(index=False)
                        st.download_button(
                            label="Download query result as CSV",
                            data=csv,
                            file_name="ai_query_result.csv",
                            mime="text/csv",
                        )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please select a data source and enter a question.")
else:
    query = st.text_area("Enter your SQL query:", height=150, key="query_input", on_change=execute_query_callback)

    if st.button("Run Query"):
        if query:
            try:
                result, error = execute_query(query)
                if error:
                    st.error(f"Error executing query: {error}")
                else:
                    st.write("Query Result:")
                    st.dataframe(result)
                    
                    # Add download button for query result
                    if not result.empty:
                        csv = result.to_csv(index=False)
                        st.download_button(
                            label="Download query result as CSV",
                            data=csv,
                            file_name="query_result.csv",
                            mime="text/csv",
                        )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a query.")

# Display data source preview
st.subheader("Data Source Preview")
selected_source = st.selectbox("Select a data source or view:", 
                               list(st.session_state.data_sources.keys()) + list(st.session_state.views.keys()))

if selected_source in st.session_state.data_sources:
    st.dataframe(st.session_state.data_sources[selected_source].head())
elif selected_source in st.session_state.views:
    view_query = st.session_state.views[selected_source]
    safe_view_name = selected_source.replace('-', '_')
    result, error = execute_query(f"SELECT * FROM {safe_view_name}")
    if error:
        st.error(f"Error in view '{selected_source}': {error}")
        logging.error(f"Error in view '{selected_source}': {error}")
    else:
        st.dataframe(result.head())

logging.info("App execution completed")
