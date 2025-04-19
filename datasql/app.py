import streamlit as st
st.set_page_config(page_title="Chat with SQL DB", page_icon='💻')

from pathlib import Path
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine, text
import sqlite3
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download all required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('omw-1.4')  # Open Multilingual Wordnet
    except Exception as e:
        st.error(f"Error downloading NLTK data: {str(e)}")

# Download NLTK data
download_nltk_data()

st.title('SQL Database Assistant')

# Sidebar for database configuration
st.sidebar.header("Database Configuration")

# Database type selection
db_type = st.sidebar.radio(
    "Select Database Type",
    ["SQLite", "MySQL"]
)

if db_type == "SQLite":
    db_uri = 'USE_LOCALDB'
    st.sidebar.info("Using local SQLite database: pr_report.db")
else:
    db_uri = 'USE_MYSQL'
    mysql_host = st.sidebar.text_input('MySQL Host')
    mysql_user = st.sidebar.text_input('MySQL Username')
    mysql_pass = st.sidebar.text_input('MySQL Password', type='password')
    mysql_db = st.sidebar.text_input('MySQL Database Name')

# Groq API Key
api_key = st.sidebar.text_input(
    label='Groq API Key', 
    type='password',
    help="Get your API key from https://console.groq.com/keys"
)

# Validate API key
if not api_key:
    st.warning("Please enter your Groq API Key in the sidebar")
    st.stop()

# Test API key before proceeding
try:
    llm = ChatGroq(groq_api_key=api_key, model_name='Llama3-8b-8192', streaming=True)
    # Test the API key with a simple query
    llm.invoke("test")
except Exception as e:
    if "invalid_api_key" in str(e).lower():
        st.error("Invalid Groq API Key. Please check your API key and try again.")
        st.info("You can get your API key from https://console.groq.com/keys")
        st.stop()
    else:
        st.error(f"Error connecting to Groq API: {str(e)}")
        st.stop()

# Main content area
st.header("Database Operations")

# Operation selection
operation = st.selectbox(
    "Choose an operation",
    [
        "View Database Schema",
        "Create New Table",
        "Insert Data",
        "Query Data",
        "Update Data",
        "Delete Data",
        "Show Table Contents",
        "Drop Table",
        "Custom SQL Query"
    ]
)

# Initialize database connection
if not api_key:
    st.warning("Please enter your Groq API Key in the sidebar")
    st.stop()

llm = ChatGroq(groq_api_key=api_key, model_name='Llama3-8b-8192', streaming=True)

@st.cache_resource(ttl='2h')
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_pass=None, mysql_db=None):
    if db_uri == 'USE_LOCALDB':
        dbfilepath = (Path(__file__).parent / 'pr_report.db').absolute()
        return SQLDatabase(create_engine(f'sqlite:///{dbfilepath}'))
    elif db_uri == 'USE_MYSQL':
        if not (mysql_host and mysql_user and mysql_pass and mysql_db):
            st.error('Please provide the MySQL database information')
            st.stop()
        return SQLDatabase(create_engine(f'mysql+mysqlconnector://{mysql_user}:{mysql_pass}@{mysql_host}/{mysql_db}'))

# Configure database based on selection
if db_uri == 'USE_MYSQL':
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_pass, mysql_db)
else:
    db = configure_db(db_uri)

# Create a more robust SQL toolkit
toolkit = SQLDatabaseToolkit(
    db=db,
    llm=llm,
    verbose=True
)

# Create a more robust SQL agent
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    handle_parsing_errors=True,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    max_iterations=3
)

# Operation-specific UI and logic
if operation == "View Database Schema":
    st.subheader("Database Schema")
    try:
        # First get list of all tables
        with db._engine.connect() as connection:
            tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
            result = connection.execute(text(tables_query))
            tables = [row[0] for row in result.fetchall()]
            
            if not tables:
                st.info("No tables found in the database.")
            else:
                st.write("Available tables:")
                for table in tables:
                    st.write(f"- {table}")
                
                # Let user select a table to view its schema
                selected_table = st.selectbox("Select a table to view its schema:", tables)
                
                if selected_table:
                    # Get schema for selected table
                    schema_query = f"PRAGMA table_info({selected_table})"
                    result = connection.execute(text(schema_query))
                    columns = result.fetchall()
                    
                    if columns:
                        st.write(f"\nSchema for table '{selected_table}':")
                        # Create a formatted table display
                        col_names = ["Column Name", "Data Type", "Nullable", "Default Value", "Primary Key"]
                        data = [[col[1], col[2], "No" if col[3] else "Yes", col[4] or "None", "Yes" if col[5] else "No"] for col in columns]
                        st.table(data)
                    else:
                        st.info(f"No schema information available for table '{selected_table}'")
    except Exception as e:
        st.error(f"Error fetching schema: {str(e)}")

elif operation == "Create New Table":
    st.subheader("Create New Table")
    table_name = st.text_input("Table Name")
    columns = st.text_area("Column Definitions (SQL format)", 
                          "column1 datatype1,\ncolumn2 datatype2")
    if st.button("Create Table"):
        try:
            # Create SQL query
            query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
            
            # Execute query directly using SQLAlchemy
            with st.spinner("Creating table..."):
                with db._engine.connect() as connection:
                    connection.execute(text(query))
                    connection.commit()
                st.success(f"Table '{table_name}' created successfully!")
        except Exception as e:
            st.error(f"Error creating table: {str(e)}")

elif operation == "Insert Data":
    st.subheader("Insert Data")
    table_name = st.text_input("Table Name")
    columns = st.text_input("Columns (comma-separated)", "column1, column2")
    
    # Add helper text for values format
    st.info("For string values, use quotes. Example: 1, 'John', 'Hello World'")
    values = st.text_area("Values (comma-separated)", "1, 'value1', 'value2'")
    
    if st.button("Insert Data"):
        try:
            # Clean up the values string to ensure proper SQL formatting
            values = values.strip()
            if not any(c in values for c in ["'", '"']):
                # If no quotes are present, add them around string values
                values = values.split(',')
                values = [f"'{v.strip()}'" if not v.strip().replace('.', '').isdigit() else v.strip() 
                         for v in values]
                values = ', '.join(values)
            
            # First check if table exists
            check_table_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
            with db._engine.connect() as connection:
                result = connection.execute(text(check_table_query))
                table_exists = result.fetchone() is not None
                
                if not table_exists:
                    # Create table with appropriate columns
                    column_defs = []
                    for col in columns.split(','):
                        col = col.strip()
                        # Try to infer data type from the first value
                        val = values.split(',')[len(column_defs)].strip()
                        if val.replace('.', '').isdigit():
                            col_def = f"{col} INTEGER"
                        else:
                            col_def = f"{col} TEXT"
                        column_defs.append(col_def)
                    
                    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(column_defs)})"
                    connection.execute(text(create_table_query))
                    connection.commit()
                    st.info(f"Table '{table_name}' was created automatically.")
            
            # Now insert the data
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"
            
            # Show the query for debugging
            st.code(query, language="sql")
            
            with st.spinner("Inserting data..."):
                with db._engine.connect() as connection:
                    connection.execute(text(query))
                    connection.commit()
                st.success("Data inserted successfully!")
        except Exception as e:
            st.error(f"Error inserting data: {str(e)}")
            st.info("Make sure to use quotes around string values. Example: 1, 'John', 'Hello World'")

elif operation == "Query Data":
    st.subheader("Query Data")
    query = st.text_area("Enter your query", "SELECT * FROM table_name")
    if st.button("Execute Query"):
        try:
            with st.spinner("Executing query..."):
                response = agent.run(f"Execute this SQL query: {query}")
                st.write(response)
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")

elif operation == "Update Data":
    st.subheader("Update Data")
    table_name = st.text_input("Table Name")
    set_clause = st.text_input("SET clause", "column = value")
    where_clause = st.text_input("WHERE clause", "condition")
    if st.button("Update Data"):
        try:
            query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
            with st.spinner("Updating data..."):
                with db._engine.connect() as connection:
                    connection.execute(text(query))
                    connection.commit()
                st.success("Data updated successfully!")
        except Exception as e:
            st.error(f"Error updating data: {str(e)}")

elif operation == "Delete Data":
    st.subheader("Delete Data")
    table_name = st.text_input("Table Name")
    where_clause = st.text_input("WHERE clause", "condition")
    if st.button("Delete Data"):
        try:
            query = f"DELETE FROM {table_name} WHERE {where_clause}"
            with st.spinner("Deleting data..."):
                with db._engine.connect() as connection:
                    connection.execute(text(query))
                    connection.commit()
                st.success("Data deleted successfully!")
        except Exception as e:
            st.error(f"Error deleting data: {str(e)}")

elif operation == "Show Table Contents":
    st.subheader("Show Table Contents")
    table_name = st.text_input("Table Name")
    if st.button("Show Contents"):
        try:
            query = f"SELECT * FROM {table_name}"
            with st.spinner("Fetching data..."):
                with db._engine.connect() as connection:
                    result = connection.execute(text(query))
                    data = result.fetchall()
                    if data:
                        st.write(data)
                    else:
                        st.info("No data found in the table.")
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")

elif operation == "Drop Table":
    st.subheader("Drop Table")
    table_name = st.text_input("Table Name to Drop")
    
    if st.button("Drop Table"):
        try:
            # First check if table exists
            check_table_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
            with db._engine.connect() as connection:
                result = connection.execute(text(check_table_query))
                table_exists = result.fetchone() is not None
                
                if not table_exists:
                    st.error(f"Table '{table_name}' does not exist.")
                else:
                    # Drop the table
                    drop_query = f"DROP TABLE IF EXISTS {table_name}"
                    with st.spinner(f"Dropping table '{table_name}'..."):
                        connection.execute(text(drop_query))
                        connection.commit()
                    st.success(f"Table '{table_name}' has been dropped successfully!")
        except Exception as e:
            st.error(f"Error dropping table: {str(e)}")

elif operation == "Custom SQL Query":
    st.subheader("Custom SQL Query")
    query = st.text_area("Enter your custom SQL query")
    if st.button("Execute"):
        try:
            with st.spinner("Executing query..."):
                response = agent.run(f"Execute this SQL query: {query}")
                st.write(response)
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")

# Chat interface for natural language queries
st.header("Natural Language Query")
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role': 'assistant', 'content': 'How can I help you with the database?'}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

user_query = st.chat_input(placeholder='Ask anything about the database')

if user_query:
    st.session_state.messages.append({'role': 'user', 'content': user_query})
    st.chat_message('user').write(user_query)

    with st.chat_message('assistant'):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        try:
            # Add context to help the agent understand the database
            context = f"""You are a SQL expert. The database is {db_type}. 
            Please help with this query: {user_query}
            If you need to see the schema, use the list_tables_sql tool first."""
            
            try:
                response = agent.run(context, callbacks=[streamlit_callback])
                st.session_state.messages.append({'role': 'assistant', 'content': response})
                st.write(response)
            except Exception as e:
                if "invalid_api_key" in str(e).lower():
                    error_message = "There was an issue with the API key. Please check your Groq API key in the sidebar."
                    st.error(error_message)
                    st.info("You can get your API key from https://console.groq.com/keys")
                else:
                    error_message = f"I encountered an error while processing your query. Please try rephrasing your question or use one of the specific operations above. Error: {str(e)}"
                    st.error(error_message)
                st.session_state.messages.append({'role': 'assistant', 'content': error_message})
        except Exception as e:
            error_message = f"An unexpected error occurred. Please try again or use one of the specific operations above. Error: {str(e)}"
            st.session_state.messages.append({'role': 'assistant', 'content': error_message})
            st.error(error_message)
