import streamlit as st
import os
import asyncio
import nest_asyncio

# Apply nest_asyncio for async support
nest_asyncio.apply()

# Set page config
st.set_page_config(
    page_title="Basic Pipeline Test",
    page_icon="🤖",
    layout="wide"
)

st.title("🧪 Basic Pipeline Test")
st.write("Testing if research agent passes data to writer agent")

# Initialize session state
if 'agents_initialized' not in st.session_state:
    st.session_state.agents_initialized = False
if 'google_api_key' not in st.session_state:
    st.session_state.google_api_key = ""
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = ""
if 'runner' not in st.session_state:
    st.session_state.runner = None

# Sidebar for API keys
with st.sidebar:
    st.header("API Configuration")
    
    google_api_key = st.text_input(
        "Google Gemini API Key:",
        type="password",
        value=st.session_state.google_api_key
    )
    
    groq_api_key = st.text_input(
        "Groq API Key:",
        type="password",
        value=st.session_state.groq_api_key
    )
    
    if st.button("Initialize Agents"):
        if google_api_key and groq_api_key:
            with st.spinner("Initializing..."):
                try:
                    # Set API keys
                    os.environ["GOOGLE_API_KEY"] = google_api_key
                    os.environ["GROQ_API_KEY"] = groq_api_key
                    
                    # Import inside try block
                    from google.adk.agents import Agent, SequentialAgent
                    from google.adk.models.google_llm import Gemini
                    from google.adk.models.lite_llm import LiteLlm
                    from google.adk.runners import InMemoryRunner
                    from google.adk.tools.google_search_tool import GoogleSearchTool
                    
                    # SIMPLIFIED Research Agent
                    research_agent = Agent(
                        name="research_agent",
                        model=Gemini(model="gemini-2.5-flash-lite"),
                        description="Researcher that finds information",
                        instruction="""You are a research agent. When given a topic:
1. Search for information about it
2. Find 3 key points
3. Store your findings with: context.state['research_findings'] = [your text here]

Your output format:
TOPIC: [topic]
KEY POINTS:
1. Point 1
2. Point 2
3. Point 3

Remember to store with: context.state['research_findings'] = """,
                        tools=[GoogleSearchTool(bypass_multi_tools_limit=True)],
                    )
                    
                    # SIMPLIFIED Writer Agent
                    writer_agent = Agent(
                        name="writer_agent",
                        model=LiteLlm(model="groq/llama-3.3-70b-versatile"),
                        description="Writer that creates content from research",
                        instruction="""You are a writer agent. You will receive research findings.
Your task is to write a simple summary based ONLY on the research in context.state['research_findings'].

Write: "Based on research: [summary of what you see in the research]"

IMPORTANT: Only use what's in context.state['research_findings']. Don't make up information.""",
                    )
                    
                    # Create pipeline
                    pipeline_agent = SequentialAgent(
                        name="pipeline",
                        sub_agents=[research_agent, writer_agent],
                        description="Test pipeline"
                    )
                    
                    # Create runner
                    runner = InMemoryRunner(agent=pipeline_agent)
                    
                    # Store
                    st.session_state.google_api_key = google_api_key
                    st.session_state.groq_api_key = groq_api_key
                    st.session_state.runner = runner
                    st.session_state.agents_initialized = True
                    
                    st.success("Agents initialized!")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Enter both API keys")

# Main area
st.header("Test Pipeline")

if not st.session_state.get('agents_initialized', False):
    st.info("Please initialize agents first (sidebar)")
else:
    topic = st.text_input("Enter a topic to research:", "AI in agriculture")
    
    if st.button("Run Pipeline"):
        with st.spinner("Running pipeline..."):
            try:
                # Simple async execution
                async def run_test():
                    return await st.session_state.runner.run_debug(topic)
                
                # Run with new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    response_events = loop.run_until_complete(run_test())
                finally:
                    loop.close()
                
                # Display ALL events for debugging
                st.subheader("🔍 RAW EVENTS (for debugging):")
                for i, event in enumerate(response_events):
                    with st.expander(f"Event {i}: {type(event).__name__}"):
                        st.write("Event object:", event)
                        if hasattr(event, 'content'):
                            st.write("Content:", event.content)
                        if hasattr(event, '__dict__'):
                            st.write("Attributes:", event.__dict__)
                
                # Try to extract specific outputs
                st.subheader("📊 EXTRACTED OUTPUTS:")
                
                # Look for research agent output
                research_text = None
                writer_text = None
                
                for event in response_events:
                    if hasattr(event, 'content'):
                        content_str = str(event.content)
                        # Check if it's from research agent
                        if 'research_agent' in str(event).lower() or 'TOPIC:' in content_str:
                            research_text = content_str
                        # Check if it's from writer agent
                        elif 'writer_agent' in str(event).lower() or 'Based on research' in content_str:
                            writer_text = content_str
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Research Agent Output:")
                    if research_text:
                        st.text(research_text[:1000] + "..." if len(research_text) > 1000 else research_text)
                    else:
                        st.warning("No research output found")
                
                with col2:
                    st.markdown("### Writer Agent Output:")
                    if writer_text:
                        st.text(writer_text)
                    else:
                        st.warning("No writer output found")
                        # Try alternative extraction
                        st.info("Trying alternative extraction...")
                        for event in response_events:
                            if hasattr(event, '__dict__'):
                                st.write(f"Event keys: {list(event.__dict__.keys())}")
                                if 'parts' in event.__dict__:
                                    st.write("Parts found!")
                
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                import traceback
                st.code(traceback.format_exc())

# Debug section
with st.expander("Debug Info"):
    st.write("Agents initialized:", st.session_state.get('agents_initialized', False))
    st.write("Runner exists:", st.session_state.get('runner') is not None)
    st.write("Google API:", "Set" if st.session_state.google_api_key else "Not set")
    st.write("Groq API:", "Set" if st.session_state.groq_api_key else "Not set")
