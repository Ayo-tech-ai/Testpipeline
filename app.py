import streamlit as st
import os
import asyncio
import nest_asyncio

# Apply nest_asyncio for async support
nest_asyncio.apply()

# Set page config
st.set_page_config(
    page_title="Multi-Platform Pipeline Test",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Multi-Platform Pipeline Test")
st.write("Testing research agent with 3 platform writers")

# Initialize session state
if 'agents_initialized' not in st.session_state:
    st.session_state.agents_initialized = False
if 'google_api_key' not in st.session_state:
    st.session_state.google_api_key = ""
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = ""
if 'runner' not in st.session_state:
    st.session_state.runner = None
if 'platform_outputs' not in st.session_state:
    st.session_state.platform_outputs = {}

# ============================================
# TEXT EXTRACTION FUNCTION
# ============================================
def extract_text(event):
    """Extract text content from ADK event objects."""
    if not hasattr(event, "content"):
        return ""
    
    content = event.content
    
    # Case 1: Direct text attribute
    if hasattr(content, "text") and content.text:
        return content.text
    
    # Case 2: It's iterable (list, tuple, etc.)
    if hasattr(content, "__iter__"):
        texts = []
        for item in content:
            # Check if it's a Part-like object with text
            if hasattr(item, "text") and item.text:
                texts.append(item.text)
            # Or if it's a string/direct text
            elif isinstance(item, str):
                texts.append(item)
        if texts:  # Only return if we found text
            return "\n".join(texts)
    
    # Case 3: Fallback to string representation
    return str(content)

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
    
    if st.button("Initialize All Agents"):
        if google_api_key and groq_api_key:
            with st.spinner("Initializing 4 agents..."):
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
                    
                    # ======================
                    # 1. RESEARCH AGENT
                    # ======================
                    research_agent = Agent(
                        name="research_agent",
                        model=Gemini(model="gemini-2.5-flash-lite"),
                        description="Researcher that finds current information",
                        instruction="""You are a research agent. When given a topic:

1. Search for recent information (last 1-2 years)
2. Find 5 key insights with examples
3. Include practical applications
4. MUST store findings with: context.state['research_findings'] = [your research text here]

Format:
TOPIC: [topic]
KEY INSIGHTS:
1. [Insight 1]
2. [Insight 2]
APPLICATIONS:
[Application 1]
STATISTICS:
[Statistic if available]

Remember to store with: context.state['research_findings'] = """,
                        tools=[GoogleSearchTool(bypass_multi_tools_limit=True)],
                    )

                    # ======================
                    # 2. LINKEDIN AGENT
                    # ======================
                    linkedin_agent = Agent(
                        name="linkedin_agent",
                        model=LiteLlm(model="groq/llama-3.3-70b-versatile"),
                        description="Writer for professional LinkedIn articles",
                        instruction="""You are a LinkedIn content writer. Read context.state['research_findings'] and create a LinkedIn post.

STRUCTURE:
1. Engaging headline with emoji
2. Hook with surprising fact
3. 3-4 key insights from research
4. Practical applications
5. Thought-provoking question
6. Hashtags including #9jaAI_Farmer

TONE: Professional, insightful
LENGTH: 300-400 words
ONLY use information from context.state['research_findings'].

Write only the LinkedIn post.""",
                    )

                    # ======================
                    # 3. FACEBOOK AGENT
                    # ======================
                    facebook_agent = Agent(
                        name="facebook_agent",
                        model=LiteLlm(model="groq/llama-3.3-70b-versatile"),
                        description="Writer for engaging Facebook posts",
                        instruction="""You are a Facebook content writer. Read context.state['research_findings'] and create a Facebook post.

Facebook posts should be:
- Shorter than LinkedIn (1500 characters max)
- More conversational
- Include emojis 🎯✨
- End with question for comments
- Include "Read more" link placeholder
- Suggest an image idea

TONE: Friendly, engaging, community-focused
ONLY use information from context.state['research_findings'].

Write only the Facebook post with image suggestion.""",
                    )

                    # ======================
                    # 4. WHATSAPP AGENT
                    # ======================
                    whatsapp_agent = Agent(
                        name="whatsapp_agent",
                        model=LiteLlm(model="groq/llama-3.3-70b-versatile"),
                        description="Writer for WhatsApp messages",
                        instruction="""You are creating a WhatsApp message. Read context.state['research_findings'] and create a WhatsApp broadcast.

WhatsApp messages should be:
- Very short (300 characters max)
- Conversational "Hey friends/family!"
- Key takeaway from research
- Include 2 link placeholders: [LINK_TO_LINKEDIN] and [LINK_TO_FACEBOOK]
- Personal call-to-action
- Use 2-3 relevant emojis

TONE: Personal, excited, share-with-friends
ONLY use information from context.state['research_findings'].

Write only the WhatsApp message.""",
                    )

                    # ======================
                    # CREATE PIPELINE
                    # ======================
                    # Sequential: Research first, then all writers
                    pipeline_agent = SequentialAgent(
                        name="multi_platform_pipeline",
                        sub_agents=[research_agent, linkedin_agent, facebook_agent, whatsapp_agent],
                        description="Research → LinkedIn → Facebook → WhatsApp"
                    )
                    
                    # Create runner
                    runner = InMemoryRunner(agent=pipeline_agent)
                    
                    # Store
                    st.session_state.google_api_key = google_api_key
                    st.session_state.groq_api_key = groq_api_key
                    st.session_state.runner = runner
                    st.session_state.agents_initialized = True
                    
                    st.success("✅ 4 Agents initialized! (Research + 3 Platforms)")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Enter both API keys")

# Main area
st.header("Test Multi-Platform Pipeline")

if not st.session_state.get('agents_initialized', False):
    st.info("Please initialize agents first (sidebar)")
else:
    topic = st.text_input("Enter a topic to research:", "IoT in smart farming")
    
    if st.button("Run Multi-Platform Pipeline"):
        with st.spinner("Researching and creating 3 platform posts..."):
            try:
                # Clear previous outputs
                st.session_state.platform_outputs = {}
                
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
                        st.write("Agent:", getattr(event, 'agent_name', 'Unknown'))
                        if hasattr(event, 'content'):
                            # Use the extract_text function here too
                            content_preview = extract_text(event)
                            st.write("Content preview:", content_preview[:200] + "..." if len(content_preview) > 200 else content_preview)
                
                # ======================
                # EXTRACT PLATFORM OUTPUTS
                # ======================
                st.subheader("📱 PLATFORM OUTPUTS:")
                
                research_output = ""
                linkedin_output = ""
                facebook_output = ""
                whatsapp_output = ""
                
                for event in response_events:
                    if hasattr(event, 'content'):
                        # USE THE EXTRACT_TEXT FUNCTION INSTEAD OF str(event.content)
                        content_str = extract_text(event)
                        agent_name = str(event).lower()
                        
                        # Research agent
                        if 'research_agent' in agent_name or 'TOPIC:' in content_str:
                            research_output = content_str
                        
                        # LinkedIn agent
                        elif 'linkedin_agent' in agent_name or '#9jaai_farmer' in content_str.lower():
                            linkedin_output = content_str
                            st.session_state.platform_outputs['linkedin'] = content_str
                        
                        # Facebook agent
                        elif 'facebook_agent' in agent_name or 'read more' in content_str.lower():
                            facebook_output = content_str
                            st.session_state.platform_outputs['facebook'] = content_str
                        
                        # WhatsApp agent
                        elif 'whatsapp_agent' in agent_name or '[link_' in content_str.lower():
                            whatsapp_output = content_str
                            st.session_state.platform_outputs['whatsapp'] = content_str
                
                # ======================
                # DISPLAY RESULTS
                # ======================
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 🌐 LinkedIn Post")
                    if linkedin_output:
                        st.text_area("LinkedIn", linkedin_output, height=300)
                        st.caption(f"Length: {len(linkedin_output)} chars")
                    else:
                        st.warning("No LinkedIn output")
                
                with col2:
                    st.markdown("### 👍 Facebook Post")
                    if facebook_output:
                        st.text_area("Facebook", facebook_output, height=250)
                        st.caption(f"Length: {len(facebook_output)} chars")
                    else:
                        st.warning("No Facebook output")
                
                with col3:
                    st.markdown("### 💬 WhatsApp Message")
                    if whatsapp_output:
                        st.text_area("WhatsApp", whatsapp_output, height=150)
                        st.caption(f"Length: {len(whatsapp_output)} chars")
                    else:
                        st.warning("No WhatsApp output")
                
                # ======================
                # RESEARCH FINDINGS
                # ======================
                st.markdown("---")
                with st.expander("🔍 Research Findings (Used by all platforms)"):
                    if research_output:
                        # Clean up the context.state line for display
                        clean_research = research_output
                        if 'context.state[' in clean_research:
                            clean_research = clean_research.split('context.state[')[0]
                        st.text(clean_research[:1500] + "..." if len(clean_research) > 1500 else clean_research)
                    else:
                        st.warning("No research output found")
                
                # ======================
                # QUICK STATS
                # ======================
                st.markdown("---")
                st.subheader("📊 Pipeline Statistics")
                
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                
                with stats_col1:
                    st.metric("Research Complete", "✅" if research_output else "❌")
                
                with stats_col2:
                    st.metric("LinkedIn", "✅" if linkedin_output else "❌")
                
                with stats_col3:
                    st.metric("Facebook", "✅" if facebook_output else "❌")
                
                with stats_col4:
                    st.metric("WhatsApp", "✅" if whatsapp_output else "❌")
                
                # Store for later use
                st.session_state.last_research = research_output
                
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                import traceback
                st.code(traceback.format_exc())

# Debug section
with st.expander("🔧 Debug Info"):
    st.write("Agents initialized:", st.session_state.get('agents_initialized', False))
    st.write("Runner exists:", st.session_state.get('runner') is not None)
    st.write("Platform outputs:", list(st.session_state.platform_outputs.keys()))
    st.write("Google API:", "✅ Set" if st.session_state.google_api_key else "❌ Not set")
    st.write("Groq API:", "✅ Set" if st.session_state.groq_api_key else "❌ Not set")
    
    if st.session_state.get('last_research'):
        st.write("Last research length:", len(st.session_state.last_research))
