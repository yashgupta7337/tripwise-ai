from crewai import Agent
from textwrap import dedent
from langchain_openai import ChatOpenAI

from tools.search_tools import SearchTools
from tools.calculator_tools import CalculatorTools


class TravelAgents:
    def __init__(self):
        self.OpenAIGPT35 = ChatOpenAI(
            model_name="gpt-3.5-turbo", temperature=0.7)
        self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

    def expert_travel_agent(self):
        return Agent(
            role="Expert Travel Agent",
            backstory=dedent(
                f"""Expert in travel planning and logistics. 
                I have decades of expereince making travel iteneraries."""),
            goal=dedent(f"""
                        Create a 7-day travel itinerary with detailed per-day plans,
                        include budget, packing suggestions, and safety tips.
                        """),
            tools=[
                SearchTools.search_internet,
                CalculatorTools.calculate
            ],
            verbose=True,
            llm=self.OpenAIGPT4,
        )

    def city_selection_expert(self):
        return Agent(
            role="City Selection Expert",
            backstory=dedent(
                f"""Expert at analyzing travel data to pick ideal destinations"""),
            goal=dedent(
                f"""Select the best cities based on weather, season, prices, and traveler interests"""),
            tools=[SearchTools.search_internet],
            verbose=True,
            llm=self.OpenAIGPT4,
        )

    def local_tour_guide(self):
        return Agent(
            role="Local Tour Guide",
            backstory=dedent(f"""
                             As a passionate local, I provide authentic experiences 
                             that showcase the hidden gems and cultural highlights of my city. 
                             My goal is to create memorable adventures that connect travelers with the local way of life.
                             """),
            goal=dedent(f"""
                        Provide the BEST insights about the selected city by creating a detailed guide that includes:
                        1. Key attractions and must-visit landmarks
                        2. Local customs and cultural insights
                        3. Special events during the travel period
                        4. Hidden gems and local favorites
                        5. Recommended restaurants and food spots
                        6. Transportation tips
                        7. Estimated costs for activities
                        8. Safety tips and cultural etiquette
                        
                        The guide should be well-organized and practical for travelers.
                        """),
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4,
        )