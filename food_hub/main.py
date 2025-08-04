import json
from agents import Runner, Agent, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool, GuardrailFunctionOutput, InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered, RunContextWrapper, TResponseInputItem, input_guardrail, output_guardrail
from dotenv import load_dotenv
import os

import chainlit as cl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pydantic import BaseModel

load_dotenv()
set_tracing_disabled(True)


# ========== Model Setup ==========
gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)





@function_tool(name_override="menu")
async def menu():
    """get menu and their prices"""
    with open("menu.json") as f:
        return json.load(f)


@function_tool(name_override="deal")
async def deals():
    """get deals and their price"""
    with open("deal.json") as f:
        return json.load(f)


cancel = Agent(
    name="cancel_agent",
    instructions="""if the user cancel their order you can call the tool `cancel` fucntion.
     make sure user's cancel id and name correct and than you can cancel their order.""",
     model=model
)


class MenuDealsOutputType(BaseModel):
    response: str
    is_menu_show_related: float
    is_deals_show_related: float


menu_deals = Agent(
    name="Menu_&_Deals_Agent",
    instructions="""
    You are a Menu & Deals decision agent. 
    Based on the user's message:
    - Set `is_menu_show_related` to 1.0 ONLY if they clearly want to see the full menu image (keywords like 'menu dikhao', 'menu bhejo', 'show menu or manu').
    - Set `is_deals_show_related` to 1.0 ONLY if they clearly want to see the full deals image (keywords like 'deal dikhao', 'deals bhejo').
    - If they ask for price (e.g., 'deal 1 price', 'menu item price'), then set both to 0.0, and also using `menu' or `deals` function tool and get their price and show him.
    - `response` should contain your natural language reply if not showing an image.
    - make sure always be you can call `menu` and `deals` fucntions call for you better performance.
    """,
    tools=[menu, deals],  # price tools
    model=model,
    output_type=MenuDealsOutputType
)

triage_agent = Agent(
    name="Main_FoodHub_Restaurant_Agent",
    instructions="""You are a helpful of FoodHub Restaurant assistant that can users help.
    if the user's query about menu, deals, items price you can handoff menu_&_deals agent.
    if the user concel their order you can handoff cancel_agent agent 
    make sure always respond the user only FoodHub Restaurant nothing else""",
    handoffs=[menu_deals, cancel],
    model=model,
    tools=[],

)

@cl.on_chat_start
async def on_chat():
    cl.user_session.set("history", [])
    await cl.Message(content="Hi, There this is FoodHub Restaurant",).send()


@cl.on_message
async def on_message(message: cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    # Run AI with schema
    result = await Runner.run(menu_deals, input=history)
    output = result.final_output  # This will be MenuDealsOutputType

    # --- Menu image ---
    if output.is_menu_show_related >= 0.8:
        msg = await cl.Message(content="📋 Here's our menu:").send()
        await cl.Image(name="Menu", path="menu.jpg", display="inline").send(for_id=msg.id)
        return

    # --- Deals image ---
    if output.is_deals_show_related >= 0.8:
        msg = await cl.Message(content="🔥 Here are our deals:").send()
        await cl.Image(name="Deals", path="deals.jpg", display="inline").send(for_id=msg.id)
        return

    # --- Otherwise send normal AI response ---
    await cl.Message(content=output.response).send()
