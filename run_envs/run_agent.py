"""
simple llama agent for webshop
"""
import gym
from rich import print
from rich.markup import escape

from web_agent_site.envs import WebAgentSiteEnv
from web_agent_site.models import (
    HumanPolicy,
    RandomPolicy,
)
from web_agent_site.utils import DEBUG_PROD_SIZE

from rich.pretty import pprint
from pydantic import BaseModel, create_model
from typing import Literal, Union

import os
os.environ['TOGETHER_API_KEY'] = "e14bea8e4ea6fd032d9e30ba61d17e6eae1a619d4781ecd309a4e931b0b66aff"
os.environ['TAVILY_SEARCH_API_KEY'] = "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"

from llama_stack_client import LlamaStackClient 
client = LlamaStackClient(base_url="http://0.0.0.0:8321", provider_data = {"tavily_search_api_key": os.environ['TAVILY_SEARCH_API_KEY']})

model_id = "meta-llama/Llama-3.1-8B-Instruct"
# model_id = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

# print("Available models:")
# for m in client.models.list():
#     print(f"{m.identifier} (provider's alias: {m.provider_resource_id}) ")


system_prompt = """You are an autonomous shopping agent tasked with searching and purchasing items on a mock e-commerce website called WebShop.

You will be given an instruction of a product the user wants to buy. 

Your goal is to find a product that exactly matches with user instruction. This means that the product name, description, and attributes all need to match with user instruction. 

"""

"""User instruction may sometimes be less detailed than the product descriptions. It is acceptable to buy this product.

If the product page is missing details in the user instruction, this is probably not the right product and you may need to find a new product.
"""

action_instruction_prompt = """
The following list describes the purposes and effects of different actions you might be able to take. Not all of them may be available on the current page. There might be other actions not in the list on this page. Use your common sense knowledge to infer the purpose and effects of those actions.

- The "Next >" action will take you to the next product page.
- The "< Prev" action will take you to the previous product page.
- The "Back to Search" action will allow you to go to the search page so you can try another search query.
- Actions like "B7ZXBGDXF" and "B0957XW92M" are indices of products on the current page. If you click on them, you will be able to view product details and determine whether they match with the instruction.
- The "Description" action will allow you to see the product description.
- The "Features" action will allow you to see the product features.
- The "Reviews" action will allow you to see the product reviews.
- The "Attributes" action will allow you to see the product attributes.
- The "Buy Now" action will purchase the product.
- Other actions like "navy", "red", "small", "x-large" are attributes of the product to be selected. Once an attribute is selected, the purchased product will have this attribute.

Note that if you are on the same product page as the previous time step, you only need to click on each product attribute like "navy", "red", "small", "x-large" once, unless you want to choose a different product feature.
"""

choice_prompt = lambda observation, action_history: f"""The current web page is summarized as follows: 

{observation}

Here are the actions you have taken:

{action_history}

Please propose an action to take and return as a JSON. 
"""

search_prompt = lambda observation, search_history: f"""The current web page is summarized as follows: 

{observation}

Here are the search queries you have tried:

{search_history}

Please propose a search query and return as a JSON. DO NOT include price in your search query.
"""


def create_pydantic_class(class_name: str, field_name: str, allowed_strings: list[str]):
    # Create a Literal type from the allowed strings
    literal_type = Union[tuple(Literal[s] for s in allowed_strings)]
    
    # Dynamically create a Pydantic model
    return create_model(class_name, **{field_name: (literal_type, ...)})


class Answer(BaseModel):
    query: str


class Agent:
    """llm agent wrapper"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.search_history = []
        self.action_history = []

    def search_action(self, observation: str):
        _search_history = "[" + ", ".join(self.search_history) + "]"
        cur_prompt = search_prompt(observation, _search_history)
        response = client.inference.chat_completion(
            model_id=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": cur_prompt},
            ],
            stream=False,
            response_format={
                "type": "json_schema",
                "json_schema": Answer.model_json_schema(),
            },
        )
        response_message = response.completion_message.content
        response_object = Answer.model_validate_json(response_message)

        # pprint(response_message)

        # compose valid action
        action = f"search[{response_object.query}]"

        self.search_history.append(action)
        return action
    
    def choice_step(self, observation: str, available_actions: dict):
        ActionClass = create_pydantic_class("ClickAction", "action", available_actions["clickables"])

        _action_history = "[" + ", ".join(self.action_history) + "]"
        cur_prompt = choice_prompt(observation, _action_history)
        response = client.inference.chat_completion(
            model_id=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": action_instruction_prompt},
                {"role": "user", "content": cur_prompt},
            ],
            stream=False,
            response_format={
                "type": "json_schema",
                "json_schema": ActionClass.model_json_schema(),
            },
        )
        response_message = response.completion_message.content
        response_object = ActionClass.model_validate_json(response_message)

        # compose valid action
        action = f"click[{response_object.action}]"

        # if "click[red]" in self.action_history and action == "click[red]":
        #     import pdb
        #     pdb.set_trace()

        self.action_history.append(action)
        return action
    
    def forward(self, observation: str, available_actions: dict):
        action = self.choice_step(observation, available_actions)
        
        # maybe search
        if action.startswith("click[Search]") and available_actions["has_search_bar"]:
            pprint("decided to search")
            action = self.search_action(observation)
        return action

if __name__ == '__main__':
    env = WebAgentSiteEnv(
        observation_mode='text', 
        render=True, 
        pause=2., 
        num_products=DEBUG_PROD_SIZE,
        session_id=1,
    )

    max_steps = 12
    max_eps = 3

    global_step = 0
    global_eps = 0
    
    try:
        # policy = RandomPolicy()
        policy = Agent()

        observation = env.observation
        while True:
            print(observation)
            available_actions = env.get_available_actions()
            # print('Available actions:', available_actions)
            action = policy.forward(observation, available_actions)
            observation, reward, done, info = env.step(action)
            
            print(f'Taking action "{escape(action)}" -> Reward = {reward}')

            # termination condition
            if done or (global_step + 1) >= max_steps:
                global_eps += 1

                if (global_eps + 1) >= max_eps:
                    break
                env.reset()
                policy.reset()
                
                # break

            global_step += 1

            # if global_step >= max_steps:
            #     break

    finally:
        env.close()