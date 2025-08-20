from env.simulator import Simulator
from omegaconf import OmegaConf
import os
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI
from env.object_memory import ObjectMemory
from env.executor import Executor
import json
from internvl2 import InternVL2_VQA
import time
import yaml


with open("config/api.yaml") as f:
    data = yaml.safe_load(f)
API_KEY = data["API_KEY"]
REGION = data["REGION"]
MODEL = data["MODEL"]
API_BASE = data["API_BASE"]
ENDPOINT = f"{API_BASE}/{REGION}"

os.environ["AZURE_OPENAI_API_KEY"] = API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = ENDPOINT


if __name__ == "__main__":
    episode_dir="data/scenes/home-robot-remake/episodes/104862660_172226844_new/layout_0"
    with open(os.path.join(episode_dir, "episode.json")) as f:
        episode = json.load(f)
    sim_settings = OmegaConf.load('config/default_sim_config.yaml') #default simulator settings
    sim_settings["scene_dataset_config_file"] = 'data/scenes/home-robot-remake/hssd-hab-uncluttered.scene_dataset_config.json'
    sim_settings["scene"] = episode["scene"]
    default_agent_settings = OmegaConf.load('config/default_agent_config.yaml')
    agents_settings = [default_agent_settings]
    sim = Simulator(sim_settings=sim_settings, agents_settings=agents_settings)
    sim.get_articulated_object_rotate_angle()
    sim.update_articulated_id_mapping()
    sim.recompute_navmesh()
    agent_position = episode["agent_position"]
    sim.initialize_agent(agent_id=0, position=agent_position)
    for obj_config_file in episode["objects"]:
        obj_translation = episode["objects"][obj_config_file]
        sim.load_object(object_config_path=obj_config_file, translation=obj_translation)
    target_island = episode["island_index"]
    
    with open("data/objects/pickable_objects.json") as f:
        pickable = json.load(f)
    with open("data/objects/unpickable_objects.json") as f:
        unpickable = json.load(f)
    with open("data/objects/rigid_receptacles.json") as f:
        receptacle = json.load(f)
    with open("data/objects/articulated_receptacles.json") as f:
        articulated_receptacle = json.load(f)
    with open("data/objects/receptacle2relation.json") as f:
        category2relation = json.load(f)

    # image_saving_path = os.path.join(episode_dir, "user_images")
    # if not os.path.exists(image_saving_path):
    #     os.makedirs(image_saving_path)
    object_memory_path = "res/object_memory"
    episode_history_path = "res/episode"
    object_memory = ObjectMemory(sim, pickable, unpickable, receptacle, articulated_receptacle, category2relation, save_path=object_memory_path)
    hfov = default_agent_settings["hfov"]
    executor = Executor(sim=sim, object_memory=object_memory, hfov=hfov, fps=60, island_index=target_island, visualize=True, plot_text=True, save_dir=episode_history_path)
    executor.execute_steps(['no_op' for i in range(10)])
    #executor.exhaustive_exploration()

    def remove_quotes(s):
        return s.strip('"').strip("'")
    cnt = 0


    @tool
    def GOTO(target: str) -> str:
        """Go to the target receptacle or object and look at it."""
        global cnt
        cnt += 1
        target = remove_quotes(target)
        res = executor.goto(target)
        return res

    @tool
    def OPEN(receptacle: str) -> str:
        """Open an articulated receptacle in view."""
        global cnt
        cnt += 1
        receptacle = remove_quotes(receptacle)
        res = executor.open(receptacle)
        return res

    @tool
    def CLOSE(receptacle: str) -> str:
        """Close an articulated receptacle in view."""
        global cnt
        cnt += 1
        receptacle = remove_quotes(receptacle)
        res = executor.close(receptacle)
        return res
    
    @tool
    def PICK(object: str) -> str:
        """Pick an object in view."""
        global cnt
        cnt += 1
        object = remove_quotes(object)
        res = executor.pick(object)
        return res
    
    @tool
    def PLACE(receptacle: str) -> str:
        """Place the inventory object in/on a receptacle in view."""
        global cnt
        cnt += 1
        receptacle = remove_quotes(receptacle)
        res = executor.place(receptacle)
        return res


    @tool
    def SEARCH(object: str) -> str:
        """Search for the target object by navigating in the appartment until the target object is found. It will not check the articulated receptacle such as fridge and microwave."""
        global cnt
        cnt += 1
        object = remove_quotes(object)
        res = executor.search(object)
        return res
    
    @tool
    def OBJECT_QUERY(object: str) -> str:
        """Query where is the object that has been explored."""
        global cnt
        cnt += 1
        object = remove_quotes(object)
        res = object_memory.query_object_state(object)
        return res

    @tool
    def FRAME_VQA(question: str) -> str:
        """Query where is the object that has been explored."""
        global cnt
        cnt += 1
        question = remove_quotes(object)
        res = object_memory.query_object_state(object)
        return res


    @tool
    def retrieve_objects_by_appearance(description):
        """return the top-10 candidate object IDs from the object memory that satisfy the appearance description (e.g. brown chair)."""
        candidate_dict = object_memory.retrieve_objects_by_appearence(description=description)
        final_objects = []
        for name in candidate_dict:
            image_path = os.path.join("object_memory", f"{name}.jpg")
            answer = InternVL2_VQA(f"Whether the object in the image satisfies '{description}'? Only output Yes or No.", image_path)
            if 'Yes' in answer or 'yes' in answer:
                # final_objects[idx] = ""candidate_dict[idx]
                final_objects.append(name)
                # bbox_path = os.path.join(database.base_dir, str(idx), "3dbbox_in_frame.jpg")
                # object_description = InternVL2_VQA("Indentify and briefly describe the object in the bounding box.", bbox_path)
                # #image_path = os.path.join(database.base_dir, str(idx), "3dbbox_in_frame.jpg")
                # final_objects[idx] = object_description 
        return f"The objects that satisfy '{description}' are {str(final_objects)}"


    @tool
    def retrieve_objects_by_environment(description):
        """return the top-10 candidate object IDs from the object memory that satisfy the environment description (e.g. blue wall)."""
        candidate_dict = object_memory.retrieve_objects_by_environment(description=description)
        final_objects = []
        for name in candidate_dict:
            image_path = os.path.join("object_memory", f"{name}_in_frame.jpg")
            answer = InternVL2_VQA(f"Whether the image is about {description}? Only output Yes or No.", image_path)
            if 'Yes' in answer or 'yes' in answer:
                # final_objects[idx] = ""candidate_dict[idx]
                final_objects.append(name)
                # bbox_path = os.path.join(database.base_dir, str(idx), "3dbbox_in_frame.jpg")
                # object_description = InternVL2_VQA("Indentify and briefly describe the object in the bounding box.", bbox_path)
                # #image_path = os.path.join(database.base_dir, str(idx), "3dbbox_in_frame.jpg")
                # final_objects[idx] = object_description 
        return f"The objects that satisfy '{description}' are {str(final_objects)}"
    
    @tool
    def GET_EXPLORED_OBJECTS(object: str) -> str:
        """Get all the objects that have been explored."""
        global cnt
        cnt += 1
        object = remove_quotes(object)
        res = object_memory.get_explored_objects(object)
        return res
        
    executor.execute_steps(action_list=['no_op'])

    class UserAgent:
        def __init__(self, system_prompt):
            self.client = AzureOpenAI(
                api_version="2024-02-01",
            )
            self.system_prompt = system_prompt
            self.messages = [{"role": "system", "content": self.system_prompt}]
            
        def start(self):
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
            )
            content = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": content})
            return content


        def chat(self, input):
            self.messages.append({"role": "user", "content": input})
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
            )
            content = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": content})
            print(self.messages)
            return content


    receptacles = []
    objects = []
    for id in object_memory.articulated_receptacle_ids:
        receptacles.append(object_memory.id2object[id].name)
    for id in object_memory.receptacle_ids:
        receptacles.append(object_memory.id2object[id].name)

    for id in object_memory.pickable_ids:
        objects.append(object_memory.id2object[id].name)
    with open("prompt/user_prompt.txt") as f:
        system_prompt = f.read()
    #obtained_by_navigation
    #objects = ['laptop_1', 'bundt_pan_1', 'butter_dish_1', 'glass_2', 'knife_1', 'toy_animal_1', 'hand_towel_1', 'sushi_mat_1', 'board_game_1', 'kettle_1', 'laptop_stand_1', 'casserole_1', 'glass_1', 'multiport_hub_1', 'cake_pan_1', 'jug_1', 'butter_dish_2']
    system_prompt = system_prompt.replace("{object_list}", str(objects))
    system_prompt = system_prompt.replace("{recep_list}", str(receptacles))
    print(system_prompt)
    user_agent = UserAgent(system_prompt=system_prompt)

    @tool
    def CHAT(content: str) -> str:
        """Communicate with the user."""
        user_reply = user_agent.chat(content)
        return user_reply
    
    prompt = hub.pull("hwchase17/react")
    with open('prompt/assistant_prompt.txt') as f:
        t = f.read()
    t = t.replace("{recep}", str(receptacles))
    print(t)
    prompt.template = t
    llm = AzureChatOpenAI(temperature=0, openai_api_version='2024-02-01', azure_deployment=MODEL, streaming=False)
    #tools = [CHAT, GOTO, OPEN, CLOSE, PICK, PLACE, SEARCH, OBJECT_QUERY, GET_EXPLORED_OBJECTS]
    tools = [CHAT, GOTO, OPEN, CLOSE, PICK, PLACE, SEARCH, retrieve_objects_by_appearance, retrieve_objects_by_environment]
    robot = create_react_agent(llm, tools, prompt)
    robot_executor = AgentExecutor(agent=robot, tools=tools, verbose=True, max_iterations=100)
    # print(GOTO("microwave_1"))
    # print(OPEN("microwave_1"))
    # print(GOTO("fridge_1"))
    # print(OPEN("fridge_1"))

    initial_task = user_agent.start()
    print(initial_task)
    robot_executor.invoke({"input": initial_task})
    end_time = time.time()
    print("tool_cnt: ", cnt)
    print("action_cnt: ", executor.step_cnt)