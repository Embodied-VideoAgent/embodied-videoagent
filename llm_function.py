from openai import AzureOpenAI
import yaml


with open("config/api.yaml") as f:
    data = yaml.safe_load(f)
API_KEY = data["API_KEY"]
REGION = data["REGION"]
MODEL = data["MODEL"]
API_BASE = data["API_BASE"]
ENDPOINT = f"{API_BASE}/{REGION}"
print(ENDPOINT)


client = AzureOpenAI(
    api_key=API_KEY,
    api_version="2024-02-01",
    azure_endpoint=ENDPOINT,
)



def llm_select_object(target, obj_name_list):
    if len(obj_name_list) == 0:
        return None
    prompt = f"Retrieve an object that is {target} from {obj_name_list}. Output the retrieved object with no additional output. If there is no qualified object output None with no additional output."
    #print(prompt)
    print(prompt)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],)
    content = response.choices[0].message.content
    #print(content)
    if content == 'None':
        return None
    return content


if __name__ == "__main__":
    target = 'orange'
    obj_name_list = ["apple_1", "apple_2", "kettle_1", "knife_4", "pan_1", "bundt_1", "knife_2"]
    t = llm_select_object(target, obj_name_list)
    print(t)