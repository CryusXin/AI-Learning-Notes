import json

from openai.lib.azure import AzureOpenAI

api_key = "2640762c94384b6a98436a8048d84670"
def get_completion(messages):
    '''封装 Azure openai 接口'''
    api_base = 'https://micker-gpt-4-tubro.openai.azure.com/'  # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
    deployment_name = 'gpt-4-turbo'
    api_version = '2023-07-01-preview'  # this might change in the future

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=api_base
    )
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
        tools=[{
            "type": "function",
            "function": {
                "name": "add_contact",
                "description": "添加联系人",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "联系人姓名"
                        },
                        "address": {
                            "type": "string",
                            "description": "联系人地址"
                        },
                        "tel": {
                            "type": "string",
                            "description": "联系人电话"
                        },
                    }
                }
            }
        }],
    )
    return response.choices[0].message

def print_json(data):
    """
    打印参数。如果参数是有结构的（如字典或列表），则以格式化的 JSON 形式打印；
    否则，直接打印该值。
    """
    if hasattr(data, 'model_dump_json'):
        data = json.loads(data.model_dump_json())

    if (isinstance(data, (list, dict))):
        print(json.dumps(
            data,
            indent=4,
            ensure_ascii=False
        ))
    else:
        print(data)

if __name__ == '__main__':
    prompt = "帮我寄给xxx，地址是上海市，电话1555555555。"
    messages = [
        {"role": "system", "content": "你是一个联系人录入员。"},
        {"role": "user", "content": prompt}
    ]
    response = get_completion(messages)
    print("====GPT回复====")
    print_json(response)
    args = json.loads(response.tool_calls[0].function.arguments)
    print("====函数参数====")
    print_json(args)